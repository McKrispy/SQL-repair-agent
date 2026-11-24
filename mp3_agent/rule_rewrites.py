"""Six approved local rewrite families, with explicit trigger evidence."""
import re

from sqlglot import exp

from mp3_agent.rule_support import (COMPARISONS, count_comparisons, local_columns, parse,
    root_select, scopes, signature, singular, text, unalias)


def replace_at(tree, old, new):
    path = []
    cursor = old
    while cursor.parent is not None:
        path.append((cursor.arg_key, cursor.index))
        cursor = cursor.parent
    copied = tree.copy()
    if not path:
        return new.copy()
    target = copied
    for key, index in reversed(path):
        target = target.args[key] if index is None else target.args[key][index]
    target.replace(new.copy())
    return copied


def proposal(tree, family, evidence, risks=()):
    return {"sql": text(tree), "rule": family, "evidence": evidence, "risks": list(risks)}


def output_rules(tree, constraints, schema):
    select = root_select(tree)
    if select is None or any(h is None for h in constraints.headers):
        return
    wanted = [signature(h) for h in constraints.headers]
    offered = [signature(e) for e in select.expressions]
    if len(set(wanted)) != len(wanted) or any(offered.count(k) != 1 for k in wanted):
        return
    positions = [offered.index(k) for k in wanted]
    if positions == list(range(len(offered))):
        return
    corrected = select.copy()
    removed = [e for i, e in enumerate(select.expressions) if i not in positions]
    corrected.set("expressions", [select.expressions[i].copy() for i in positions])
    order = corrected.args.get("order")
    if order:
        aliases = {e.alias.lower(): e.this for e in removed if isinstance(e, exp.Alias)
                   and isinstance(e.this, exp.Count)}
        for col in list(order.find_all(exp.Column)):
            if not col.table and col.name.lower() in aliases:
                col.replace(aliases[col.name.lower()].copy())
        # Positional ORDER BY must retain the identity of its original projection.
        for ordered in order.expressions:
            value = ordered.this
            if isinstance(value, exp.Literal) and not value.is_string and value.this.isdigit():
                index = int(value.this) - 1
                if 0 <= index < len(select.expressions):
                    ordered.set("this", unalias(select.expressions[index]).copy())
    yield proposal(corrected, "output", {"example_columns": constraints.labels, "projection_positions": positions},
                   ["removed_projection"] if removed else [])


def binding_rules(tree, constraints, schema):
    for scope in scopes(tree):
        sources = schema.sources(scope)
        select = scope.expression
        if not isinstance(select, exp.Select):
            continue
        aliases = {e.alias.lower() for e in select.expressions if e.alias}
        for col in scope.columns:
            name, qualifier = col.name.lower(), col.table.lower()
            if col.is_star or col.find_ancestor(exp.Select) is not select:
                continue
            owners = [alias for alias, (_, columns) in sources.items() if name in columns]
            target = None
            evidence = {}
            if qualifier in sources and name not in sources[qualifier][1] and len(owners) == 1:
                target = owners[0]
                evidence = {"missing_column": text(col), "unique_introduced_owner": target}
            elif not qualifier and len(owners) > 1 and name not in aliases:
                # Only inner-join equalities establish interchangeable ownership.
                equalities = []
                if all(not j.args.get("side") and j.args.get("kind") != "CROSS" for j in select.args.get("joins", [])):
                    for join in select.args.get("joins", []):
                        on = join.args.get("on")
                        for eq in on.find_all(exp.EQ) if on is not None else []:
                            if all(isinstance(n, exp.Column) and n.name.lower() == name for n in (eq.this, eq.expression)):
                                equalities.append({eq.this.table.lower(), eq.expression.table.lower()})
                if len(owners) == 2 and set(owners) in equalities:
                    target = owners[0]
                    evidence = {"ambiguous_column": text(col), "inner_equijoin_equivalence": owners}
                else:
                    # A qualified projection of this exact field can establish the intended owner.
                    projected = {c.table.lower() for e in select.expressions for c in local_columns(e, select)
                                 if c.name.lower() == name and c.table.lower() in owners}
                    if len(projected) == 1 and col.find_ancestor(exp.Order, exp.Group) is not None:
                        target = next(iter(projected))
                        evidence = {"projection_owner": target, "column": name}
            if target:
                corrected = col.copy()
                corrected.set("table", exp.to_identifier(target))
                yield proposal(replace_at(tree, col, corrected), "binding", evidence)


def joins_grain_rules(tree, constraints, schema):
    for scope in scopes(tree):
        select = scope.expression
        if not isinstance(select, exp.Select):
            continue
        sources = schema.sources(scope)
        real = {alias: table for alias, (table, _) in sources.items() if table is not None}
        joins = select.args.get("joins", [])
        from_clause = select.args.get("from")
        if len(real) == 2 and len(sources) == 2 and len(joins) == 1 and from_clause:
            left_alias = from_clause.this.alias_or_name.lower()
            right_alias = joins[0].this.alias_or_name.lower()
            old_on = joins[0].args.get("on")
            if (left_alias in real and right_alias in real and isinstance(old_on, exp.EQ)
                    and all(isinstance(n, exp.Column) for n in (old_on.this, old_on.expression))
                    and not joins[0].args.get("side")):
                path = schema.unique_path(real[left_alias], real[right_alias])
                if path:
                    aliases = {real[left_alias]: left_alias, real[right_alias]: right_alias}
                    for i, edge in enumerate(path[:-1]):
                        if edge[2] not in aliases:
                            candidate_alias = f"_b06_bridge_{i}"
                            while candidate_alias in sources:
                                candidate_alias += "_"
                            aliases[edge[2]] = candidate_alias
                    new_joins = []
                    for edge in path:
                        table = schema.tables[edge[2]]["name"]
                        alias = aliases[edge[2]]
                        source = joins[0].this.copy() if edge[2] == real[right_alias] else exp.Table(
                            this=exp.to_identifier(table), alias=exp.TableAlias(this=exp.to_identifier(alias)))
                        on = exp.EQ(this=exp.column(edge[1], table=aliases[edge[0]]),
                                    expression=exp.column(edge[3], table=alias))
                        new_joins.append(exp.Join(this=source, on=on))
                    corrected = select.copy()
                    corrected.set("joins", new_joins)
                    # Do not emit a different serialization of the identical FK predicate.
                    expected = new_joins[-1].args["on"]
                    pairs = lambda eq: {text(eq.this).lower(), text(eq.expression).lower()}
                    if len(path) > 1 or pairs(old_on) != pairs(expected):
                        yield proposal(replace_at(tree, select, corrected), "joins_grain",
                                       {"unique_declared_fk_path": path, "replaced_on": text(old_on)},
                                       ["changed_join_coverage"])
        if not joins or len(real) != len(sources) or not from_clause:
            continue
        if not any(isinstance(e, exp.AggFunc) for projection in select.expressions for e in projection.walk()):
            continue
        non_join_parts = [*select.expressions] + [select.args[k] for k in ("where", "group", "having", "order") if select.args.get(k)]
        columns = [c for part in non_join_parts for c in local_columns(part, select)]
        owners = set()
        unresolved = False
        for col in columns:
            possible = [a for a, (_, cs) in sources.items() if col.name.lower() in cs]
            owner = col.table.lower() if col.table.lower() in real else possible[0] if len(possible) == 1 else None
            if owner:
                owners.add(owner)
            else:
                unresolved = True
        if unresolved or len(owners) != 1 or any(isinstance(n, exp.Subquery) for p in non_join_parts for n in p.walk()):
            continue
        # Removing a table with literal predicates in ON would discard a filter.
        if any(j.args.get("on") is None or any(isinstance(n, exp.Literal) for n in j.args["on"].walk()) for j in joins):
            continue
        owner = next(iter(owners))
        source = next((n for n in [from_clause.this, *(j.this for j in joins)] if n.alias_or_name.lower() == owner), None)
        if source is not None:
            corrected = select.copy()
            corrected.set("from", exp.From(this=source.copy()))
            corrected.set("joins", [])
            yield proposal(replace_at(tree, select, corrected), "joins_grain",
                           {"aggregate_owner": owner, "removed_joins": [text(j) for j in joins]},
                           ["changed_join_coverage", "removed_possible_fanout"])


def quantity_group_rules(tree, constraints, schema):
    target = constraints.quantity
    if target:
        for node in count_comparisons(tree, target):
            if type(node) is not target["operator"]:
                replacement = target["operator"](this=node.this.copy(), expression=node.expression.copy())
                yield proposal(replace_at(tree, node, replacement), "quantity_group",
                               {"question_boundary": target["evidence"], "old": text(node), "new": text(replacement)})
    question = constraints.question.lower()
    for scope in scopes(tree):
        select = scope.expression
        sources = schema.sources(scope)
        if not isinstance(select, exp.Select):
            continue
        if (select is tree and re.search(r"\bolder than (?:the )?average\b", question)
                and len(sources) == 1):
            alias, (table, columns) = next(iter(sources.items()))
            ages = [name for name in columns if name in {"age", "ages"}]
            if table and len(ages) == 1:
                age = ages[0]
                condition = exp.GT(this=exp.column(age, table=alias), expression=exp.Subquery(this=exp.select(
                    exp.Avg(this=exp.column(age))).from_(exp.to_table(schema.tables[table]["name"]))))
                if not any(signature(n) == signature(condition) for n in select.find_all(exp.GT)):
                    corrected = select.copy()
                    corrected = corrected.where(condition, append=True)
                    yield proposal(replace_at(tree, select, corrected), "quantity_group",
                                   {"question": "older than average", "unique_age_owner": table}, ["added_filter"])
        # "largest <grouping entity>" is only actionable when MAX targets that entity's PK.
        match = re.search(r"\blargest\s+([a-z_]+)", question)
        if match:
            noun = singular(match[1])
            for maximum in select.find_all(exp.Max):
                if maximum.find_ancestor(exp.Select) is not select or not isinstance(maximum.this, exp.Column):
                    continue
                col = maximum.this
                candidates = [(a, t) for a, (t, cs) in sources.items() if t and noun in singular(t).split()
                              and col.name.lower() in cs and (not col.table or col.table.lower() == a)
                              and schema.columns[t][col.name.lower()].get("pk")]
                if len(candidates) != 1:
                    continue
                alias, table = candidates[0]
                related = [(t, fk) for t, meta in schema.tables.items() for fk in meta.get("foreign_keys", [])
                           if fk["table"].lower() == table and fk["to"].lower() == col.name.lower()]
                if len(related) != 1:
                    continue
                relation, fk = related[0]
                sub = (exp.select(exp.column(fk["from"])).from_(exp.to_table(schema.tables[relation]["name"]))
                       .group_by(exp.column(fk["from"])).order_by(exp.Ordered(this=exp.Count(this=exp.Star()), desc=True)).limit(1))
                # Replace the scalar MAX subquery as a whole, retaining outer equality.
                parent_select = maximum.find_ancestor(exp.Select)
                if parent_select is select and len(select.expressions) == 1 and select.parent and isinstance(select.parent, exp.Subquery):
                    yield proposal(replace_at(tree, select, sub), "quantity_group",
                                   {"largest_group": noun, "group_relation": relation, "old": text(select)},
                                   ["changed_grouping", "top_one_ties"])


def negative_set_rules(tree, constraints, schema):
    question = constraints.question
    negative = bool(re.search(r"\bnever\b|\bnot\b.+\bany\b", question, re.I))
    both = bool(re.search(r"\bboth\b", question, re.I))
    if not negative and not both:
        return
    for scope in scopes(tree):
        select = scope.expression
        if not isinstance(select, exp.Select) or not select.args.get("where") or select.args.get("group"):
            continue
        joins = select.args.get("joins", [])
        sources = schema.sources(scope)
        if len(joins) != 1 or len(sources) != 2 or not select.args.get("from"):
            continue
        relation = joins[0].this
        if not isinstance(relation, exp.Table) or joins[0].args.get("side"):
            continue
        relation_alias = relation.alias_or_name.lower()
        base_alias = select.args["from"].this.alias_or_name.lower()
        on = joins[0].args.get("on")
        if not isinstance(on, exp.EQ) or not all(isinstance(c, exp.Column) for c in (on.this, on.expression)):
            continue
        if {on.this.table.lower(), on.expression.table.lower()} != {base_alias, relation_alias}:
            continue
        if any(c.table.lower() != base_alias for e in select.expressions for c in local_columns(e, select)):
            continue
        where = select.args["where"].this
        terms = list(where.flatten()) if isinstance(where, exp.And) else [where]
        related = [term for term in terms if any(c.table.lower() == relation_alias for c in local_columns(term, select))]
        other = [term for term in terms if term not in related]
        if len(related) != 1:
            continue
        predicate = related[0]

        def exists(condition):
            sub = exp.select("1").from_(relation.copy()).where(exp.and_(on.copy(), condition.copy()))
            return exp.Exists(this=sub)

        replacement = None
        evidence = {}
        if negative and isinstance(predicate, COMPARISONS) and isinstance(predicate.this, exp.Column) and isinstance(predicate.expression, exp.Literal):
            opposite = {exp.NEQ: exp.EQ}
            number = re.escape(str(predicate.expression.this))
            if re.search(r"\bnever\b.*\bmore than\s+" + number + r"\b", question, re.I):
                opposite[exp.LTE] = exp.GT
            if re.search(r"\bnever\b.*\bless than\s+" + number + r"\b", question, re.I):
                opposite[exp.GTE] = exp.LT
            if type(predicate) in opposite:
                excluded = opposite[type(predicate)](this=predicate.this.copy(), expression=predicate.expression.copy())
                replacement = exp.Not(this=exists(excluded))
                evidence = {"entity_negation": text(predicate), "correlation": text(on)}
        values = []
        value_column = None
        if both and isinstance(predicate, exp.In) and isinstance(predicate.this, exp.Column):
            values, value_column = predicate.expressions, predicate.this
        elif both and isinstance(predicate, exp.Or):
            parts = list(predicate.flatten())
            if len(parts) == 2 and all(isinstance(p, exp.EQ) and isinstance(p.this, exp.Column) and isinstance(p.expression, exp.Literal) for p in parts):
                if signature(parts[0].this) == signature(parts[1].this) and parts[0].this.table == parts[1].this.table:
                    values, value_column = [p.expression for p in parts], parts[0].this
        if values:
            if len(values) == 2 and all(isinstance(v, exp.Literal) for v in values):
                # Literal strings/numbers must also occur explicitly in the question.
                if all(str(v.this).lower() in question.lower() for v in values):
                    replacement = exp.and_(*(exists(exp.EQ(this=value_column.copy(), expression=v.copy())) for v in values))
                    evidence = {"both_values": [v.this for v in values], "correlation": text(on)}
        if replacement is not None:
            corrected = select.copy()
            corrected.set("joins", [])
            corrected.set("where", exp.Where(this=exp.and_(*(t.copy() for t in other), replacement)))
            yield proposal(replace_at(tree, select, corrected), "negative_set", evidence,
                           ["changed_join_coverage", "changed_multiplicity", "null_semantics"])


def literal_format_rules(tree, constraints, schema):
    question = constraints.question
    for node in tree.walk():
        if not isinstance(node, exp.EQ) or not isinstance(node.this, exp.Column) or not isinstance(node.expression, exp.Literal) or not node.expression.is_string:
            continue
        old = node.expression.this
        # Restore a full named entity only when the question explicitly spells it out.
        match = re.search(r"\b(Team\s+" + re.escape(old) + r")\b", question, re.I)
        if match and not old.lower().startswith("team "):
            corrected = exp.EQ(this=node.this.copy(), expression=exp.Literal.string(match[1]))
            yield proposal(replace_at(tree, node, corrected), "literal_format",
                           {"question_named_entity": match[1], "old_literal": old})


RULES = (output_rules, binding_rules, joins_grain_rules, quantity_group_rules,
         negative_set_rules, literal_format_rules)


def rewrites(tree, constraints, schema):
    for rule in RULES:
        yield from rule(tree, constraints, schema)


def probe_requests(candidate, constraints, schema):
    """A diagnosis costs one execute, and always reserves one subsequent check."""
    feedback = candidate.get("feedback")
    if not feedback or feedback.get("matches_example"):
        return []
    tree = parse(candidate["sql"])
    suspicious = []
    for scope in scopes(tree):
        sources = schema.sources(scope)
        for node in scope.expression.find_all(exp.EQ):
            if node.find_ancestor(exp.Select) is not scope.expression:
                continue
            if not isinstance(node.this, exp.Column) or not isinstance(node.expression, exp.Literal):
                continue
            col, value = node.this, node.expression
            owners = [(a, t) for a, (t, cs) in sources.items() if t and col.name.lower() in cs
                      and (not col.table or col.table.lower() == a)]
            if len(owners) != 1:
                continue
            _, table = owners[0]
            column = schema.columns[table][col.name.lower()]
            duration = (not value.is_string and re.search(r"\bminutes?\b", constraints.question, re.I)
                        and re.search(r"duration|length|time", col.name, re.I)
                        and "TEXT" in column["type"].upper())
            if value.is_string or duration:
                suspicious.append((node, table, column["name"], value.this, bool(duration)))
    if len(suspicious) != 1:
        return []
    node, table, column, old, duration = suspicious[0]
    actual = exp.column(column)
    if duration:
        condition = exp.Like(this=actual.copy(), expression=exp.Literal.string(str(old) + ":%"))
    else:
        # SQLite LOWER is ASCII based. Remove only ASCII whitespace in both operands.
        def normalize(n):
            n = exp.Lower(this=n)
            for whitespace in (" ", "\t", "\r", "\n"):
                n = exp.Anonymous(this="REPLACE", expressions=[n, exp.Literal.string(whitespace), exp.Literal.string("")])
            return n
        condition = exp.EQ(this=normalize(actual.copy()), expression=normalize(exp.Literal.string(str(old))))
    query = exp.select(actual).distinct().from_(exp.to_table(schema.tables[table]["name"])).where(condition)
    return [{"identity": [table, column.lower(), str(old)], "sql": text(query),
             "parent_key": candidate["key"], "old_predicate": text(node), "duration": duration,
             "old_literal": old}]


def apply_probe(parent, request, feedback):
    if not feedback.get("ok") or feedback.get("truncated") or feedback.get("row_count") != 1:
        return None
    rows = feedback.get("rows", [])
    if len(rows) != 1 or len(rows[0]) != 1 or not isinstance(rows[0][0], str):
        return None
    actual = rows[0][0]
    tree = parse(parent["sql"])
    nodes = [n for n in tree.find_all(exp.EQ) if text(n) == request["old_predicate"]]
    if len(nodes) != 1:
        return None
    node = nodes[0]
    if request["duration"]:
        if not re.fullmatch(re.escape(str(request["old_literal"])) + r":[0-5]\d", actual):
            return None
        replacement = exp.Like(this=node.this.copy(), expression=exp.Literal.string(str(request["old_literal"]) + ":%"))
    else:
        if actual == str(request["old_literal"]):
            return None
        replacement = exp.EQ(this=node.this.copy(), expression=exp.Literal.string(actual))
    return proposal(replace_at(tree, node, replacement), "literal_format",
                    {"paid_probe": request, "unique_actual_value": actual}, ["storage_format_assumption"])
