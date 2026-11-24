"""Pure AST helpers. Inputs are restricted to the current official observation."""
import re
from collections import defaultdict

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import Scope, traverse_scope

FAMILIES = ("output", "binding", "joins_grain", "quantity_group", "negative_set", "literal_format")


def parse(sql):
    return sqlglot.parse_one(sql, read="sqlite")


def text(node):
    return node.sql(dialect="sqlite", comments=False)


def unalias(node):
    return node.this if isinstance(node, exp.Alias) else node


def signature(node):
    node = unalias(node).copy()
    for column in node.find_all(exp.Column):
        column.set("table", None)
        column.set("db", None)
        column.set("catalog", None)
    return node.sql(dialect="sqlite", normalize=True, comments=False)


def header_expression(label):
    """Only parse literal SQL-like labels; prose labels remain unknown."""
    if not isinstance(label, str) or not re.fullmatch(r"[\w.]+|(?:MIN|MAX|SUM|AVG|COUNT)\s*\(\s*(?:DISTINCT\s+)?[\w.*]+\s*\)", label.strip(), re.I):
        return None
    try:
        node = parse("SELECT " + label).expressions[0]
        return node if isinstance(node, (exp.Column, exp.AggFunc)) else None
    except (ValueError, sqlglot.errors.SqlglotError):
        return None


def root_select(tree):
    return tree if isinstance(tree, exp.Select) else None


def output_branches(tree):
    """2026-11-19 HKT: collect only the result-producing query branches.

    FROM/WHERE subqueries do not supply the outer projection. Compound queries
    share one output contract: a second branch must not earn an extra vote.
    This inspects the AST only and never expands stars against database rows.
    """
    if isinstance(tree, exp.Subquery):
        return output_branches(tree.this)
    if isinstance(tree, exp.Select):
        return [tree.expressions]
    if isinstance(tree, exp.SetOperation):
        left, right = output_branches(tree.this), output_branches(tree.expression)
        return left + right if left is not None and right is not None else None
    return None


def output_states(tree, headers):
    """Apply the same positional checks to SELECT and compound output.

    A compound field/aggregate is identifiable only when all its branches agree
    after stripping aliases/table qualifiers. Differing branch expressions are
    unknown, since labels from the first branch do not prove their semantics.
    """
    branches = output_branches(tree)
    if not branches:
        return "unknown", ["unknown"] * len(headers)
    widths = [None if any(e.is_star for e in branch) else len(branch) for branch in branches]
    arity = ("contradicted" if any(n is not None and n != len(headers) for n in widths)
             else "satisfied" if all(n is not None for n in widths) else "unknown")
    states = []
    for i, header in enumerate(headers):
        state = "unknown"
        if header is not None and all(n is not None for n in widths):
            if any(i >= len(branch) for branch in branches):
                state = "contradicted"
            else:
                nodes = [unalias(branch[i]) for branch in branches]
                if all(isinstance(n, (exp.Column, exp.AggFunc)) for n in nodes):
                    signatures = {signature(n) for n in nodes}
                    if len(signatures) == 1:
                        state = "satisfied" if signature(header) in signatures else "contradicted"
        states.append(state)
    return arity, states


class Schema:
    def __init__(self, tables):
        self.tables = {t["name"].lower(): t for t in tables}
        self.columns = {name: {c["name"].lower(): c for c in t["columns"]}
                        for name, t in self.tables.items()}
        self.edges = defaultdict(list)
        for name, table in self.tables.items():
            for fk in table.get("foreign_keys", []):
                other = fk["table"].lower()
                edge = (name, fk["from"].lower(), other, fk["to"].lower())
                if edge not in self.edges[name]:
                    self.edges[name].append(edge)
                    self.edges[other].append((other, edge[3], name, edge[1]))

    def sources(self, scope):
        result = {}
        try:
            selected = scope.selected_sources
        except sqlglot.errors.SqlglotError:
            # Duplicate aliases are an invalid/ambiguous input, not a reason to
            # crash the whole bank. No owner can safely be inferred in this scope.
            return result
        for alias, (_, source) in selected.items():
            if isinstance(source, exp.Table):
                table = source.name.lower()
                result[alias.lower()] = (table, set(self.columns.get(table, {})))
            elif isinstance(source, Scope):
                result[alias.lower()] = (None, {n.lower() for n in source.expression.named_selects if n != "*"})
        return result

    def unique_path(self, start, end):
        # Retain up to two simple paths; any second path makes the result ambiguous.
        found = []

        def visit(node, seen, path):
            if len(found) > 1:
                return
            if node == end:
                found.append(path)
                return
            for edge in self.edges.get(node, []):
                if edge[2] not in seen:
                    visit(edge[2], seen | {edge[2]}, path + [edge])

        visit(start, {start}, [])
        return found[0] if len(found) == 1 else None


def scopes(tree):
    try:
        return list(traverse_scope(tree))
    except (ValueError, sqlglot.errors.SqlglotError):
        return []


def local_columns(node, select):
    return [c for c in node.find_all(exp.Column) if c.find_ancestor(exp.Select) is select]


def singular(word):
    word = word.lower().replace("_", " ")
    if word.endswith("ies"):
        return word[:-3] + "y"
    return word[:-1] if word.endswith("s") and not word.endswith("ss") else word


NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
           "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def quantity(question):
    pattern = r"\b(exactly|at least|more than|at most|(?<!at )(?<!than )with|have|has|having)?\s*(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+([a-z]+)\b"
    matches = []
    for match in re.finditer(pattern, question.lower()):
        prefix, number, noun = match.groups()
        if noun in {"and", "or", "of", "in", "to", "than", "a", "the", "year", "years", "minute", "minutes"}:
            continue
        n = int(number) if number.isdigit() else NUMBERS[number]
        operator = {"at least": exp.GTE, "more than": exp.GT, "at most": exp.LTE}.get(prefix, exp.EQ)
        matches.append({"number": n, "noun": singular(noun), "operator": operator,
                        "evidence": match.group().strip()})
    return matches[0] if len(matches) == 1 else None


COMPARISONS = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)


def count_comparisons(tree, target):
    result = []
    for node in tree.walk():
        if not isinstance(node, COMPARISONS) or not isinstance(node.expression, exp.Literal) or node.expression.is_string:
            continue
        try:
            if float(node.expression.this) != target["number"]:
                continue
        except ValueError:
            continue
        left = node.this
        if isinstance(left, exp.Subquery):
            inner = left.this
            count = inner.expressions[0] if isinstance(inner, exp.Select) and len(inner.expressions) == 1 else None
            entity_tables = [t.name for t in inner.find_all(exp.Table)]
        else:
            count = left if isinstance(left, exp.Count) else None
            select = left.find_ancestor(exp.Select)
            entity_tables = [t.name for t in select.find_all(exp.Table)] if select else []
        if isinstance(unalias(count), exp.Count) and any(target["noun"] in singular(n).split() for n in entity_tables):
            result.append(node)
    return result


class Constraints:
    def __init__(self, observation):
        self.question = observation["task"]["question"]
        self.labels = observation["example"]["columns"]
        self.headers = [header_expression(label) for label in self.labels]
        self.quantity = quantity(self.question)
        self.definitions = [{"id": "output_count", "evidence": list(self.labels)}]
        self.definitions += [{"id": f"output_{i}", "evidence": label,
                              "recognizable": h is not None}
                             for i, (label, h) in enumerate(zip(self.labels, self.headers))]
        if self.quantity:
            self.definitions.append({"id": "count_boundary", "evidence": self.quantity["evidence"]})
        if re.search(r"\bnever\b|\bnot\b.+\bany\b", self.question, re.I):
            self.definitions.append({"id": "entity_negation", "evidence": self.question})
        if re.search(r"\bboth\b", self.question, re.I):
            self.definitions.append({"id": "entity_intersection", "evidence": self.question})

    def bind_schema(self, schema):
        """A SQL-looking output alias is still opaque if it names no real field."""
        names = {name for columns in schema.columns.values() for name in columns}
        for i, header in enumerate(self.headers):
            if header is not None and any(c.name.lower() not in names for c in header.find_all(exp.Column)):
                self.headers[i] = None
                definition = next(d for d in self.definitions if d["id"] == f"output_{i}")
                definition["recognizable"] = False
                definition["unknown_reason"] = "label_does_not_resolve_to_schema_field"

    def evaluate(self, tree):
        arity, states = output_states(tree, self.headers)
        result = [{"id": "output_count", "state": arity}]
        result += [{"id": f"output_{i}", "state": state} for i, state in enumerate(states)]
        if self.quantity:
            nodes = count_comparisons(tree, self.quantity)
            state = ("satisfied" if all(type(n) is self.quantity["operator"] for n in nodes)
                     else "contradicted") if nodes else "unknown"
            result.append({"id": "count_boundary", "state": state})
        for item in self.definitions:
            if item["id"] == "entity_negation":
                not_exists = any(isinstance(n.this, exp.Exists) for n in tree.find_all(exp.Not))
                # EXISTS without a supporting rewrite proof is not enough to certify the full question.
                state = "unknown"
                if not not_exists and any(isinstance(n, exp.NEQ) for n in tree.walk()):
                    state = "contradicted"
                result.append({"id": item["id"], "state": state})
            if item["id"] == "entity_intersection":
                result.append({"id": item["id"], "state": "unknown"})
        return result
