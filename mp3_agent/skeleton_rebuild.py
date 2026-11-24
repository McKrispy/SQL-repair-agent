"""Approved observation-only, bounded flat-query reconstruction hypotheses."""
from collections import deque, defaultdict

from sqlglot import exp
from sqlglot.errors import SqlglotError

from mp3_agent.rule_support import parse, text, unalias, header_expression, signature
from mp3_agent.semantic_facts import words


ROUTES = ('P', 'S', 'J')


def _parts(node):
    if node is None:
        return []
    if isinstance(node, exp.Paren):
        return _parts(node.this)
    if isinstance(node, exp.And):
        return _parts(node.this) + _parts(node.expression)
    return [node]


def _supported(tree):
    if not isinstance(tree, exp.Select):
        return 'root_not_flat_select'
    if any(isinstance(n, (exp.Subquery, exp.SetOperation, exp.Window)) or (isinstance(n, exp.Select) and n is not tree) for n in tree.walk()):
        return 'nested_set_or_window'
    if tree.args.get('with'):
        return 'cte'
    source = tree.args.get('from')
    tables = [source.this] if source else []
    for join in tree.args.get('joins', []):
        if join.args.get('side') or join.args.get('using') or join.args.get('method') or join.args.get('kind') not in (None, 'INNER'):
            return 'non_inner_or_using_join'
        tables.append(join.this)
    if not tables or any(not isinstance(t, exp.Table) for t in tables):
        return 'non_physical_source'
    if len({t.name.casefold() for t in tables}) != len(tables) or len({t.alias_or_name.casefold() for t in tables}) != len(tables):
        return 'duplicate_table_role'
    return None


def _sources(tree):
    nodes = [tree.args['from'].this, *(j.this for j in tree.args.get('joins', []))]
    return {n.alias_or_name.casefold(): n.name.casefold() for n in nodes}


def _physical_owner(col, sources, schema):
    name = col.name.casefold()
    if col.table:
        t = sources.get(col.table.casefold())
        return t if t in schema.columns and name in schema.columns[t] else None
    owners = {t for t in sources.values() if name in schema.columns.get(t, {})}
    return next(iter(owners)) if len(owners) == 1 else None


def _prepare(tree, schema):
    """Expand only unambiguous aliases/positions before changing projections."""
    s = tree.copy()
    sources = _sources(s)
    aliases = defaultdict(list)
    for e in s.expressions:
        if e.alias:
            aliases[e.alias.casefold()].append(unalias(e))
    for clause in ('group', 'order', 'having'):
        node = s.args.get(clause)
        if node is None:
            continue
        if clause in ('group', 'order'):
            for item in list(node.expressions):
                value = item.this if isinstance(item, exp.Ordered) else item
                if isinstance(value, exp.Literal) and not value.is_string and value.this.isdigit():
                    i = int(value.this) - 1
                    if not 0 <= i < len(s.expressions):
                        raise ValueError('invalid_positional_reference')
                    value.replace(unalias(s.expressions[i]).copy())
        for c in list(node.find_all(exp.Column)):
            if not c.table and c.name.casefold() in aliases:
                name = c.name.casefold()
                if len(aliases[name]) != 1 or any(name in schema.columns.get(t, {}) for t in sources.values()):
                    raise ValueError('ambiguous_projection_alias')
                c.replace(aliases[name][0].copy())
    return s


def _aggregate_key(node):
    node = node.copy()
    for identifier in node.find_all(exp.Identifier):
        identifier.set('this', identifier.name.casefold())
        identifier.set('quoted', False)
    return signature(node)


def _projection_plans(s, labels, schema):
    plans = [('source', [e.copy() for e in s.expressions])]
    all_names = {c for cs in schema.columns.values() for c in cs}
    aggregates = {_aggregate_key(n) for e in s.expressions for n in e.find_all(exp.AggFunc)}
    target = []
    for i, label in enumerate(labels):
        h = header_expression(label)
        # Output labels cannot replace an existing aggregate's operation, argument,
        # DISTINCT or counted entity. A genuinely missing output slot can add an
        # expression; an identifiable existing aggregate can still be reordered.
        if isinstance(h, exp.AggFunc) and aggregates and _aggregate_key(h) not in aggregates and i < len(s.expressions):
            h = None
        if h is not None and all(c.name.casefold() in all_names for c in h.find_all(exp.Column)):
            target.append(h)
            continue
        aliases = [e for e in s.expressions if e.alias and e.alias.casefold() == str(label).casefold()]
        if len(aliases) == 1:
            target.append(aliases[0].copy())
        elif len(labels) == len(s.expressions):
            target.append(s.expressions[i].copy())
        else:
            return plans
    if [text(e) for e in target] != [text(e) for e in s.expressions]:
        plans.append(('target', target))
    return plans


def _join_residual(s, sources, schema):
    residual = []
    replaced = []
    for join in s.args.get('joins', []):
        if join.args.get('on') is None:
            raise ValueError('join_without_on')
        for p in _parts(join.args['on']):
            if isinstance(p, exp.EQ) and all(isinstance(x, exp.Column) for x in (p.this, p.expression)):
                # Qualifiers still identify roles when a source join key is nonexistent.
                a = sources.get(p.this.table.casefold()) or _physical_owner(p.this, sources, schema)
                b = sources.get(p.expression.table.casefold()) or _physical_owner(p.expression, sources, schema)
                if a and b and a != b:
                    replaced.append(text(p))
                    continue
            residual.append(p.copy())
    return residual, replaced


def _key(c):
    return (c.table.casefold(), c.name.casefold())


def _bindings(s, projection, residual, schema, question, count_table, cfg, events):
    sources = _sources(s)
    requirements = {}
    outside = [s.args[k] for k in ('where', 'group', 'having', 'order') if s.args.get(k)] + residual
    for nodes, nonprojection in ((projection, False), (outside, True)):
        for part in nodes:
            for c in part.find_all(exp.Column):
                if c.is_star:
                    raise ValueError('unexpanded_star')
                k = _key(c)
                req = requirements.setdefault(k, {'col': c, 'locked': None})
                if nonprojection:
                    owner = _physical_owner(c, sources, schema)
                    if owner:
                        if req['locked'] and req['locked'] != owner:
                            raise ValueError('inconsistent_binding')
                        req['locked'] = owner
    # SELECT * has no ordinary Column nodes in SQLGlot.
    if any(e.is_star for e in projection):
        raise ValueError('unexpanded_star')
    mentioned = set()
    qw = words(question)
    for t in schema.tables:
        tw = words(t)
        if tw and any(qw[i:i+len(tw)] == tw for i in range(len(qw)-len(tw)+1)):
            mentioned.add(t)
    source_tables = set(sources.values())

    def rank(binding):
        changed = sum(bool(requirements[k]['col'].table and _physical_owner(requirements[k]['col'], sources, schema)
                           and _physical_owner(requirements[k]['col'], sources, schema) != t) for k, t in binding.items())
        owners = set(binding.values())
        return (changed, -len(owners & mentioned), len(owners - source_tables - mentioned), len(owners), tuple(sorted(binding.items())))

    beam = [{}]
    for k, req in sorted(requirements.items()):
        name = k[1]
        choices = [req['locked']] if req['locked'] else sorted(t for t, cs in schema.columns.items() if name in cs)
        # A literal table qualifier in a target header supplies a physical binding.
        if not req['locked'] and k[0] in schema.tables and k[0] not in sources:
            choices = [k[0]] if name in schema.columns[k[0]] else []
        if not choices:
            raise ValueError('no_same_name_schema_column:' + name)
        next_beam = [{**b, k: t} for b in beam for t in choices]
        next_beam.sort(key=rank)
        if len(next_beam) > cfg['binding_beam']:
            events.append({'reason': 'binding_beam', 'discarded': len(next_beam)-cfg['binding_beam']})
        beam = next_beam[:cfg['binding_beam']]
    counted = any(isinstance(n, exp.Count) and isinstance(n.this, exp.Star) for p in projection for n in p.walk())
    extra = set()
    if counted:
        extra.add(count_table or s.args['from'].this.name.casefold())
    return beam, extra


def _graph(schema):
    pairs = defaultdict(set)
    for t, table in schema.tables.items():
        for fk in table.get('foreign_keys', []):
            u, c, d = fk['table'].casefold(), fk['from'].casefold(), fk['to'].casefold()
            if t == u or u not in schema.tables or c not in schema.columns[t] or d not in schema.columns[u]:
                continue
            if t < u:
                pairs[t, u].add((t, c, u, d))
            else:
                pairs[u, t].add((u, d, t, c))
    graph = []
    for (a, b), clauses in sorted(pairs.items()):
        clauses = sorted(clauses)
        options = [(x,) for x in clauses]
        if len(clauses) > 1:
            options.append(tuple(clauses))
        for option in options:
            graph.append((a, b, option, len(clauses) > 1))
    return graph


def _trees(terminals, graph, cfg, events):
    if len(terminals) < 2 or len(terminals) > cfg['skeleton_max_tables']:
        return
    start = min(terminals)
    queue = deque([(frozenset([start]), ())])
    seen = {(frozenset([start]), ())}
    visited = produced = 0
    while queue:
        if visited >= cfg['skeleton_tree_states']:
            events.append({'reason': 'tree_states', 'pending': len(queue)})
            return
        nodes, edges = queue.popleft()
        visited += 1
        if terminals <= nodes:
            degree = defaultdict(int)
            for i in edges:
                a, b, _, _ = graph[i]
                degree[a] += 1; degree[b] += 1
            if all(n in terminals or degree[n] >= 2 for n in nodes):
                produced += 1
                yield [graph[i] for i in edges]
                if produced >= cfg['skeleton_max_trees']:
                    if queue:
                        events.append({'reason': 'tree_schemes', 'pending': len(queue)})
                    return
            continue
        if len(nodes) >= cfg['skeleton_max_tables'] or len(edges) >= cfg['skeleton_max_edges']:
            continue
        for i, (a, b, _, _) in enumerate(graph):
            if (a in nodes) == (b in nodes):
                continue
            nxt = (nodes | {a, b}, tuple(sorted((*edges, i))))
            if nxt not in seen:
                seen.add(nxt); queue.append(nxt)


def _assemble(s, projection, binding, residual, schema, route, tree, extra):
    old_sources = _sources(s)
    needed = set(binding.values()) | extra
    chosen = needed | {t for edge in tree for t in edge[:2]}
    if route == 'P':
        if not needed <= set(old_sources.values()):
            return None
        aliases = {t: a for a, t in old_sources.items()}
    else:
        used = set(old_sources) | {e.alias.casefold() for e in projection if e.alias}
        aliases = {}
        for i, t in enumerate(sorted(chosen)):
            existing = [a for a, table in old_sources.items() if table == t]
            alias = existing[0] if existing else '_s21_' + str(i)
            while not existing and alias.casefold() in used:
                alias += '_'
            used.add(alias.casefold()); aliases[t] = alias

    def bound(node):
        n = node.copy()
        for c in list(n.find_all(exp.Column)):
            if c.is_star:
                return None
            t = binding.get(_key(c))
            if t is None:
                if route != 'P':
                    return None
                t = _physical_owner(c, old_sources, schema)
            if t not in aliases or c.name.casefold() not in schema.columns.get(t, {}):
                return None
            c.set('table', exp.to_identifier(aliases[t], quoted=True))
            c.set('this', exp.to_identifier(schema.columns[t][c.name.casefold()]['name'], quoted=True))
            c.set('db', None); c.set('catalog', None)
        return n

    q = s.copy()
    expressions = [bound(p) for p in projection]
    if any(p is None for p in expressions):
        return None
    q.set('expressions', expressions)
    for k in ('where', 'group', 'having', 'order'):
        if s.args.get(k):
            n = bound(s.args[k])
            if n is None:
                return None
            q.set(k, n)
    if route == 'P':
        joins = []
        for j in s.args.get('joins', []):
            j = j.copy()
            if j.args.get('on'):
                on = bound(j.args['on'])
                if on is None:
                    return None
                j.set('on', on)
            joins.append(j)
        q.set('joins', joins)
        return q

    if not chosen:
        return None
    root = s.args['from'].this.name.casefold()
    root = root if root in chosen else min(chosen)
    def table(t):
        return exp.Table(this=exp.to_identifier(schema.tables[t]['name'], quoted=True),
                         alias=exp.TableAlias(this=exp.to_identifier(aliases[t], quoted=True)))
    q.set('from', exp.From(this=table(root)))
    joins = []; pending = list(tree); introduced = {root}
    while pending:
        edge = next((e for e in pending if (e[0] in introduced) != (e[1] in introduced)), None)
        if edge is None:
            return None
        a, b, clauses, _ = edge
        target = b if a in introduced else a
        eqs = [exp.EQ(this=exp.column(schema.columns[x][c]['name'], table=aliases[x], quoted=True),
                      expression=exp.column(schema.columns[y][d]['name'], table=aliases[y], quoted=True)) for x, c, y, d in clauses]
        joins.append(exp.Join(this=table(target), on=exp.and_(*eqs)))
        introduced.add(target); pending.remove(edge)
    if introduced != chosen:
        return None
    q.set('joins', joins)
    if residual:
        nodes = [bound(p) for p in residual]
        if any(n is None for n in nodes):
            return None
        old = q.args.get('where')
        q.set('where', exp.Where(this=exp.and_(*([old.this] if old else []), *nodes)))
    return q


def proposals(sql, labels, schema, question, count_table, cfg, route, events):
    try:
        s = parse(sql)
        reason = _supported(s)
        if reason:
            events.append({'reason': 'unsupported', 'detail': reason}); return
        s = _prepare(s, schema)
        sources = _sources(s)
        if any(t not in schema.tables for t in sources.values()) and route == 'P':
            events.append({'reason': 'missing_source_table'}); return
        residual, replaced = _join_residual(s, sources, schema) if route != 'P' else ([], [])
        for plan, projection in _projection_plans(s, labels, schema):
            try:
                bindings, extra = _bindings(s, projection, residual, schema, question, count_table, cfg, events)
            except ValueError as e:
                events.append({'reason': 'unconstructible', 'projection_plan': plan, 'detail': str(e)}); continue
            for binding in bindings:
                needed = set(binding.values()) | extra
                trees = [()] if route in ('P', 'S') else _trees(needed, _graph(schema), cfg, events)
                if route == 'S' and len(needed) != 1:
                    continue
                for tree in trees:
                    q = _assemble(s, projection, binding, residual, schema, route, tree, extra)
                    if q is None:
                        continue
                    risks = []
                    if route != 'P':
                        risks += ['changed_join_coverage', 'possible_fanout_change']
                    if any(e[3] for e in tree):
                        risks.append('fk_grouping_unknown')
                    yield {'sql': text(q), 'skeleton_route': route, 'projection_plan': plan,
                           'binding': [{'qualifier': k[0], 'column': k[1], 'table': t} for k, t in sorted(binding.items())],
                           'fk_tree': list(tree), 'replaced_equalities': replaced,
                           'residual_conditions': [text(x) for x in residual],
                           'removed_tables': sorted(set(sources.values())-needed-{t for e in tree for t in e[:2]}),
                           'risks': risks, 'semantic_proof': False}
    except (ValueError, RecursionError, SqlglotError) as e:
        events.append({'reason': 'unsupported', 'detail': str(e)})
