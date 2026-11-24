"""Approved binding-only execution identity; no data or supervision access."""
from sqlglot import exp
from sqlglot.errors import SqlglotError
from mp3_agent.rule_support import parse, text


def bound_query(sql, schema, *, execution=False):
    """Return a flat AST and physical bindings, or an explicit fallback reason."""
    q = parse(sql)
    if not isinstance(q, exp.Select) or q.args.get('with'):
        return None, {}, 'not_flat_select'
    if any(isinstance(n, (exp.Subquery, exp.SetOperation, exp.Window)) or
           (isinstance(n, exp.Select) and n is not q) for n in q.walk()):
        return None, {}, 'nested_scope'
    f = q.args.get('from')
    tables = [f.this, *(j.this for j in q.args.get('joins', []))] if f else []
    if not tables or any(not isinstance(t, exp.Table) or t.db or t.catalog for t in tables):
        return None, {}, 'nonphysical_or_catalog'
    if any(j.args.get('using') or j.args.get('method') for j in q.args.get('joins', [])):
        return None, {}, 'using_or_natural'
    sources = {t.alias_or_name.casefold(): t.name.casefold() for t in tables}
    if len(sources) != len(tables) or len(set(sources.values())) != len(tables):
        return None, {}, 'duplicate_role'
    if any(t not in schema.columns for t in sources.values()):
        return None, {}, 'unknown_table'
    if execution:
        if any(not t.name.isascii() for t in tables): return None, {}, 'nonascii_table'
        if any(isinstance(n, exp.Collate) for n in q.walk()): return None, {}, 'explicit_collation'
        # SQLGlot models AND/OR as Func subclasses; they are logical syntax,
        # not additional SQL function calls. Preserve their AST and operand order.
        if any(isinstance(n, exp.Func) and not isinstance(n, (exp.And, exp.Or, exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max, exp.Cast)) for n in q.walk()):
            return None, {}, 'unsupported_function'
        if any(isinstance(n, exp.Star) and not (isinstance(n.parent, exp.Count) and n.parent.this is n) for n in q.walk()):
            return None, {}, 'projection_star'
    aliases = {e.alias.casefold() for e in q.expressions if e.alias}
    bindings = {}
    for c in q.find_all(exp.Column):
        if c.is_star: return None, {}, 'column_star'
        if c.db or c.catalog: return None, {}, 'catalog_column'
        name = c.name.casefold()
        if execution and not c.name.isascii(): return None, {}, 'nonascii_column'
        if not c.table and name in aliases and c.find_ancestor(exp.Where, exp.Join, exp.Group, exp.Having, exp.Order):
            return None, {}, 'projection_alias_reference'
        owner = sources.get(c.table.casefold()) if c.table else None
        if not c.table:
            owners = [t for t in sources.values() if name in schema.columns[t]]
            owner = owners[0] if len(owners) == 1 else None
        if owner is None or name not in schema.columns[owner]: return None, {}, 'unresolved_column'
        bindings[id(c)] = (owner, name)
    return q, bindings, None


def identity(sql, schema):
    try:
        q, bindings, reason = bound_query(sql, schema, execution=True)
        if reason: return None, reason
        tables = [q.args['from'].this, *(j.this for j in q.args.get('joins', []))]
        aliases = {t.name.casefold(): '_exec' + str(i) for i, t in enumerate(tables)}
        for c in q.find_all(exp.Column):
            owner, name = bindings[id(c)]
            c.set('this', exp.to_identifier(name, quoted=True))
            c.set('table', exp.to_identifier(aliases[owner], quoted=True))
        for t in tables:
            name = t.name.casefold()
            t.set('this', exp.to_identifier(name, quoted=True))
            t.set('alias', exp.TableAlias(this=exp.to_identifier(aliases[name], quoted=True)))
        return text(q), None
    except (ValueError, RecursionError, SqlglotError) as e:
        return None, type(e).__name__


def physical_expression(node, select, schema):
    """Binding signature for evidence comparisons; no executable SQL is returned."""
    try:
        q, bindings, reason = bound_query(text(select), schema)
        if reason: return None
        # Locate an expression by its path rather than trusting textual aliases.
        path = []
        n = node
        while n is not select:
            if n.parent is None: return None
            path.append((n.arg_key, n.index)); n = n.parent
        n = q
        for key, index in reversed(path):
            n = n.args[key] if index is None else n.args[key][index]
        result = n.copy()
        original_columns = list(n.find_all(exp.Column))
        for old, c in zip(original_columns, result.find_all(exp.Column)):
            owner, name = bindings[id(old)]
            c.set('this', exp.to_identifier(name, quoted=True)); c.set('table', exp.to_identifier(owner, quoted=True))
        return text(result)
    except (ValueError, RecursionError, SqlglotError, KeyError, IndexError):
        return None
