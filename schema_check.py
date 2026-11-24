"""Local AST schema-binding validator against already-paid official schema (zero tool cost)."""
from sqlglot import exp, parse_one
from sqlglot.errors import SqlglotError


def _build_schema_map(schema):
    table_cols = {}
    all_cols = set()
    for t in schema:
        tname = t["name"].lower()
        cols = {c["name"].lower() for c in t["columns"]}
        table_cols[tname] = cols
        all_cols.update(cols)
    return table_cols, all_cols


def check_schema_binding(sql, schema):
    """Verify tables and columns exist in paid schema, allowing CTEs and SELECT aliases.

    Returns (True, None) if valid or ambiguous/safe, or (False, error_message).
    """
    try:
        ast = parse_one(sql, read="sqlite")
    except (ValueError, SqlglotError, RecursionError) as e:
        return False, f"SQL parse error: {e}"

    table_cols, all_cols = _build_schema_map(schema)

    # Collect CTE names
    cte_names = set()
    for cte in ast.find_all(exp.CTE):
        alias = cte.alias_or_name
        if alias:
            cte_names.add(alias.lower())

    # Validate every physical table reference
    for tbl in ast.find_all(exp.Table):
        tname = tbl.name.lower()
        if not tname or tname in cte_names:
            continue
        if tname not in table_cols:
            valid_t = ", ".join(t["name"] for t in schema)
            return False, f"Table '{tbl.name}' does not exist. Valid tables: {valid_t}"

    # Validate columns per SELECT scope
    for sel in ast.find_all(exp.Select):
        aliases = {}
        phys_tables = []
        has_subquery_or_cte_source = False

        from_clause = sel.args.get("from")
        sources = ([from_clause.this] if from_clause else []) + [
            j.this for j in sel.args.get("joins", [])
        ]
        for src in sources:
            if isinstance(src, exp.Table):
                tn = src.name.lower()
                if tn in cte_names:
                    has_subquery_or_cte_source = True
                    aliases[src.alias_or_name.lower()] = None  # CTE source
                elif tn in table_cols:
                    phys_tables.append(tn)
                    aliases[src.alias_or_name.lower()] = tn
            else:
                has_subquery_or_cte_source = True
                if hasattr(src, "alias") and src.alias:
                    aliases[src.alias.lower()] = None

        # Collect SELECT aliases defined in this SELECT (valid in ORDER BY / GROUP BY / HAVING)
        select_aliases = set()
        for proj in sel.expressions:
            if isinstance(proj, exp.Alias):
                select_aliases.add(proj.alias.lower())

        # Check columns directly owned by this SELECT (avoid descending into nested SELECTs)
        for col in sel.find_all(exp.Column):
            if col.find_ancestor(exp.Select) is not sel:
                continue
            cname = col.name.lower()
            if not cname or cname == "*":
                continue
            tref = col.table.lower()
            if tref:
                if tref in aliases:
                    target = aliases[tref]
                    if target is None:
                        continue  # CTE or derived subquery alias
                    if cname not in table_cols[target]:
                        valid_c = ", ".join(sorted(table_cols[target]))
                        return (
                            False,
                            f"Column '{col.table}.{col.name}' does not exist in table '{target}'. "
                            f"Valid columns of {target}: {valid_c}",
                        )
                elif not has_subquery_or_cte_source and tref in table_cols:
                    if cname not in table_cols[tref]:
                        valid_c = ", ".join(sorted(table_cols[tref]))
                        return (
                            False,
                            f"Column '{col.table}.{col.name}' does not exist in table '{tref}'. "
                            f"Valid columns of {tref}: {valid_c}",
                        )
                # Note: correlated outer-query alias reference might not be in `aliases`, so allow if not in aliases
            else:
                # Unqualified column: allow if in select_aliases (when in group/order/having) or any in-scope physical table
                in_clause = col.find_ancestor(exp.Group, exp.Order, exp.Having) is not None
                if in_clause and cname in select_aliases:
                    continue
                if has_subquery_or_cte_source:
                    continue
                if phys_tables:
                    if not any(cname in table_cols[t] for t in phys_tables):
                        avail = "; ".join(
                            f"{t}: [{', '.join(sorted(table_cols[t]))}]" for t in phys_tables
                        )
                        return (
                            False,
                            f"Column '{col.name}' does not exist in FROM/JOIN tables. Available: {avail}",
                        )
                elif cname not in all_cols:
                    return False, f"Column '{col.name}' does not exist in schema."

    return True, None
