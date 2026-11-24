"""Single read-only SQL contract and canonical candidate identity."""
import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

MAX_SQL_BYTES = 16_000
_SQL_BLOCK = re.compile(r"\s*```(?:sql|sqlite)?[ \t]*\r?\n(.*?)\r?\n?```\s*", re.I | re.S)


def extract_sql(text):
    fenced = _SQL_BLOCK.fullmatch(text)
    return (fenced[1] if fenced else text).strip()


def sql_within_limit(sql):
    return isinstance(sql, str) and len(sql.encode("utf-8")) <= MAX_SQL_BYTES


def sql_key(sql):
    if not sql_within_limit(sql) or not sql.strip():
        raise ValueError("Expected nonempty SQL within 16000 UTF-8 bytes")
    try:
        statements = sqlglot.parse(sql, read="sqlite")
        if len(statements) != 1 or not isinstance(statements[0], (exp.Select, exp.SetOperation)):
            raise ValueError("Expected a single read-only SELECT or set operation")
        if any(isinstance(node, (exp.DDL, exp.DML, exp.Command, exp.Into, exp.Pragma,
                                 exp.Attach, exp.Detach)) for node in statements[0].walk()):
            raise ValueError("Query contains a forbidden statement")
        return statements[0].sql(dialect="sqlite", comments=False, normalize=True)
    except (ParseError, TokenError, RecursionError) as error:
        raise ValueError("SQL parse failure") from error
