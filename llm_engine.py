"""Local offline LLM engine for cycloneboy/SLM-SQL-0.6B with AST schema verification."""
import json
import os
import re
import time
from pathlib import Path

from mp3_sql.sqlite_tool import validate_sql
from schema_check import check_schema_binding

REPO = "cycloneboy/SLM-SQL-0.6B"
REVISION = "88f5b943278482ac9bc7a1ac631c926266f541f"
MAX_COMPLETIONS_PER_TASK = 6
MAX_NEW_TOKENS = 1024

NEG_CUE_RE = re.compile(r"\b(never|without|not|don't|doesn't|no)\b", re.I)

_ENGINE = None


def get_engine():
    """Load the declared public model once from local safetensors files only."""
    global _ENGINE
    if _ENGINE is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = os.environ.get("MP3_BASE_MODEL")
        if not model_path or not Path(model_path).is_dir():
            raise RuntimeError(
                "MP3_BASE_MODEL must point to a local directory containing "
                f"{REPO} (revision {REVISION})."
            )
        device = os.environ.get("MP3_DEVICE", "cpu")
        if device == "cpu":
            torch.set_num_threads(4)
        tok = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
            use_safetensors=True,
            trust_remote_code=False,
        ).eval().to(device)
        _ENGINE = (tok, model)
    return _ENGINE


def schema_to_ddl(schema):
    """Render official paid schema metadata into SQLite DDL statements."""
    lines = []
    for table in schema:
        cols = []
        pks = [c["name"] for c in table["columns"] if c.get("pk")]
        for c in table["columns"]:
            ctype = c.get("type") or "TEXT"
            pk = " PRIMARY KEY" if c.get("pk") and len(pks) == 1 else ""
            cols.append(f"  {c['name']} {ctype}{pk}")
        if len(pks) > 1:
            cols.append(f"  PRIMARY KEY ({', '.join(pks)})")
        for fk in table.get("foreign_keys", []):
            if fk.get("from") and fk.get("table") and fk.get("to"):
                cols.append(
                    f"  FOREIGN KEY ({fk['from']}) REFERENCES {fk['table']}({fk['to']})"
                )
        lines.append(f"CREATE TABLE {table['name']} (\n" + ",\n".join(cols) + "\n);")
    return "\n\n".join(lines)


def extract_sql(text):
    """Extract the final SQL query, splitting on the first </think> tag."""
    tail = text.split("</think>", 1)[-1] if "</think>" in text else text
    for region in (tail, text):
        blocks = re.findall(r"```(?:sql)?\s*(.*?)```", region, flags=re.I | re.S)
        for blk in reversed(blocks):
            cand = blk.strip().rstrip(";")
            if cand:
                return cand
    for region in (tail, text):
        matches = list(re.finditer(r"\b(SELECT|WITH)\b", region, flags=re.I))
        if matches:
            cand = (
                region[matches[-1].start():]
                .split("```")[0]
                .split("</think>")[0]
                .strip()
                .rstrip(";")
            )
            if cand:
                return cand
    return tail.strip().rstrip(";")


def build_clean_prompt(obs, schema, prior_attempts=None):
    """Construct the Text-to-SQL prompt from question, schema DDL, and execution feedback."""
    ddl = schema_to_ddl(schema)
    ex = obs["example"]
    ex_info = (
        f"Expected output columns: {json.dumps(ex['columns'])}; "
        f"example row count: {ex['row_count']}"
    )
    if ex.get("rows"):
        ex_info += f"; sample rows: {json.dumps(ex['rows'][:2], ensure_ascii=False)}"
    neg_hint = ""
    if NEG_CUE_RE.search(obs["task"]["question"]):
        neg_hint = (
            "\nSQLite negation guideline: For questions asking what has 'never' or 'not' happened "
            "(e.g., never participated, not living in, without), prefer EXCEPT, NOT IN, or NOT EXISTS "
            "over '!=' or '<>' on a JOIN (which fails when entities have multiple rows)."
        )
    feedback_block = ""
    critical_warning = ""
    if prior_attempts:
        parts = []
        for sql, fb in prior_attempts:
            if not fb.get("ok"):
                err = fb.get("error") or {}
                fb_str = f"ERROR ({err.get('type', 'Error')}): {err.get('message', '')}"
            else:
                fb_str = (
                    f"Returned {fb.get('row_count')} rows (columns={fb.get('columns')}, "
                    f"sample={fb.get('rows', [])[:2]}), matches_example={fb.get('matches_example')}"
                )
            parts.append(f"Attempt SQL: {sql}\nFeedback: {fb_str}")
        feedback_block = (
            "\n\nIMPORTANT - Previous failed attempts (fix the exact error / produce a different valid query):\n"
            + "\n---\n".join(parts)
        )
        last_fb = prior_attempts[-1][1]
        if not last_fb.get("ok") and (last_fb.get("error") or {}).get("type") == "SchemaError":
            critical_warning = (
                f"\nCRITICAL SCHEMA CONSTRAINT: {(last_fb.get('error') or {}).get('message', '')} "
                "Do NOT repeat that invalid table/column name."
            )
    return (
        f"Given the SQLite database schema below, write a valid SQLite SELECT query to answer the question.\n\n"
        f"Schema:\n{ddl}\n\n"
        f"Question: {obs['task']['question']}\n"
        f"{ex_info}{neg_hint}{feedback_block}{critical_warning}\n\n"
        f"Return the final SQL query inside a ```sql ... ``` code block."
    )


def _generate_once(obs, schema, completions_used, prior_attempts=None, sample=False, retry_level=0):
    import torch

    tok, model = get_engine()
    user_body = build_clean_prompt(obs, schema, prior_attempts=prior_attempts)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": user_body}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    ids = tok.encode(prompt, add_special_tokens=False)[-3584:]
    x = torch.tensor([ids], device=model.device)
    gen_kwargs = {
        "input_ids": x,
        "attention_mask": torch.ones_like(x),
        "max_new_tokens": MAX_NEW_TOKENS,
        "pad_token_id": tok.eos_token_id,
        "use_cache": True,
    }
    if sample:
        torch.manual_seed(42 + completions_used)
        if retry_level >= 2:
            gen_kwargs.update({"do_sample": True, "temperature": 0.9, "top_p": 0.95, "top_k": 40})
        else:
            gen_kwargs.update({"do_sample": True, "temperature": 0.7, "top_p": 0.95, "top_k": 20})
    else:
        gen_kwargs.update({"do_sample": False})

    t0 = time.monotonic()
    with torch.inference_mode():
        out = model.generate(**gen_kwargs)
    dt = time.monotonic() - t0
    new_ids = out[0, len(ids):].tolist()
    text = tok.decode(new_ids, skip_special_tokens=True)

    log_path = Path(os.environ.get("MP3_GENERATION_LOG", "outputs/generations.jsonl"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf8") as f:
        f.write(
            json.dumps(
                {
                    "id": obs["task"]["id"],
                    "history": len(obs["history"]),
                    "remaining": obs["remaining"],
                    "sample": sample,
                    "prompt": tok.decode(ids),
                    "text": text,
                    "in_tokens": len(ids),
                    "new_tokens": len(new_ids),
                    "seconds": round(dt, 4),
                    "repo": REPO,
                    "revision": REVISION,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    return extract_sql(text)


def generate_verified_sql(
    obs,
    schema,
    completions_used,
    seen_keys,
    dedup_fn,
    prior_attempts=None,
    max_internal_tries=2,
    start_sample=False,
    allow_seen_keys=None,
):
    """Generate and locally schema-validate SQL before spending a paid check call."""
    local_prior = list(prior_attempts or [])
    allowed = set(allow_seen_keys or ())
    for try_idx in range(max_internal_tries):
        if completions_used >= MAX_COMPLETIONS_PER_TASK:
            return None, completions_used
        use_sample = start_sample or (try_idx > 0)
        sql = _generate_once(
            obs,
            schema,
            completions_used,
            prior_attempts=local_prior or None,
            sample=use_sample,
            retry_level=try_idx,
        )
        completions_used += 1
        if not sql or re.search(r"\bLIMIT\s+0\b", sql, flags=re.I):
            local_prior.append(
                (
                    sql or "",
                    {
                        "ok": False,
                        "error": {
                            "type": "FormatError",
                            "message": "Empty SQL or invalid LIMIT 0.",
                        },
                    },
                )
            )
            continue
        try:
            validate_sql(sql)
        except ValueError as e:
            local_prior.append(
                (sql, {"ok": False, "error": {"type": "InvalidSQL", "message": str(e)}})
            )
            continue
        ok, err_msg = check_schema_binding(sql, schema)
        if not ok:
            local_prior.append(
                (sql, {"ok": False, "error": {"type": "SchemaError", "message": err_msg}})
            )
            continue
        k = dedup_fn(sql)
        if k in seen_keys and k not in allowed:
            local_prior.append(
                (
                    sql,
                    {
                        "ok": False,
                        "error": {
                            "type": "DuplicateError",
                            "message": "Duplicate SQL already checked. Write a DIFFERENT query structure, join, or filter.",
                        },
                    },
                )
            )
            continue
        return sql, completions_used
    return None, completions_used
