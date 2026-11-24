"""SQL Repair Agent entry point implementing create_agent(checkpoint=None)."""
import json
import re
from pathlib import Path
from sqlglot import exp, parse_one
from sqlglot.errors import SqlglotError

from mp3_agent.semantic_search_agent import SearchAgent
from mp3_agent.execution_identity import identity
from mp3_agent.safety import sql_key
from mp3_sql.sqlite_tool import validate_sql
from compiler import emit
from llm_engine import generate_verified_sql

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.json"
DEFAULT_PRIOR = HERE / "structure_prior.json"


def _n_tables(sql):
    try:
        ast = parse_one(sql, read="sqlite")
        return len(list(ast.find_all(exp.Table)))
    except (ValueError, SqlglotError, RecursionError):
        return 99


def _is_raw(sql, obs):
    hints = next((e["observation"] for e in obs["history"] if e["tool"] == "repair_hint"), [])
    return sql == obs["issue_sql"] or sql in hints


class Agent:
    def __init__(self, config, prior):
        self.parent = SearchAgent(config)
        self.prior = prior
        self.queue = None
        self.recomb_sent = set()
        self.llm_sent = []
        self.llm_completions = 0
        self.weak_incumbent = None
        self.weak_audited = False

    def _dedup_key(self, sql):
        try:
            return identity(sql, self.parent.schema)[0] or sql_key(sql)
        except ValueError:
            return sql

    def _seen_keys(self, checks):
        return {self._dedup_key(e["sql"]) for e in checks}

    def _valid_unseen(self, sql, seen):
        if not sql or re.search(r"\bLIMIT\s+0\b", sql, flags=re.I):
            return False
        if self._dedup_key(sql) in seen:
            return False
        try:
            validate_sql(sql)
        except ValueError:
            return False
        return True

    def _next_recombination(self, obs, seen):
        if self.queue is None:
            self.queue = iter(emit(obs, self.prior))
        for sql in self.queue:
            if self._valid_unseen(sql, seen):
                self.recomb_sent.add(sql)
                return sql
        return None

    def _should_replace_weak(self, cand_sql, incumbent_sql, ex_rows, obs):
        if re.search(r"\bLIMIT\s+0\b", cand_sql, flags=re.I):
            return False
        if _n_tables(cand_sql) < _n_tables(incumbent_sql):
            return True
        if ex_rows == 0 and not _is_raw(incumbent_sql, obs):
            return True
        return False

    def _current_incumbent_sql(self, obs):
        chosen = self.parent.select(self.parent.groups())
        if chosen:
            return chosen.get("checked_sql", chosen["sql"])
        return obs["issue_sql"]

    def act(self, obs):
        history = obs["history"]
        checks = [e for e in history if e["tool"] == "check"]
        ex_rows = obs["example"]["row_count"]

        if history and history[-1]["tool"] == "check":
            last_sql = history[-1]["sql"]
            last_fb = history[-1]["observation"]
            if last_fb["matches_example"]:
                if last_sql in self.recomb_sent:
                    return {"tool": "submit", "sql": last_sql}
                if last_sql in self.llm_sent:
                    if self.weak_incumbent is None:
                        return {"tool": "submit", "sql": last_sql}
                    chosen = (
                        last_sql
                        if self._should_replace_weak(last_sql, self.weak_incumbent, ex_rows, obs)
                        else self.weak_incumbent
                    )
                    return {"tool": "submit", "sql": chosen}

        if self.weak_incumbent is not None:
            return {"tool": "submit", "sql": self.weak_incumbent}

        action = self.parent.act(obs)
        if not obs["remaining"] or len(history) < 4:
            return action

        has_pub = any(e["observation"]["matches_example"] for e in checks)
        schema = next((e["observation"] for e in history if e["tool"] == "schema"), None)

        # Weak-example high-precision audit (ex_rows == 0 reserves Slot 8; ex_rows == 1 only on submit with >= 2 tables)
        audit_ex0 = (
            ex_rows == 0
            and has_pub
            and not self.weak_audited
            and (action["tool"] == "submit" or obs["remaining"] == 1)
        )
        audit_ex1 = (
            ex_rows == 1
            and has_pub
            and not self.weak_audited
            and action["tool"] == "submit"
            and obs["remaining"] >= 1
            and _n_tables(action["sql"]) >= 2
        )
        if audit_ex0 or audit_ex1:
            self.weak_audited = True
            inc = action["sql"] if action["tool"] == "submit" else self._current_incumbent_sql(obs)
            self.weak_incumbent = inc
            if schema is not None:
                seen = self._seen_keys(checks)
                pub_by_key = {}
                for e in checks:
                    if e["observation"].get("matches_example"):
                        k = self._dedup_key(e["sql"])
                        if k not in pub_by_key or _n_tables(e["sql"]) < _n_tables(pub_by_key[k]):
                            pub_by_key[k] = e["sql"]
                cand, self.llm_completions = generate_verified_sql(
                    obs,
                    schema,
                    self.llm_completions,
                    seen,
                    self._dedup_key,
                    max_internal_tries=3 if ex_rows == 0 else 2,
                    start_sample=False,
                    allow_seen_keys=set(pub_by_key),
                )
                if cand:
                    k = self._dedup_key(cand)
                    if k in pub_by_key:
                        chosen = (
                            pub_by_key[k]
                            if self._should_replace_weak(pub_by_key[k], inc, ex_rows, obs)
                            else inc
                        )
                        return {"tool": "submit", "sql": chosen}
                    self.llm_sent.append(cand)
                    return {"tool": "check", "sql": cand}
            return {"tool": "submit", "sql": self.weak_incumbent}

        if has_pub:
            return action

        # 0-public-pass schedule: parent uses slots 1-6, slot 7 uses structure-prior recombination,
        # slot 8 uses schema-verified LLM generation with execution feedback.
        if obs["remaining"] >= 3 and action["tool"] != "submit":
            return action

        seen = self._seen_keys(checks)

        if obs["remaining"] == 2 and action["tool"] != "submit":
            recomb_sql = self._next_recombination(obs, seen)
            if recomb_sql:
                return {"tool": "check", "sql": recomb_sql}
            return action

        if action["tool"] == "submit" and obs["remaining"] >= 2 and not self.recomb_sent:
            recomb_sql = self._next_recombination(obs, seen)
            if recomb_sql:
                return {"tool": "check", "sql": recomb_sql}

        if obs["remaining"] >= 1:
            if schema is not None and len(self.llm_sent) < 2:
                prior = []
                for sql in self.llm_sent:
                    fb = next((e["observation"] for e in reversed(checks) if e["sql"] == sql), None)
                    if fb is not None:
                        prior.append((sql, fb))
                if not prior and checks:
                    prior.append((checks[-1]["sql"], checks[-1]["observation"]))
                cand, self.llm_completions = generate_verified_sql(
                    obs,
                    schema,
                    self.llm_completions,
                    seen,
                    self._dedup_key,
                    prior_attempts=prior or None,
                    max_internal_tries=3,
                    start_sample=(len(self.llm_sent) >= 1),
                )
                if cand:
                    self.llm_sent.append(cand)
                    return {"tool": "check", "sql": cand}
            recomb_sql = self._next_recombination(obs, seen)
            if recomb_sql:
                return {"tool": "check", "sql": recomb_sql}

        return action


def create_agent(checkpoint=None):
    cfg_path = Path(checkpoint) if checkpoint else DEFAULT_CONFIG
    config = json.loads(cfg_path.read_text(encoding="utf8"))
    prior = json.loads(DEFAULT_PRIOR.read_text(encoding="utf8"))
    return Agent(config, prior)
