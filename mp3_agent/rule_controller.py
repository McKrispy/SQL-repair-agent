"""pure Python, causal full-budget controller. Never reads supervision."""
from collections import deque
import json
import os
from pathlib import Path

from sqlglot.errors import SqlglotError
from mp3_sql.sqlite_tool import validate_sql
from mp3_agent.logs import RunLog
from mp3_agent.rule_rewrites import apply_probe, probe_requests, rewrites
from mp3_agent.rule_support import Constraints, FAMILIES, Schema, parse, scopes, text
from mp3_agent.safety import sql_key, sql_within_limit


class RuleAgent:
    def __init__(self, log=None, *, rules_enabled=True, run_context=None):
        if type(rules_enabled) is not bool:
            raise ValueError("rules_enabled must be an explicit boolean")
        self.log = log
        self.rules_enabled = rules_enabled
        self.run_context = run_context or {}
        self.candidates = {}
        self.rejections = []
        self.probes = []
        self.probed = set()
        self.expanded = set()
        self.notes = []
        self.seen_events = 0
        self.pending_probe = None
        self.constraints = None
        self.schema = None
        self.acts = 0

    def add(self, sql, origin):
        try:
            key = sql_key(sql)
            validate_sql(sql)
        except (ValueError, RecursionError) as exc:
            self.rejections.append({"sql": sql, "origin": origin, "reason": str(exc)})
            return None
        if key not in self.candidates:
            self.candidates[key] = {"key": key, "sql": sql, "origins": [], "feedback": None,
                                    "checked_at": None}
        row = self.candidates[key]
        if origin not in row["origins"]:
            row["origins"].append(origin)
        return row

    def best_origin(self, row):
        return min(row["origins"], key=lambda o: (o["steps"], o["tie"]))

    def evidence(self, row):
        result = self.constraints.evaluate(parse(row["sql"]))
        # These proofs are available only for candidates derived by that rule; aliases
        # or bare EXISTS syntax cannot manufacture them.
        for item in result:
            if item["id"] in {"entity_negation", "entity_intersection"}:
                proof_key = "entity_negation" if item["id"] == "entity_negation" else "both_values"
                if any(any(proof_key in step.get("evidence", {}) for step in o.get("path", [])) for o in row["origins"]):
                    item["state"] = "satisfied"
        return result

    def rank(self, row):
        evidence = self.evidence(row)
        origin = self.best_origin(row)
        return (sum(e["state"] == "contradicted" for e in evidence),
                -sum(e["state"] == "satisfied" for e in evidence),
                origin["steps"], tuple(origin["tie"]), row["key"])

    def add_rewrite(self, parent, item):
        origin = self.best_origin(parent)
        path = origin.get("path", []) + [{"rule": item["rule"], "evidence": item["evidence"], "risks": item["risks"]}]
        return self.add(item["sql"], {"kind": "rewrite", "parent_key": parent["key"],
                                     "root_sql": origin.get("root_sql", parent["sql"]),
                                     "root_order": origin["root_order"], "steps": origin["steps"] + 1,
                                     "tie": [2, origin["root_order"], *[FAMILIES.index(s["rule"]) for s in path]],
                                     "path": path})

    def expand(self):
        if not self.rules_enabled:
            return
        queue = deque(self.candidates)
        while queue:
            key = queue.popleft()
            if key in self.expanded:
                continue
            self.expanded.add(key)
            row = self.candidates[key]
            try:
                tree = parse(row["sql"])
                for scope in scopes(tree):
                    sources = self.schema.sources(scope)
                    if not sources and scope.references:
                        self.notes.append({"key": key, "reason": "unresolvable_scope_sources",
                                           "scope": text(scope.expression)})
                    for col in scope.columns:
                        owners = [a for a, (_, cs) in sources.items() if col.name.lower() in cs]
                        if not col.table and len(owners) > 1:
                            self.notes.append({"key": key, "reason": "ambiguous_unqualified_column",
                                               "column": text(col), "possible_owners": owners})
                        elif col.table.lower() in sources and col.name.lower() not in sources[col.table.lower()][1]:
                            self.notes.append({"key": key, "reason": "missing_qualified_column",
                                               "column": text(col), "possible_owners": owners})
                for item in rewrites(tree, self.constraints, self.schema):
                    prior = len(self.candidates)
                    added = self.add_rewrite(row, item)
                    if added and len(self.candidates) > prior:
                        queue.append(added["key"])
            except (ValueError, RecursionError, SqlglotError) as error:
                # Ambiguous/unsupported ASTs remain usable raw candidates and are visible
                # in the report rather than aborting unrelated tasks.
                self.notes.append({"key": key, "reason": "unsupported_rewrite_ast", "error": str(error)})

    def ingest(self, observation):
        if self.constraints is None:
            self.constraints = Constraints(observation)
            self.add(observation["issue_sql"], {"kind": "original", "steps": 0,
                "tie": [1, 0], "root_order": 0, "root_sql": observation["issue_sql"]})
        for index in range(self.seen_events, len(observation["history"])):
            event = observation["history"][index]
            tool = event["tool"]
            if tool == "schema":
                self.schema = Schema(event["observation"])
                self.constraints.bind_schema(self.schema)
            elif tool == "repair_hint":
                for position, sql in enumerate(event["observation"], 1):
                    self.add(sql, {"kind": "hint", "position": position, "steps": 0,
                                  "tie": [0, position], "root_order": position, "root_sql": sql})
            elif tool == "check":
                try:
                    row = self.candidates.get(sql_key(event["sql"]))
                except ValueError:
                    row = None
                if row:
                    row["feedback"] = event["observation"]
                    row["checked_at"] = index
                    row["checked_sql"] = event["sql"]
            elif tool == "execute":
                if self.pending_probe is None or self.pending_probe["sql"] != event["sql"]:
                    raise ValueError("Probe feedback without its causal request")
                request = self.pending_probe
                parent = self.candidates[request["parent_key"]]
                item = apply_probe(parent, request, event["observation"])
                added = self.add_rewrite(parent, item) if item else None
                self.probes.append({**request, "feedback": event["observation"], "trace_index": index,
                                    "result_key": added["key"] if added else None})
                self.pending_probe = None
        self.seen_events = len(observation["history"])
        if len(observation["history"]) >= 3:
            self.expand()

    def finish(self, observation, reason):
        rows = list(self.candidates.values())
        passed = [r for r in rows if r["feedback"] and r["feedback"].get("matches_example")]
        executable = [r for r in rows if r["feedback"] and r["feedback"].get("ok")]
        unchecked = [r for r in rows if r["feedback"] is None]
        pool = passed or executable or unchecked
        selected = min(pool, key=self.rank) if pool else None
        sql = selected["sql"] if selected else "SELECT 1"
        sql_key(sql)
        validate_sql(sql)
        for row in rows:
            row["constraints"] = self.evidence(row)
            row["rank"] = self.rank(row)
        record = {"task_id": observation["task"]["id"], "question": observation["task"]["question"],
                  "configuration": {**self.run_context, "rules_enabled": self.rules_enabled},
                  "constraints": self.constraints.definitions, "candidates": rows,
                  "rejections": self.rejections, "probes": self.probes, "notes": self.notes,
                  "selection": {"key": selected["key"] if selected else None, "sql": sql,
                                "mode": "public_pass" if passed else "executable" if executable else "unchecked" if unchecked else "placeholder",
                                "reason": reason, "remaining": observation["remaining"]},
                  "public_pass": bool(passed), "trace": observation["history"] + [{"tool": "submit", "sql": sql}],
                  "counts": {"act_calls": self.acts, "charged_tools": 8 - observation["remaining"],
                             "completion_attempts": 0, "input_tokens": 0, "new_tokens": 0}}
        if self.log:
            self.log.finish_task(record)
        self.record = record
        return {"tool": "submit", "sql": sql}

    def act(self, observation):
        self.acts += 1
        self.ingest(observation)
        history = observation["history"]
        if not history:
            return {"tool": "schema"}
        if len(history) == 1:
            if not sql_within_limit(observation["issue_sql"]):
                return self.finish(observation, "original_outside_tool_contract")
            return {"tool": "check", "sql": observation["issue_sql"]}
        if len(history) == 2:
            return {"tool": "repair_hint"}
        if not observation["remaining"]:
            return self.finish(observation, "tool_budget")
        unchecked = sorted((r for r in self.candidates.values() if r["feedback"] is None), key=self.rank)
        diagnoses = []
        if self.rules_enabled and observation["remaining"] >= 2:
            for row in self.candidates.values():
                for request in probe_requests(row, self.constraints, self.schema):
                    if tuple(request["identity"]) not in self.probed:
                        diagnoses.append((self.rank(row), request))
        # SQL checking wins any same-tier tie against a diagnostic.
        best = min(diagnoses, key=lambda x: (x[0], x[1]["identity"])) if diagnoses else None
        if unchecked and (best is None or self.rank(unchecked[0])[:2] <= best[0][:2]):
            return {"tool": "check", "sql": unchecked[0]["sql"]}
        if best:
            self.pending_probe = best[1]
            self.probed.add(tuple(best[1]["identity"]))
            return {"tool": "execute", "sql": best[1]["sql"]}
        if unchecked:
            return {"tool": "check", "sql": unchecked[0]["sql"]}
        return self.finish(observation, "candidates_and_diagnostics_exhausted")

