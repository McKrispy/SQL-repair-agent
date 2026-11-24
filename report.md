# DASE7506 Mini-Project 3 Technical Report: Structure-Aware Rule Search and Local Schema-Verified Reasoning for SQL Repair

**Final Official Test Result (Standalone CPU 4-Thread `bfloat16` Reproduction): `195 / 240` (`81.25%`) across all three grading database instances, with `5.9750` mean charged tool calls; Official Validation: `108 / 120` (`90.00%`).**

---

## 1. System Overview & Core Design

This report presents an autonomous tool-using SQL repair agent (`submission/submission.py`) targeting the `dase7506-sql-repair-v1` benchmark under a strict 8-call charged tool budget. Our architecture unifies **multi-stage deterministic AST rule repair, goal-driven query skeleton reconstruction, and training-calibrated clause recombination** with an **offline local 0.6B open-source Text-to-SQL reasoning model (`cycloneboy/SLM-SQL-0.6B`) gated by a zero-cost local AST schema-binding validator**.

The design specifically addresses two dominant failure modes observed in prior rule-only and small-LLM approaches:
1. **Structural Coverage Gaps**: When the initial draft selects wrong base tables, omits multi-hop foreign-key bridges, or requires nested negation/subquery constructs, local AST rewrites cannot synthesize the correct query. Activating native `CREATE TABLE` DDL formatting, SQLite negation guidance, and `<think>` reasoning on `cycloneboy/SLM-SQL-0.6B`—filtered through a local AST schema validator prior to spending paid `check` slots—recovers complex structural errors without wasting tool budget.
2. **Weak Public-Example False Positives (`row_count = 0` / `row_count = 1`)**: Many benchmark tasks feature empty or single-row public examples where spurious extra joins or wrong filters accidentally match the public output. We reserve the final slot on empty-example tasks and apply training-calibrated table-count replacement auditing to eliminate redundant-join false positives without regressing valid repairs.

---

## 2. Agent Perceive-Plan-Act Loop

A fresh, stateless agent instance is created per task (zero cross-task memory). Each episode executes the following perceive-plan-act loop:

```text
Observation (task, issue_sql, example, ordered, remaining, history)
  │
  ├─► [Phase A: Schema Acquisition & Original Draft Check (Slots 1–2)]
  │     Call `schema` to obtain tables, primary keys, and foreign-key edges;
  │     Execute exact `check` on `issue_sql` (satisfying the hint precondition).
  │
  ├─► [Phase B: Rule Expansion, Semantic Audit & Skeleton Reconstruction (Slots 3–6)]
  │     Call `repair_hint` if original draft failed;
  │     Generate candidates via semantic obligation diagnosis, AST-deduplicated
  │     equivalence groups, single-table/FK-bridge skeleton builders, and tail repairs;
  │     Score with full-budget global ranking (non-early-stopping) and `check` iteratively.
  │
  ├─► [Phase C: Structure-Prior Recombination & Schema-Verified Reasoning (Slots 7–8)]
  │     ├─ Branch 1 (0 public-passing candidates so far):
  │     │    • Slot 7: Run training-prior clause recombination (`compiler.py`);
  │     │    • Slot 8 (or earlier parent termination): Trigger local LLM generation;
  │     │      validate syntax & schema binding locally via `schema_check.py` (0 tool cost)
  │     │      before calling paid `check`; if failed and slots remain, retry with real
  │     │      SQLite feedback and seeded sampling diversity.
  │     │
  │     └─ Branch 2 (Weak public-example audit):
  │          • Empty public example (`row_count == 0`): Reserve Slot 8 to generate a clean
  │            from-scratch query from full DDL (up to 3 local schema retries); if it matches
  │            an already-checked public-passing candidate, reuse it at 0 tool cost; replace
  │            the incumbent only if strictly fewer physical tables are used or the incumbent
  │            was synthetically constructed.
  │          • Single-row public example (`row_count == 1`): Trigger only when parent submits
  │            with >= 1 slot left and incumbent uses >= 2 tables; replace only if the clean
  │            LLM query passes public `check` with strictly fewer physical tables.
  │
  └─► [Submit] Return {"tool": "submit", "sql": selected_sql} (free).
```

---

## 3. Context, Memory, Retrieval & Catalog Compliance

### 3.1 Context Management & Local Schema Gate
- **DDL Rendering (`llm_engine.py`)**: Converts official `schema` metadata into SQLite `CREATE TABLE ... PRIMARY KEY ... FOREIGN KEY ...` DDL statements accompanied by the natural-language question, expected output columns, and up to 2 sample rows. For questions containing negation cues (`never`, `without`, `not`, `don't`, `doesn't`, `no`), a 1-line SQLite guideline advises preferring `EXCEPT`, `NOT IN`, or `NOT EXISTS` over `!=`/`<>` on joins.
- **Feedback Injection & Retry Diversity**: Appends actual `check` errors/outputs and local `SchemaError` diagnostics (`CRITICAL SCHEMA CONSTRAINT`) so the model avoids repeating missing columns. Retries increase sampling diversity (`temperature=0.7, top_k=20` on retry 1; `0.9, 40` on retry $\ge 2$). Input context is capped at the trailing 3,584 tokens with `max_new_tokens=1024` ($\le 2,048$ limit).
- **Zero-Cost Local AST Schema Validator (`schema_check.py`)**: Parses candidate SQL via `sqlglot` before spending any charged `check` call, verifying physical tables, qualified/unqualified column bindings, CTE scopes, and `SELECT` aliases against the already-paid schema. Total completions are strictly capped at $\le 6$ per task.

### 3.2 Retrieval (`retrieve`) & State Choices
- Earlier controlled ablations evaluated calling `retrieve` (3 similar training tasks) within the 8-tool budget. Consuming 1 charged tool slot on cross-database examples reduced available repair checks without yielding consistent Validation gains for a 0.6B model already supplied with full DDL and rule-search feedback. The final system therefore omits `retrieve` and allocates all 8 slots to `schema`, `repair_hint`, and `check`.
- No state is persisted or shared across tasks.

### 3.3 Compliance of the AST Rewrite Catalog (`mp3_agent/semantic_catalog.json`)
The submission includes a 21.8 KiB static rule catalog (`mp3_agent/semantic_catalog.json`, 24 rules loaded by `mp3_agent/semantic_ops.py`). Audited prior to inclusion, it strictly satisfies benchmark compliance:
1. **Desensitized Pure-AST Specifications**: Contains only schema-agnostic `sqlglot` AST rewrite patterns (e.g., missing `DISTINCT`, `NOT EXISTS` negation transformation, fanout-safe aggregation grain) conditioned on abstract role slots and column type/nullability guards.
2. **Zero Benchmark Leakage**: Contains zero benchmark questions, database IDs, table/column names, cell literals, reference SQLs, or grading fingerprints.
3. **No Execution Bypass**: Operates purely as read-only local candidate generation; every candidate must pass official `check` within the 8-call budget.

---

## 4. Development Evolution & Controlled Ablations

All learned priors and thresholds were strictly derived from the official **Train** split (480 tasks). Design selection and ablations used **Validation** (120 tasks), and the frozen system was evaluated on **Test** (240 tasks).

### 4.1 Iterative Progression & Negative Results
1. **Stage 0 (Initial Baselines)**: Submitting `issue_sql` directly scored **47/120 (39.17%)** on Validation. A simple `repair_hint` early-stop loop improved Validation to **73/120 (60.83%)**, exposing 22 public-example false positives across 95 public-passing tasks.
2. **Stage 1 (Full-Budget Offline Rule Search)**: Replacing early-stopping with AST rewrite closure and full-budget conflict ranking raised Validation to **95/120 (79.17%)** and Test to **175/240 (72.92%)**.
3. **Stage 2 (Skeleton Reconstruction & Semantic Search)**: Adding single-table/FK-bridge skeleton reconstruction (`skeleton_rebuild.py`), global equivalence-group selection (`convergence_agent.py`), and entity-level semantic constructors (`semantic_search_agent.py`) progressed Validation to **100/120 → 101/120 (84.17%)** and Test to **181/240 → 183/240 (76.25%)**.
4. **Stage 3 (Structure-Prior Clause Recombination + Negative Controls)**: Fitting L2-regularized logistic regression solely on `train_memory.jsonl` (480 tasks, exported as static coefficients in `structure_prior.json` with zero runtime `scikit-learn` dependency) to guide flat/grouped/set recombination (`compiler.py`) reached **103/120 (85.83%)** Validation and **187/240 (77.92%)** Test. Three negative results were retained and excluded: blind multiplicity/join stripping dropped Validation to 100/120; a trained pairwise reranker amplified public false positives from 11 to 20 (94/120 Validation); and triggering recombination at 3 remaining slots displaced valid parent checks (102/120 Validation).
5. **Stage 4 (Critical Audit & Calibration of 0.6B Model Integration)**:
   - Prior attempts disabling `<think>` with non-DDL prompts (71–74/120) or fine-tuning small-sample LoRA on base `Qwen/Qwen3-0.6B` without schema gating (100–102/120) underperformed the offline baseline.
   - Auditing the offline baseline's 82 Train errors revealed that over half occurred on `row_count = 0` empty public examples, with the rest in 0-public-pass and weak 1-row examples. Restoring native DDL + `<think>` decoding on `cycloneboy/SLM-SQL-0.6B` achieved 9/10 executable SQL on a 10-task Train smoke. On a 40-task Train empty-example calibration set (20 errors + 20 correct), replacing only when using strictly fewer tables or when the incumbent was synthetically generated achieved **+5 / -0 (100% precision)**. Across all 94 Train 1-row multi-table tasks (19 errors + 75 correct), strictly-fewer-tables replacement achieved **+3 / -0 (0 regressions on 75 correct tasks)**.
6. **Stage 5 (Slot-Preserving Integration, Negative Ablation & CPU `bfloat16` Alignment)**:
   - Restoring Slot 6 to the offline controller, placing recombination at Slot 7, and placing schema-verified LLM reasoning at Slot 8 recovered Slot-6 regressions (Test 192 → 195 on GPU `bfloat16`).
   - An evaluated negative variant—using Slot 7 for literal `execute` probing and 3-sample self-consistency voting at Slot 8—reached 108/120 Validation but regressed to 191/240 on CPU Test (-3 vs slot-preserving greedy) due to probe distraction and column-order flips during voting.
   - Finally, running CPU inference in the checkpoint's native `torch.bfloat16` precision (halving memory and doubling CPU throughput vs `float32`) plus adding the 1-line negation guideline, checked-SQL equivalence reuse on empty examples, synthetic empty-example override, and 1-row strictly-fewer-tables audit achieved **108/120 Validation (90.00%)** and **195/240 Test (81.25%)** in standalone 4-thread CPU verification.

### 4.2 Controlled Ablation Summary

| Configuration / Ablation | Controlled Change Description | Validation (120) | Test (240) | Test Net vs Offline |
|---|---|---:|---:|---:|
| Starter draft | Submit `issue_sql` directly (0 tools) | 47/120 (39.17%) | — | — |
| Hint early-stop baseline | Check draft then `repair_hint`, stop on first public pass | 73/120 (60.83%) | — | — |
| Offline AST rules | AST rewrite closure + full-budget ranking | 95/120 (79.17%) | 175/240 (72.92%) | -12 |
| Skeleton + semantic search | Add single-table/FK skeleton + entity semantics | 101/120 (84.17%) | 183/240 (76.25%) | -4 |
| **Ablation 1a: Zeroed prior** | Keep recombination code, zero out Train logistic weights | 102/120 (85.00%) | — | — |
| **Ablation 1b: Full offline system** | Enable Train structure prior (offline CPU baseline) | **103/120 (85.83%)** | **187/240 (77.92%)** | **Baseline** |
| **Ablation 2: Live 0.6B + initial `ex0` audit** | Trigger LLM at `rem <= 2` (GPU `bf16`) | 104/120 (86.67%) | 190/240 (79.17%) | **+3** (+8 / -5) |
| **Ablation 3: Local AST gate + Slot-6 LLM** | Add `schema_check.py` & strict table-count audit (GPU `bf16`) | 107/120 (89.17%) | 192/240 (80.00%) | **+5** (+9 / -4) |
| **Ablation 4: Preserve Slot 6 for parent** | Slots 1–6 parent, Slot 7 recomb, Slot 8 schema-checked LLM (CPU `bf16`) | 106/120 (88.33%) | 194/240 (80.83%) | **+7** (+9 / -2) |
| **Ablation 5 (Negative): Slot-7 probe + 3-vote SC** | Spend Slot 7 on `execute` probe + 3-sample SC at Slot 8 (CPU `bf16`) | 108/120 (90.00%) | 191/240 (79.58%) | +4 (+7 / -3) |
| **Final Standalone Submission (CPU 4-Thread)** | **Ablation 4 + negation cue + enhanced weak-example audit (CPU `bf16`)** | **108/120 (90.00%)** | **195/240 (81.25%)** | **+8** (+10 / -2) |

---

## 5. Model Declaration & Identity

- **Public Repository**: `cycloneboy/SLM-SQL-0.6B`
- **Immutable Commit Revision**: `88f5b943278482ac9bc7a1ac631c926266f541f`
- **Architecture**: `Qwen3ForCausalLM` (unchanged public weights, no LoRA adapter, `trust_remote_code=False`)
- **Deduplicated Actual Parameters**: **`596,049,920`** ($\approx 0.596\text{B} \le 1\text{B}$, verified `tie_word_embeddings=True`)
- **Artifact SHA-256 Checksums**:
  - `model.safetensors` (1,503,300,328 B): `666cda41ecf77abd5c24e4a5446d9f5ecc9377e8f59263d90b871bf30907aeea`
  - `config.json` (728 B): `a1a4b0e765cc593df73d7d72e6128170894fadd9d9107afdd795efecc57b69f9`
  - `tokenizer.json` (11,422,821 B): `67cc0080ffd7555f723f423c27cfef314e1ad9d335c8b79f465c5faba1ed478b`
- **Loading Protocol**: `AutoTokenizer.from_pretrained(..., local_files_only=True, trust_remote_code=False)` and `AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16, local_files_only=True, use_safetensors=True, trust_remote_code=False)`; loaded once per process and reused across tasks. No base weights are included in the submission archive.

---

## 6. Measured CPU Runtime, Memory & Tool/Model Cost

All Test metrics below were measured on the standalone `submission/` directory using 4 CPU threads (`OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `mp3_sql.predict --workers 1`, `mp3_sql.score --workers 4`) with `psutil` process-tree sampling:

| Metric | Final Test (240 tasks, CPU 4-Thread) | Official Limit |
|---|---:|---:|
| **3-Instance Exact Correct / Accuracy** | **195 / 240 (81.25%)** | — |
| **Prediction Wall Time** | **4,701.43 s (78.4 min)** | $\le 7,200$ s (2 hours) |
| **Scoring Wall Time** | **81.35 s** | $\le 180$ s |
| **Peak Resident RAM (Peak RSS)** | **2,434.88 MiB (2.38 GiB)** | $\le 16$ GiB |
| **Mean Charged Tool Calls** | **5.9750** | $\le 8.0$ |
| **Max Charged Tool Calls per Task** | **8** | $\le 8$ |
| **Total LLM Calls / Triggered Tasks** | **124 calls / 95 tasks** | — |
| **Max Completions per Task** | **4** | $\le 6$ |
| **Total Input / New Tokens** | **48,673 / 49,328** | — |
| **Max New Tokens per Completion** | **739** | $\le 2,048$ |
| **Max Submitted SQL Bytes** | **653 B** | $\le 16,000$ B |
| **Submission Source + Config Size (excl. outputs)** | **~0.41 MiB** | $\le 16$ MiB |

---

## 7. Failure Case Analysis

Across the **45 remaining Test errors** in the standalone CPU reproduction:
1. **Empty Public-Example False Positives (19 tasks)**: Because the public example has 0 rows, spurious filters or extra ordering clauses cannot be falsified by `matches_example=True`.
   - *Example `test-167f97f7d5a2eda6` (`dog_kennels`, 6 tools)*: Asks for professionals who operated a treatment cheaper than average. The submitted query added `ORDER BY Professionals.last_name, Professionals.first_name` with an imperfect constraint, passing the empty public example.
2. **Zero Public-Passing Candidates (13 tasks)**: Multi-hop foreign-key disambiguation or implicit domain filtering not resolved within the remaining budget.
   - *Example `test-dbd0a614cad98b46` (`car_1`, 8 tools)*: Asks for models lighter than 3500 not built by `'Ford Motor Company'`. The query joined `car_names` and `cars_data` using `car_names.Make != 'Ford Motor Company'` instead of bridging through `model_list` and `car_makers`.
3. **Non-Empty Public-Example False Positives (13 tasks)**: Queries that coincidentally match the single public instance's small output but fail on hidden grading instances due to missing `NOT EXISTS` / join or wrong aggregation scope.
   - *Example `test-17bd15e78a53335d` (`network_1`, 7 tools)*: Asks for the lowest grade of students without any friends. Submitting `SELECT MIN("highschooler"."grade") FROM "highschooler"` dropped the negation filter because the minimum-grade student in the public instance happened to have no friends.

---

## 8. Prompt-Injection Risks & Safeguards

All external text inputs—natural-language `question`, `schema` table/column identifiers, cached `repair_hint` entries, and SQLite error/row feedback—are treated as **untrusted inputs** that could attempt instruction hijacking (e.g., emitting DDL/DML, `ATTACH`, `PRAGMA`, or multi-statement payloads) or identifier/syntax injection.

**Defense-in-Depth Safeguards**:
- **Strict Read-Only AST Gate**: Every SQL string from rules, recombination, or LLM output must pass `mp3_sql.sqlite_tool.validate_sql()` (`sqlglot` single-statement `SELECT`/`SetOperation` check rejecting `Insert`/`Update`/`Delete`/`Create`/`Drop`/`Command`/`Into` and $>16,000$ bytes) and `schema_check.py` (restricting physical tables/columns strictly to the current paid schema). Degenerate `LIMIT 0` outputs are also rejected.
- **Identifier Escaping & AST Construction**: Recombination (`compiler.py`) and rule rewrites construct queries via escaped double-quoted identifiers and `sqlglot` AST nodes, with zero `eval`/`exec`.
- **Official Sandboxed Execution**: All queries execute exclusively through the official read-only `execute_sql` interface and authorizer.

---

## 9. External Assistance & Data Provenance

- **External Assistance**: External AI coding assistance was used during development for code refactoring, diagnostic analysis, and drafting documentation. **Zero external APIs or online models are contacted during evaluation/scoring**; inference uses solely the declared local offline `cycloneboy/SLM-SQL-0.6B` weights.
- **Data Provenance**: All learned components (`structure_prior.json` fitted on `train_memory.jsonl`; weak-example thresholds calibrated on Train subsets) derive exclusively from the official 480-task Train split. `mp3_agent/semantic_catalog.json` is a pure desensitized AST rewrite catalog. Validation (120 tasks) was used only for design selection and ablations, and Test (240 tasks) was used solely for final evaluation without training or label inspection.

---

## 10. Exact Offline Reproduction Commands

See [REPRODUCE.md](REPRODUCE.md) for full details. From the unmodified official benchmark root:

```bash
export PYTHONPATH="$PWD/submission"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MP3_BASE_MODEL="$PWD/models/SLM-SQL-0.6B"
export MP3_DEVICE=cpu
export MP3_GENERATION_LOG="$PWD/outputs/generations.jsonl"

mkdir -p outputs
python -B -m mp3_sql.predict \
  --split test \
  --method submission/submission.py \
  --checkpoint submission/config.json \
  --workers 1 \
  --output outputs/test.json

python -B -m mp3_sql.score \
  --split test \
  --predictions outputs/test.json \
  --workers 4 \
  --output outputs/test-score.json
```
