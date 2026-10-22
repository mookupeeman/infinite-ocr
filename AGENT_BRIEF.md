# Infinite-OCR: architecture report and brief for the next agent

You are taking over a working Python project. This document tells you what exists, what was measured, what is weak, and what to do next. Read all of it before changing code. Facts here come from live API tests and real documents (dated 10 October 2026), not from vendor docs alone; where the docs and the measurements disagree, the measurement is stated.

Working directory: `infinite_ocr/` (the repo root for this project). Run everything from there with `.venv/bin/python`. The project is not a git repository.

## 1. What the project is

A coursework system (Generative AI course, Prof. Mitesh Khapra) that OCRs PDFs of any length, in English and Indian languages, using two cloud OCR APIs as the main engines, with an LLM agent used sparingly at decision points. The main concern is hard layouts: tables with merged cells, tables that continue across pages, tables mixed with text and figures, sideways pages, mixed languages.

Constraints from the user:

- Prices and units: rupees and metric.
- Never use deprecated models or APIs.
- Do not write documents unless asked; this one was asked for.
- API budget: up to ₹70 for Sarvam and Bodhan live tests (about ₹51 spent so far); OpenRouter credit is US$20 from the course, with a configured cap of US$2 (`OPENROUTER_BUDGET_USD`) that the user is willing to raise.
- Development is on a Mac. An RTX 5070 (8 GB) machine will be used later for local models; the code must run on both and switch by `.env` settings.

## 2. The engines and their measured behaviour

### Sarvam Vision (`client.doc_ai.digitise`, SDK `sarvamai>=0.1.36`)

- Job flow: submit, poll status, get a download URL, download a ZIP. The ZIP holds `<file>/<file>.md`, `<file>/metadata/page_NNN.json` per page, and `manifest.json`.
- Do not pass `model`. The API rejects `"sarvam-vision"` (`MODEL_INVALID`); the server default is Sarvam Vision. Never use the legacy `/doc-digitization` endpoints.
- Limits: 10 pages per PDF file (per file, not per job: one job with 8 one-page files and one job with 2 six-page files both worked), 10 images per ZIP, 200 MB, **10 requests per minute per account, and status polls count**. Price ₹0.50 per page.
- The SDK silently retries 429, 408, 409 and any 5xx twice. A retried `digitise` can start a second paid job. All calls here pass `request_options={"max_retries": 0}`.
- `language` is a single string. The API accepts and bills **any** string (`auto`, `mul`, `hi-IN,en-IN` all ran). Measured on a four-language page: every Indic code, and no code at all, gave byte-identical output for all four scripts; `en-IN` damaged the Indic sections (digits became `२0२९`). So `language` is an Indic/English switch in practice. This is observed behaviour from one test page, not documented; the docs say nothing about mixed-language documents.
- Per-page JSON: `page_num`, `image_width`, `image_height`, `blocks[]` with `block_id`, `coordinates{x1,y1,x2,y2}` (pixels of Sarvam's own ~300 dpi render), `bbox_norm` `[x0,y0,x1,y1]` in 0–1, `layout_tag`, `confidence`, `reading_order`, `text`. Tags seen: paragraph, header, footer, page-number, folio, footnote, section-title, headline, image, image-caption, chart, table, formula. Tables are HTML with `rowspan`/`colspan`.
- Sarvam renders each page's MediaBox, while PyMuPDF renders the CropBox. This project builds chunk PDFs with MediaBox set to the CropBox so the two frames match, and turns sideways pages upright before sending.

### Bodhan indic-ocr (`POST https://api.bodhan.ai/v1/chat/completions`)

- OpenAI-style, exactly one user message with one base64 PNG/JPEG. No language field, no prompt. `table_format` is `html` (default; keeps rowspan/colspan) or `markdown` (flattens merged cells). `max_tokens` is per block (default 2048; this project sends 4096).
- Response: `choices[0].message.content` (markdown; headings not marked up; pictorial blocks omitted; math as LaTeX) and a top-level `blocks[]` with `order`, `label` (37-class taxonomy), `type`, `bbox_xyxy` (pixels of the uploaded image), `conf` (layout confidence only, not text accuracy), `text`.
- **Measured: 8 requests per ~60 s per model.** The 9th gets `429 {"error":{"type":"throttling_error"}}` with `Retry-After: 42`. Success responses carry `X-Ratelimit-Api_key-Limit-Requests: 120` and `...Max_parallel_requests: 10`, which are an outer limit and not the one that binds.
- **Measured: an undecodable image returns HTTP 200 with body `null`** (the docs say 502). A 502 means a block exceeded `max_tokens` (the `detail` field says which). Dense table pages take up to 44 s.
- Price ₹0.20 per image. Open weights exist (`bodhan-ai/indic-doc-parser`: a 33M PP-DocLayoutV3 layout model plus a 0.8B OCR model built on Qwen3.5-0.8B with the Sarvam-30B tokenizer); running them locally is allowed by the user.

### OpenRouter (agent and optional Qwen lane)

OpenAI-compatible. Models used now: `qwen/qwen3.5-9b` (decision LLM), `google/gemini-3.1-flash-lite` (vision tie-breaker), `qwen/qwen3.8-flash` (optional OCR lane). Spend is logged to `out/openrouter_ledger.jsonl`. Prices are fetched from `/api/v1/models` and cached for a day. Total spent on OpenRouter so far is about US$0.01.

## 3. The pipeline

Two stages that share only files on disk.

```
PDF ─► triage ─► manifest.sqlite ─► lanes (Sarvam, Bodhan, optional Qwen) ─► blocks/*.json
                                                                                  │
                                            STAGE 2: assemble.py ◄────────────────┘
                                            tables · text · figures · languages ─► doc.md, tables/, figures/, report.md
```

Entry points:

```bash
python -m src.run FILE.pdf [-l auto|hi-IN|...] [--pages 90-101] [--max-cost-inr N] [--resume|--fresh]
                  [--policy sarvam-primary|both|sarvam-only|bodhan-only] [--qwen]
                  [--no-assemble] [--no-agent] [--no-reocr] [--dpi 200] [-y]
python -m src.assemble out/docs/<name>-<sha8> [--no-agent] [--no-reocr] [--max-cost-inr N]
python -m unittest discover -s tests -t .        # 66 tests, no API calls, about 20 s
```

Outputs per document: `out/docs/<slug>-<first 8 hex of the PDF SHA-256>/` with `manifest.sqlite`, `events.jsonl`, `summary.json`, `sarvam/`, `bodhan/`, `blocks/page_NNNNN.<engine>.json`, `md/`, then from stage 2 `doc.md`, `tables/<id>.html|csv|json`, `figures/*.png`, `report.md`, `assembly.json`.

### Module map (src/, about 7,700 lines including tests)

| Module | Lines | Role |
|---|---|---|
| `run.py` | 457 | CLI; `Run` wires triage and lanes; progress, summary; calls assembly; Ctrl-C handling |
| `lanes.py` | 751 | `Router`, `Budget`, `Context`, `SarvamLane`, `PageLane`, `BodhanAPIBackend`, `QwenBackend`, `BodhanLocalBackend` (placeholder) |
| `manifest.py` | 435 | SQLite state: `pages`, `tasks`, `jobs`, `extras`, `meta`; `EventLog` |
| `triage.py` | 175 | Free per-page checks (PyMuPDF only) |
| `langid.py` | 240 | Free language identification of OCR text, per block and per page |
| `rate_limit.py` | 414 | `Limiter`, `call_with_retry`, response classifiers; old helpers kept for the Qwen week-1 path |
| `sarvam_digitise.py` | 344 | Sarvam client: `submit`, `poll`, `fetch`, `unzip`, week-1 `digitise_pdf` |
| `bodhan_ocr.py` | 452 | `bodhan_request` with its error handling; week-1 sequential and parallel runners |
| `pdf_pages.py` | 224 | Upright rendering, `render_page`, `page_transform`, `upright_rotation`, `build_chunk_pdf` |
| `blocks.py` | 210 | Shared `Block` format; label maps; `page_markdown` |
| `tables.py` | 719 | Table grid, number checks, fusion, join scoring (see section 6) |
| `agent.py` | 259 | OpenAI-compatible agent: `decide_continuation`, `read_cells`; cache; ledger |
| `assemble.py` | 848 | Stage 2 |
| `config.py`, `cache.py`, `compare.py`, `qwen_ocr.py`, `run_week1.py` | — | Settings, content-keyed disk cache, week-1 report and runners (kept working) |

Tests: `tests/` uses `unittest`, fake engines (`tests/fakes.py`: `FakeSarvam`, `FakeBodhan`, `make_pdf`) and a temporary cache. The README (`README.md`) documents commands and results.

### Stage 1: engine run

**Triage** (`triage.py`, 25–180 ms per page, no API calls) records per page: the extra rotation that makes text upright (from text-line direction vectors and `/Rotate`; needs a text layer), blank page, characters per Unicode script, `script_share`, `legacy_font` (font-name patterns such as `DV-TTYogesh`, or more than 20% accented Latin in a non-Latin document), ruled-table outlines (`page.find_tables()`), the **OCR mode** (`indic` or `en`) with its source, and a language hint from a clean Unicode text layer. `script_share` is a script statistic, not a correctness score: Marathi page 2 had `script_share=1.0` and still read `वर्षन` where the page says `वर्णन`.

**Manifest** (SQLite, WAL, one connection behind an RLock, `BEGIN IMMEDIATE` transactions, no network calls inside a transaction). Task states per (page, engine): `pending → inflight` (page engines) or `queued` (inside a Sarvam job) `→ done | failed`, plus `stolen` and `skipped`. Jobs: `submitting → submitted → done | failed | abandoned`. A job row is written **before** the API call and the `job_id` is saved right after, so a crash never loses a paid job: `--resume` polls an existing `job_id` instead of resubmitting, and a job stuck in `submitting` is resubmitted and logged as a possible duplicate. Older run databases are migrated in place when opened.

**Router** (`lanes.Router`) creates tasks; it never calls an API.

- `sarvam-primary` (default): every page gets a Sarvam task (priority 0); a page with a ruled table, or one where Sarvam later returns a table, also gets a Bodhan task (priority 1); a page Sarvam misses twice gets a Bodhan task (priority 2, fallback). A Sarvam lane that dies (bad key, no credit) hands its pages to Bodhan, and vice versa; a Bodhan failure on a stolen page reopens the Sarvam task. `both`, `sarvam-only`, `bodhan-only` and `--qwen` (English pages) are the other policies.
- Priorities order tasks within one engine's queue only. Bodhan workers take their highest-priority, lowest-page task. When they have none and Sarvam has more than 150 pending pages (50 pages per job × 3 open jobs), they take pages from the **end** of Sarvam's queue (state `stolen`).

**Sarvam lane** (one thread): claims up to 50 pending pages of one OCR mode, serves cached pages at ₹0, builds upright chunk PDFs (up to 10 pages each, 5 files per job), records the job, submits, polls (first poll after about 1 s per page, then backs off; polls count against 10/min), downloads, maps `metadata/page_NNN.json` back to original page numbers, and sends missing pages to a new job (max 2 attempts, then fallback).

**Bodhan lane** (4 worker threads): renders the page upright at 200 dpi (PNG), checks the cache, calls `bodhan_request`. Handling: 200 with `null` body, re-send as JPEG; 502 from a block over `max_tokens`, re-send with markdown tables and mark the page degraded; 5xx and timeouts, back off and retry; 429, pause the whole lane for `Retry-After`; 401 or credit exhaustion, stop the lane.

**Rate limiting**: `Limiter` = at most N request starts in any 62 s window, spaced evenly, with an in-flight cap and a lane-wide `pause_until`. Bodhan: 8 starts per 62 s, 8 s apart, 4 in flight. Sarvam: 10 per 62 s, 3 s apart, 1 in flight. `call_with_retry` sorts outcomes into OK, RETRY, RATE_LIMITED, DEGRADE, FATAL (a 429 whose body mentions credit is FATAL).

**Budget** (`lanes.Budget`): reserve before a call, settle after; at `--max-cost-inr` no new work starts. **Cache** (`cache.py`): content-keyed (page image bytes or page identity plus engine, model and settings), atomic writes; re-runs and `--fresh` cost ₹0.

**Languages**: Sarvam gets either an Indic code as a hint or no `language` field. Triage picks English mode only for clean Latin text with no Indic script. After OCR, `langid.page_languages` tags every block (Unicode ranges; Hindi/Marathi/Nepali/Sanskrit/Konkani/Maithili told apart by function words; Assamese by ৰ/ৱ; Sindhi and Kashmiri by extra letters; short blocks take their nearest decided neighbour's language). A page read in English mode whose output contains Indic script is re-read once in Indic mode. `-l` defaults to `auto`; invalid codes are rejected before any request.

### Stage 2: assembly (`assemble.py`)

Reads the manifest, the per-engine `blocks/*.json` and the original PDF. Per page the **primary engine** is the first of Sarvam, Bodhan, Qwen that finished it; it supplies reading order, text and headings. For each table (a `Block` with `kind == "table"`), `_build_table` does the following (the design problem is described in section 6):

1. Collect each engine's version of the table, matched by box overlap (IoU at least 0.3), parsed into a grid by `tables.parse_tables` (HTML with rowspan/colspan, or markdown pipes).
2. Rank versions by the key (text-layer digit ratio, then fewest failed arithmetic checks, then most checks; ties go to the primary engine).
3. `fuse`: for same-shaped grids, swap in the other engine's value for each differing numeric cell and keep it only if fewer arithmetic checks fail.
4. Text-layer vote on cells still disputed.
5. If checks still fail, or cells are disputed with no text layer: crop the table at 300 dpi, send it to Bodhan, and take a majority of three on disputed cells (capped at 10 re-reads per document).
6. A vision model (`gemini-3.1-flash-lite`) rules on cells still disputed, and on cells whose arithmetic check fails (at most 20 questions per table).
7. After all tables are built, `_join_tables` scores each consecutive page pair (last body block on N is a table, first on N+1 is a table). Score at least 5 joins; at most 1 stays separate; 2–4 goes to the decision LLM (`qwen/qwen3.5-9b`), accepted if `continues` and `confidence >= 0.6`. If column counts differ it first tries the other engine's version of page N+1. A joined table drops the repeated header rows and merges a row split by the page break, and arithmetic checks re-run on the whole table.

Then the document is composed: headings get `#` levels from labels, paragraphs split by page breaks, columns or drop caps are rejoined (hyphen-aware), figures are cropped from their boxes with captions attached by position, footnotes are collected, page markers list each page's languages, blocks whose language differs from their page carry `<!-- lang: -->`, and `report.md` / `assembly.json` record triage, tables (engine chosen, checks, fixes, confirmed and unresolved cells), joins with the deciding evidence, languages, labels seen and cost.

## 4. Test data and measured results

`samples/`: `marathi-fiscal-report.pdf` (8 pp, sideways fiscal tables, legacy-font text layer), `Ponniyan-Selvan.pdf` (Tamil), `Patrick-Deneen-sample.pdf` (English, drop caps), `union-receipt-budget-2026-27-hindi.pdf` (106 pp; a 23-column table running about 20 pages; pp. 8–25 scanned), `ncert-class12-maths-hindi-ch6.pdf` (equations, MCQs, figures), `economic-survey-2025-26-ch4-external-sector.pdf` (charts; a table over pp. 16–18), `yojana-telugu-march-2025.pdf` (three-column magazine), and two mixed-language pages built from these (`mixed-language-test-page.pdf` and its scanned copy). Source URLs are in the README.

| Window | Pages | Cost | Wall | Tables | Arithmetic checks |
|---|---|---|---|---|---|
| Union budget pp. 90–101 | 12 | ₹8.51 | 94 s | 12 joined into 1 (210 rows) | 198/199 |
| NCERT maths pp. 1–13 | 13 | ₹7.54 | 35 s | 4 | none apply |
| Economic Survey pp. 11–18 | 8 | ₹4.61 | 19 s | 3 joined into 1 (pp. 16–18) | none apply |
| Yojana Telugu pp. 50–57 | 8 | ₹4.20 | 14 s | 1 | none apply |
| Marathi fiscal (8 pp) | 8 | — | — | 8 | 100/100 |

No 429s in any run. Upright rendering fixed digit errors in both engines on the sideways Marathi pages (page 2: Sarvam had misread 3 of 5 key numbers and Bodhan the total, at 137 dpi sideways; both correct upright at 200–300 dpi).

## 5. What is weak outside the table layer (known limits)

- Scanned pages are not rotated automatically; upright detection needs a text layer.
- Language: Dogri and Bodo have no word lists (they fall back to the `-l` hint or Hindi); a page of only numbers takes the document's majority language. "Indic mode reads several scripts on one page" was observed on one four-language page only.
- `BODHAN_BACKEND=local` runs `bodhan-ai/indic-ocr` from `models/indic-ocr` (`src/indic_local.py`). The local agent (`AGENT_BASE_URL`) is still a separate switch.
- Evaluation is by inspection and by internal checks; there is no ground-truth benchmark, no CER/WER, no table-structure metric.
- Figures are cropped and linked but not described.

## 6. The core problem: the Stage 2 table design is the weakest part of the system

The user's assessment, which I agree with: the table layer checks and the way the two engines are used for it are the weakest part of the architecture. The agent models are used very sparsely, and the trust model is wrong in several ways.

### 6.1 What is wrong

1. **Agreement between the engines is treated as evidence of correctness.** Sarvam and Bodhan can both be wrong, and wrong the same way: both are Indic-tuned, and Bodhan's OCR model is built on Qwen. Where the two agree on a cell, nothing ever looks at the image again. There is no mechanism that can catch a shared error.
2. **The verification signals only cover a narrow class of tables.**
   - Arithmetic checks (`tables.check_table`: formulas read from headers such as `एकूण (४ ते ७)`, consensus totals, a product rule) apply only to tables that contain sums. They never apply to a table that lists names, categories, dates or descriptions, or to mixed tables' non-numeric columns. On NCERT, the Economic Survey and Telugu runs they produced zero checks. They are also heuristics tuned on three real documents (consensus thresholds 60% and 80%, exclusion rules for serial, code and copied columns); each real document exposed a new false alarm, which shows how fragile they are.
   - The text-layer vote compares **numbers only** and only on born-digital pages with a usable layer. It is useless on scanned pages and on textual tables. Where text matching was considered, the layer itself is unreliable for Indic scripts (legacy fonts give garbage; Unicode layers scramble matras and conjuncts: `वर्षन` for `वर्णन`).
3. **Textual and non-numeric tables are effectively unverified.** The text-layer score has no numbers to match, so both versions get "no evidence", and the primary engine's version wins by tie-break. `fuse` skips cells where neither value parses as a number, so text cells are never compared or repaired. The engine-agreement figure counts numbers only and reports 0.0 for a textual table, which drags the document average down. Real example: on Union budget page 92 the engines read a project name as `घोमन-टांडा` and `घोमने-टांड…`; nothing compared them and Sarvam's was used.
4. **Ranking between engines per table is weak.** The key rewards the share of an engine's numbers found in the text layer. That measures precision, not completeness: a version that dropped a row still scores 100%. Missing rows, merged or split columns and wrong header structure are only caught if the two versions' grid shapes happen to differ (`engines_disagree_on_shape`), and then fusion is skipped altogether.
5. **The LLM agents are underused and too weak.** The vision model (`gemini-3.1-flash-lite`) is a cheap tie-breaker that sees only cells the heuristics have already flagged, so it cannot find errors the heuristics miss. The decision LLM (`qwen/qwen3.5-9b`, text only) is consulted only for page joins in the score band 2–4, using text evidence alone; it never sees the pages. A real failure: on the Economic Survey it ruled that a table on page 18 was separate from the one on page 17 because the row categories changed, which was wrong (fixed by adding rule evidence, not by improving the judge). Models are not benchmarked; they were picked by price and reputation.
6. **The Bodhan crop re-read is a third opinion from the same model family** as one of the first two, and is capped at 10 per document, so it is not an independent check.
7. **Cross-page joining is rule-scored from engine output, not from the pages.** The score uses column counts, repeated header text, numbering rows, bbox position and "continued" markers. It depends on both engines having segmented the table the same way. Specific limits: it only compares page N with page N+1; it fails when a table resumes after a figure or blank page; a column count mismatch between engines is handled by trying the other engine's version, but there is no column mapping when an engine split or merged a column; there is no memory of a table's signature, so a 20-page table makes 19 separate decisions; and a join decision is never verified visually across the seam.
8. **Post-join verification is again arithmetic only**, so a join error in a non-numeric table is invisible.
9. **There is no way to measure any of this.** No table benchmark, no ground truth, no metric (TEDS, cell accuracy, join accuracy), no measurement of how often the judge catches planted errors. All current "checks passed" figures say the arithmetic was consistent, not that the table is correct.

### 6.2 The direction the user wants

- Do not trust Unicode matching, number matching or column arithmetic as the basis. At most they are hints that raise suspicion.
- Check tables without caring whether they hold numbers, text, a mixture, or neither.
- Use a heavier model, and a vision model, as a judge or evaluator after the engines have run.
- Handle tables that span several pages in a general way, including consistency checks and stitching.

### 6.3 Proposed re-architecture (the design I recommended; the user has not yet answered the three decisions below)

Principle: **every cell is backed by a reading of the image, and agreement is not proof; independence is.** No cell changes on heuristics alone.

```
table segments from Sarvam + Bodhan
 ① ALIGN      one grid from all readings; every cell keeps all candidate values and provenance.
              Type-agnostic cell similarity (normalized edit distance); rows aligned by sequence
              alignment; columns matched by header and content similarity.
 ② PROFILE    free structural signals: readers disagree on shape/rows/columns; per-column "shape"
              (character-class pattern) breaks; rows seen by one reader only. Arithmetic and
              text-layer results become optional hints that raise a risk score.
 ③ JUDGE      a vision model sees the table image (tiled into overlapping bands for tall tables)
              plus the merged grid. It rules on disputed cells, AUDITS a random sample of rows the
              engines AGREED on (this is how shared errors are caught), and gives a structural
              verdict (rows, columns, header rows, merged cells).
 ④ THIRD      escalate when the judge finds errors or risk is high: a vision model from a different
    READER    family than Bodhan/Sarvam transcribes the whole table; majority of three per cell.
 ⑤ ADJUDICATE a heavy model sees the image and all readings for the residual cells only.
 ⑥ NEEDS      anything still uncertain goes to a review list in report.md with the crop.
    REVIEW
```

- Every cell records candidates, final value, who decided and confidence. Every table ends as verified, repaired or needs-review.
- Tall tables are tiled into overlapping bands read at full resolution, then joined by aligning the overlapping rows.
- A `--table-qa light|standard|strict` setting controls spend (strict puts every table through the third reader).

OpenRouter models available today (prices US$ per million input/output tokens, checked 11 October 2026; all support JSON-schema output): `google/gemini-3.8-flash` 0.75/3.75, `google/gemini-3.5-flash` 1.50/9.00, `anthropic/claude-sonnet-5.5` 2/10, `openai/gpt-5.6-sol` 2/10, `anthropic/claude-opus-5.5` 4/20, `qwen/qwen3.8-27b` 0.42/2.55, `qwen/qwen3-vl-235b-a22b-instruct` 0.21/1.90. Suggested roles: judge `gemini-3.8-flash`; third reader from a family that is not Qwen (Bodhan's OCR is Qwen-based); adjudicator `claude-sonnet-5.5` or `gpt-5.6-sol`. Do **not** pick the judge by reputation: Bodhan's published benchmark has Gemini 3.1 Pro (80.4) below Bodhan (86.2) and Sarvam Vision (86.6) on printed Indic text. Measure candidates on the benchmark below. Rough cost estimates (to be confirmed in a pilot): judge about ₹0.5 per table, third reader ₹1–3, adjudicator ₹1.5 on residual cells; the Union budget's 12 table pages would add about ₹10–15 in `standard` mode.

**Cross-page joining, rebuilt:**

1. A document-wide **table registry**: each segment gets a type-agnostic signature (column geometry found from the image itself, via ruling lines or whitespace gaps; header text; per-column shape profile; caption; position). Once two segments are confirmed as one table, later matching segments join without another model call.
2. Compare each segment with the next one or two table segments, not only the next page, so a table can resume after a figure or blank page.
3. For any uncertain join, show the vision judge **one composite image**: the last rows of page N stacked above the first rows of page N+1. It returns whether the table continues, the **column mapping** (handles an engine that split or merged a column), how many header rows repeat, and whether a row is cut across the page break.
4. After joining, check structural consistency across the seam (column shapes, a serial or key column continues, no duplicated or missing rows); a failure goes back to the judge.

**Benchmark first** (the user can trust nothing, including a judge, that is not measured):

1. Synthetic tables with known truth: generate realistic Indic and English tables (Indian digits, merged headers, empty cells, mixed and purely textual tables, symbols), render to PDF with PyMuPDF HTML layout and Noto fonts, optionally degrade like scans (blur, noise, JPEG, skew), split long ones across pages with repeated headers.
2. Error injection: take a correct transcription, plant errors (digit swaps, missing rows, merged columns, wrong script) and measure judge recall and false-alarm rate.
3. Real gold: born-digital English tables (Economic Survey) and 10–20 hand-checked Indic tables.
4. Metrics: TEDS, cell accuracy, judge recall and precision on planted errors, join accuracy across pages.

Run three candidate judges on the same set and choose by accuracy per rupee.

**Latency:** start table work as soon as a page's engine results arrive, in parallel with Stage 1 still reading later pages; run judge calls concurrently (OpenRouter limits are generous); join in page order as consecutive pages complete.

### 6.4 Where to start (suggested build order)

1. Alignment of all readings, per-cell provenance, tiled crops, column geometry from the image (all free). Replace `tables.fuse` and the numeric-only `numbers_in` / `compare_numbers` / `text_layer_match` as the basis for decisions; keep the arithmetic checks only as hints.
2. The synthetic table benchmark with error injection and metrics (free, local); a script under `tests/` or a new `bench/` folder.
3. Judge, third reader and adjudicator in `agent.py` (keep the existing cache, ledger and JSON-schema pattern); pilot two or three judge models on the benchmark.
4. Rebuild joining: registry, composite seam images, column mapping, post-join structural checks; evaluate on synthetic multi-page tables plus the Union budget (pp. 90–101) and the Economic Survey (pp. 16–18).
5. Report outputs: verified / repaired / needs-review per table, with crops; update `README.md`.

### 6.5 Decisions the user still owes (ask before spending)

1. OpenRouter budget: the cap is US$2; the pilot plus normal use needs about US$10–15 of the US$20 course credit.
2. Default QA level: `standard` (judge on every table, third reader only on flagged tables) or `strict`.
3. Benchmark first (recommended) or pipeline first.

## 7. Working agreements

- Keep the 66 tests green (`python -m unittest discover -s tests -t .`); add tests for new behaviour. Tests use fake engines and a temporary cache and must not call paid APIs.
- Any paid API call outside tests needs a spend estimate first; report actual spend afterwards. Never print key values (`.env` holds `SARVAM_API_KEY`, `BODHAN_API_KEY`, `OPENROUTER_API_KEY`; do not commit or echo them).
- Do not change the manifest schema without a migration (`manifest.MIGRATIONS`).
- Keep results cached by content so re-assembly costs ₹0; keep the agent behind `--no-agent`.
- Prefer measured facts. When a vendor doc and a live test disagree, record the measurement.
- Update `README.md` after finishing a milestone.
