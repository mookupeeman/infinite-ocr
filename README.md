# Infinite-OCR

OCR for PDFs of any length, in English and Indian languages, built on two cloud OCR engines:

- **Sarvam Vision.** It takes up to 10 pages per PDF and several PDFs per job.
- **Bodhan indic-ocr.** It takes one page image per request.

A small LLM agent is consulted only at decision points; it runs on OpenRouter now and is meant to run locally on an RTX 5070 later.

The system's main concern is getting tables right:

- merged cells are kept;
- tables that continue across page breaks are joined;
- the numbers are checked against the table's own totals, against the PDF text layer, and between the two engines.

## How it works

```
PDF ─► triage (free) ─► Sarvam lane ──┐
        rotation, blank,  Bodhan lane ┼─► one block format ─► assemble ─► doc.md, tables/, figures/, report.md
        script, tables    (Qwen lane) ┘      (per page)        tables: check → repair → join
                    └──────── SQLite manifest: resume after a crash, cost cap, events log ────────┘
```

1. **Triage** (PyMuPDF, no API calls) runs on each page:
   - It finds the rotation that makes text upright. Budget books print wide tables sideways; turning them upright fixed digit errors in both engines.
   - It marks blank pages and finds tables drawn with ruled lines.
   - It records the text layer's script, and whether the PDF uses a legacy (non-Unicode) Indic font.
   - It decides the page's OCR mode for Sarvam: English mode only for clean English text, Indic mode otherwise (see Languages below).
2. **Lanes** run in parallel, each behind its own rate limiter:
   - **Sarvam** reads every page. It sends upright chunk PDFs, several per job.
   - **Bodhan** reads:
     - pages with tables, as a second reading;
     - pages Sarvam failed;
     - pages it takes from the end of Sarvam's queue when Sarvam is backlogged.
   - **Qwen** (optional) reads English pages.
   - Every result is cached, so a re-run or a resumed run never pays for the same page twice.
3. **Assembly** builds the document:
   - For each table it picks the best engine's version, then settles disputed cells in order of cost:
     1. number checks (free);
     2. PDF text-layer digits (free);
     3. a Bodhan re-read of the table crop (₹0.20);
     4. a vision model (about ₹0.04).
   - Next it joins tables that continue onto the next page.
   - Finally it crops figures, joins paragraphs split by page or column breaks, and writes the report.

### Languages, including several on one page

Measured on a test page with English, Tamil, Hindi and Marathi sections (2026-10-10): Sarvam's `language` code is only a switch between Indic and English reading.

- Every Indic code (`hi-IN`, `mr-IN`, `ta-IN`), and no code at all, gave byte-identical output for all four sections, English included.
- `en-IN` changed the result and damaged the Indic sections. The Marathi table fell to 70% similarity, with digits corrupted (`२०२१` became `२0२९`).

So the pipeline needs to know the languages only *after* OCR, apart from keeping Indic pages out of English mode:

1. **Before OCR, triage picks the mode.** A page gets English mode only if its text layer is clean Latin with no Indic script. Indic text, legacy fonts and scanned pages go to Indic mode, unless you pass `-l en-IN` for a scanned document. Indic pages share one language code, so a mixed document still fills large Sarvam jobs.
2. **After OCR, every block is tagged with its language** from the text either engine returned (`src/langid.py`, free):
   - Each Indic script maps to its language by Unicode range.
   - Languages sharing a script are told apart by their words: Hindi है/और/में, Marathi आहे/आणि/च्या, Nepali छ/पनि/हरू, Sanskrit, Konkani, Maithili. Assamese is told from Bengali by ৰ/ৱ, and Sindhi and Kashmiri from Urdu by their extra letters.
   - A block with too few words, such as a table, takes the language of its nearest decided neighbour.
3. **Safety net.** A page read in English mode whose output turns out to contain Indic script is re-read once in Indic mode (₹0.50 for that page only).
4. **Invalid codes are refused before any request.** Sarvam accepts and bills any string as `language`, including `auto`, `mul` and `hi-IN,en-IN`.

Results:
- **Saved test runs:** 94 of 94 pages got the right main language with no hint, including every Hindi and Marathi page.
- **The four-language test page:** every block was tagged correctly, and Hindi and Marathi were separated on the same page.
- **Where it shows up:**
  - `doc.md` page markers list each page's languages;
  - blocks in a different language from their page carry a `<!-- lang: … -->` tag;
  - `report.md` has a Languages section.

### Number checks (no API calls)

Indian government tables state their own arithmetic, and the checks read it from the table:

- **Column formulas.** A numbering row (१ २ ३ …) plus a header like `एकूण (४ ते ७)` means column 8 = columns 4 to 7.
- **Row formulas.** A total row like `एकूण (क+ख+ग+घ)` means the sum of the rows labelled क, ख, ग and घ.
- **Consensus rules, for tables without formulas.** A "Total" column or row is a sum, or a column is a product of two others (annual × years = total). Such a rule is accepted only if most rows obey it; the rows that don't are flagged. Serial-number columns, budget-code columns (००४९) and columns that mostly copy another column are ignored.

### Table checking (`--table-qa standard`, the default)

The number checks above are hints only. Tables are checked like this:

- **Line up every reading.** Sarvam's and Bodhan's tables are aligned row by row and column by column into one grid; each cell keeps every engine's text. Phantom columns (empty everywhere, no header) are dropped first.
- **Read the structure from the page image** (`src/table_geometry.py`), inside the table's box only. In a ruled table the ruling lines give the row and column counts, so a row an engine dropped is caught; the lines only choose between the engines' shapes and never override both. A column line broken by merged cells still counts, a note row at the bottom of the frame (टीप: …) is not a table row, and a box that cuts off the table's side is widened. Measured: 322 of 322 ruled benchmark tables read exactly (born digital and scanned), no grid claimed on 172 unruled ones (`python -m bench.geometry_eval`). Tables with only header rules (the Union budget) give a column count from gaps, as a hint. A printed column-numbering row (१ २ ३ …) also confirms a column count. PyMuPDF's own table finder is only used to flag table pages: it counts invisible cell boxes (55 × 23 on a 9-column Union page) and fails on sideways pages.
- **Settle cells with free evidence.** Readings that differ only in spacing, dashes or `$..$` are the same. A printed formula decides only when it holds for exactly one reading; the PDF text layer decides only after it has proved itself on the cells the engines agree on. Neither writes a value of its own.
- **Image check for the rest** (`src/table_judge.py`), on numbered cut-outs of the table (`src/table_cutouts.py`), in Normal and Think modes only. The rule was chosen on a pilot of 395 real questions with known answers (`bench/pilot.py`):
  - born-digital pages: GPT-5.6 Luna, accepted only if its reading equals one engine's reading;
  - scanned pages with ruling lines: Gemini 3.1 Flash-lite and GPT-5.6 Luna must read the same value, and it must equal an engine's reading (Think adds Gemini 3.5 Flash-Lite and accepts 2 of 3: 46 of 56 decided right, harm 0, on the pilot's scanned ruled disputes);
  - scanned pages without ruling lines: no model (row strips are guesses there); the cells stay open;
  - never: a value no engine produced, an empty value over printed text, cells with rare numerals (Tamil, Telugu) or where a proven text layer contains neither reading.
- **Do not trust agreement.** Agreed cells that look risky are flagged; up to 5 per document (Think: 15) are read blind by a model. An audit can add doubt but never removes it.
- **Join across pages** (`src/table_join.py`): a table that ends its page is compared with the first table of the next one or two pages (figure-only pages are skipped), using column positions, a repeated header, the kind of content per column, serial numbers and "continued" markers; a different table title keeps tables apart. Merging maps columns, drops a repeated header, mends a row cut by the page break and drops a seam row printed twice. Uncertain joins get one picture of the seam for the model, or are queued for review.
- A table read by one engine only (the other never saw it, or filed it as text without a table block) is never verified: at least half of its cells having one reader means *needs review*. A table Bodhan wrote inside a text block is found and used as the second reading (`assemble.table_in_text`).
- Each table ends **verified**, **repaired** or **needs review**; `report.md` says how every cell was settled and what is left for review.

`--table-qa legacy` keeps the previous behaviour (number checks, Bodhan re-read, tie-breaker, page-pair joins).

### Modes (`--mode fast|normal|think`, `src/modes.py`)

| | Fast | Normal (default) | Think |
|---|---|---|---|
| Engines | Sarvam on every page, Bodhan on table pages | same | Bodhan on every page too |
| Free table check and joins | yes | yes | yes |
| Image check | none | format C: 24 questions a sheet, cell-only snippets where ruling lines give exact boxes | format A: 12 row strips a sheet; scanned ruled tables 2 of 3 models |
| Audits | 0 | up to 5 | up to 15 |
| Page text | one engine's | one engine's | both engines compared word by word (free; see below) |

The modes have no spending cap of their own (a cap leaves tables unchecked); `--max-cost-inr` is an explicit opt-in, and then tables with fewer open cells are checked first and the rest stay "needs review" with the primary engine's reading. A finished run can be re-assembled in another mode (`python -m src.assemble <dir> --mode think`); cached answers cost nothing.

**Page-text check (Think, `src/text_qa.py`, free).** With Bodhan on every page, the two readings of each page's prose are compared word by word after markup is stripped (Bodhan writes superscripts and inline maths as LaTeX, `$^{22}$` against Sarvam's `²²`). Differences are settled only by free evidence that picks between the two readings: words the other engine files as a header, figure text or table; spacing and punctuation; the PDF text layer when it has proven itself on the words both agree on. Everything else is listed in `report.md` under "Page text" for review; no model reads text yet, because there is no ground truth to measure one against (the Mozhi word crops could provide it). On the seven sample documents (11,000 prose words) 57 differences were left open (0.5%), most of them Telugu and Hindi-legacy spellings; Bodhan keeps line-break hyphens (`liberal- ism`), which the free rule ignores in favour of Sarvam's text.

### Page classes (`--doc-type auto|digital|scanned`, `src/page_class.py`)

Each page is classed on its own, because documents are often mixed (the Union budget has scanned pages 8–25 inside a digital PDF): **vector** (drawn text and lines) or **raster** (one image covers ≥ 80% of the page with almost no visible text), and the text layer as **unicode** (may prove itself as evidence), **legacy** (a non-Unicode font), **hidden_ocr** (invisible text over a scan, someone else's OCR) or **none**. Raster pages take the scanned route in the image check; only a unicode layer can settle cells. `--doc-type digital` or `scanned` forces every page; `scanned` also sets every text layer aside.

### Estimate before the run (`src/estimate.py`)

Before "Start?", a preflight reads up to 40 pages (one at a random place in each stretch of the document) in a few seconds, classes them, counts pages with tables, and prints a cost and time range for all three modes (the cost is the expected spend, not a ceiling) from measured coefficients (`src/estimate_coefficients.json`). After the run, the estimate and the actual cost and time are printed and appended to `out/estimates.jsonl`.

## Architecture report

`docs/Infinite-OCR-Architecture-Report.pdf` (77 pages) walks through every component next to the code: each section names the files and functions it describes with line numbers, followed by a generated code map, the configuration and a glossary. The same system is drawn in the online system map.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then fill in SARVAM_API_KEY, BODHAN_API_KEY, OPENROUTER_API_KEY
```

## Commands

### Run a document

```bash
python -m src.run samples/union-receipt-budget-2026-27-hindi.pdf --pages 90-101 --max-cost-inr 10
```

The command shows a cost estimate, runs the lanes with a progress line and ETA, then assembles the document. The main options:

| Option | Meaning |
|---|---|
| `-l / --language` | `auto` (default): languages found per page and per block. A code (`hi-IN`, `mr-IN`, `en-IN`, …) is only a hint: it breaks ties between same-script languages, and `en-IN` puts scanned pages in English mode (checked after OCR). |
| `--pages 120-135,200-210` | Process only these pages, for testing on windows of a long PDF |
| `--max-pages N` | Only the first N pages of the selection |
| `--max-cost-inr X` | Stop taking new work once this much has been spent, assembly included |
| `--resume` | Continue a previous run of the same PDF, after a crash, Ctrl-C, or to add pages |
| `--fresh` | Start over, keeping the cache, so pages already read cost nothing |
| `--mode` | `fast`, `normal` (default), `think`: see Modes above |
| `--doc-type` | `auto` (default, per page), `digital`, `scanned` |
| `--policy` | override the mode's engine routing: `sarvam-primary` (Fast, Normal), `both` (Think), `sarvam-only`, `bodhan-only` |
| `--qwen` | Also read English pages with Qwen through OpenRouter |
| `--no-assemble` / `--no-agent` / `--no-reocr` | Skip assembly / no LLM or vision calls / no Bodhan re-reads of tables |
| `--chunk-pages`, `--files-per-job`, `--max-open-jobs` | Sarvam job shape (defaults: 10 pages × 5 files per job, 3 jobs open) |
| `--dpi` | Render resolution for Bodhan (default 200) |

Ctrl-C once lets the requests in flight finish and saves state. Ctrl-C twice leaves at once. Either way, `--resume` picks up from there, re-attaching to Sarvam jobs already submitted instead of paying for them again.

### Re-assemble only

```bash
python -m src.assemble out/docs/<name>-<hash> [--mode fast|normal|think] [--no-agent] [--no-reocr] [--max-cost-inr X] [--table-qa standard]
```

Re-assembling costs nothing when nothing new needs deciding, because agent and vision answers are cached too.

### Tests (no API calls)

```bash
python -m unittest discover -s tests -t .
```

The 141 tests use fake engines and a temporary cache. They cover:

- the rate limiter;
- language identification, OCR modes and the English-mode re-read;
- error handling;
- triage and rotation;
- crash and resume;
- the cost cap;
- Bodhan taking pages from Sarvam's queue;
- table parsing and checks;
- joining tables across pages;
- a complete assembly;
- the standard table checker, image geometry, the image-check rule (with a fake model, including Think's 2 of 3) and joining;
- page classes on synthetic PDFs, the three modes, the estimate and the page-text check;
- the benchmark's generator, error planting and metrics.

### Table benchmark (`bench/`, no API calls unless stated)

Synthetic tables with known content, in Hindi, Marathi, Tamil, Telugu, Bengali and English, rendered to PDF by headless Chrome and copied as scan-like images. Two sets: `v1` (clean pages) and `v2` (dense 7.5 pt pages, budget-style wide tables, harsh scans).

```bash
python -m bench.run_bench --set v2 --checker standard --cell-error 0.03 0.12 0.25 --out bench/results/v2_standard.json
python -m bench.report bench/results/v2_standard.json          # accuracy, statuses, joins
python -m bench.compare bench/results/v2_baseline.json bench/results/v2_standard.json
python -m bench.real --set v2 --dry-run                          # real Sarvam + Bodhan on benchmark pages (paid; shows the cost first)
python -m bench.calibrate --set v2 [--image-check]               # real readings vs truth, both checkers (free; image check is paid, small)
python -m bench.pilot --dry-run                                  # model pilot on the real questions (shows the cost first)
python -m bench.pilot_report                                     # compare every model run in the pilot
python -m bench.geometry_eval                                    # ruling-line shape reading vs truth, v1 + v2 (free, ~4 min)
python -m bench.text_dispute_report out/docs/<run> ...           # what the page-text check found (run assembled with --mode think)
```

The engines in `run_bench` are simulated: each cell of the truth is misread with planted errors, some shared by both readers, so the checker can be measured against known mistakes. The vocabulary each set was rendered with is frozen in `bench/vocab_snapshot.json`, so a case regenerates exactly as it was printed.

### Week 1 commands (still available)

`python -m src.run_week1 …`, `python -m src.sarvam_digitise …`, `python -m src.bodhan_ocr …` and `python -m src.qwen_ocr …` are kept for comparison with the Week 1 report.

## Outputs

Each run goes to `out/docs/<name>-<first 8 characters of the PDF's SHA-256>/`:

| File | Contents |
|---|---|
| `doc.md` | The document: headings, text, HTML tables (merged cells kept), figures, footnotes, `<!-- page N -->` markers |
| `tables/<id>.html / .csv / .json` | Each table after joining. The JSON records its pages, the engine chosen, the checks, the fixes and the unresolved cells. |
| `figures/*.png` | Cropped figures and charts |
| `report.md` / `assembly.json` | Triage, layout labels seen, table checks and fixes, join decisions (with the agent's reason), printed page-number gaps, cost |
| `summary.json` | The engine run: pages per engine, 429s, retries, jobs, wall time, pages/min |
| `blocks/`, `md/` | Each engine's output per page in the shared block format, plus a markdown preview |
| `sarvam/`, `bodhan/` | Raw API results, including Sarvam job ZIPs |
| `manifest.sqlite`, `events.jsonl` | Run state and the event log |

## Test documents (`samples/`)

| File | Language | What it exercises | Source |
|---|---|---|---|
| `union-receipt-budget-2026-27-hindi.pdf` (106 pp) | Hindi | A 23-column table that continues over about 20 pages with merged headers; grand total; footnotes; pp. 8–25 scanned | https://www.indiabudget.gov.in/doc/rec/hallrec.pdf |
| `ncert-class12-maths-hindi-ch6.pdf` (41 pp) | Hindi + math | Equations, solved examples, exercises, MCQs, captioned figures; legacy-font text layer | https://ncert.nic.in/textbook/pdf/lhmh106.pdf |
| `economic-survey-2025-26-ch4-external-sector.pdf` (57 pp) | English | Charts, a table that continues over 3 pages, footnotes | https://www.indiabudget.gov.in/economicsurvey/doc/eschapter/echap04.pdf |
| `yojana-telugu-march-2025.pdf` (76 pp) | Telugu | Three-column magazine layout, infographics, charts, a merged-header table | https://www.publicationsdivision.nic.in/journals/Journalarchives/Yojana/Yojana-Telugu/2025/March/Yojana_Telugu_March_2025.pdf |
| `marathi-fiscal-report.pdf` (8 pp) | Marathi | Sideways fiscal tables with formula totals; legacy-font text layer | Week 1 sample |
| `mixed-language-test-page.pdf`, `…-scanned.pdf` (1 p each) | English, Tamil, Hindi, Marathi | Four languages on one page; the scanned copy has no text layer | Built from the samples above |
| `Ponniyan-Selvan.pdf`, `Patrick-Deneen-sample.pdf` (8 pp each) | Tamil, English | Book prose; drop caps; broken text layers | Week 1 samples |

### Results on the test windows (2026-10-10)

| Window | Pages | Cost | Wall time | Tables | Checks | Notes |
|---|---|---|---|---|---|---|
| Union budget, pp. 90–101 | 12 | ₹8.51 | 94 s | 12 → **1** (210 rows over 12 pages) | 198/199 | The repeated two-level header was dropped at every join; for page 101 the other engine's version was used so the join could go through. The one failed check is a ₹0.55 crore gap in a ministry subtotal. Every number involved matches the PDF's own text layer, so it is most likely an inconsistency in the published document. |
| NCERT maths, pp. 1–13 | 13 | ₹7.54 | 35 s | 4 | – | Bodhan labelled Equation, MCQ, Question, Solved-example and Diagram; Sarvam tagged 32 formulas |
| Economic Survey, pp. 11–18 | 8 | ₹4.61 | 19 s | 3 → **1** (pp. 16–18) | – | 4 charts cropped as figures |
| Yojana Telugu, pp. 50–57 | 8 | ₹4.20 | 14 s | 1 | – | Three-column text comes out in clean Unicode, where the PDF's own text layer is scrambled |
| Marathi fiscal | 8 | – | – | 8 | 100/100 | The text layer settled 3 disputed cells; pp. 2 and 5 show each engine misreading digits that the other got right |

There were no 429 rate-limit errors in any run.

## Limits and costs

| Engine | Limit | Price |
|---|---|---|
| Sarvam Vision | 10 requests/min per account (status polls count); 10 pages per PDF | ₹0.50/page |
| Bodhan indic-ocr | 8 requests per ~60 s (measured); 10 in parallel per key | ₹0.20/page |
| OpenRouter agent | `qwen/qwen3.5-9b` for decisions, `google/gemini-3.1-flash-lite` for reading cells | Spend logged in `out/openrouter_ledger.jsonl` and capped by `OPENROUTER_BUDGET_USD` |

## Moving to the RTX 5070 (Milestone 5)

The code is written to run on both machines; switching is done in `.env`. When it's time:

1. Copy the whole folder, including `cache/`, so pages already read cost nothing.
2. Create a fresh `.venv`.
3. Set `AGENT_BASE_URL=http://localhost:8080/v1` (`llama-server` running Qwen3.5-4B or 9B).
4. Set `BODHAN_BACKEND=local`. The lane then runs the open weights instead of `https://api.bodhan.ai`. On this Mac, `BODHAN_RUNTIME=mlx` uses the 8-bit Metal recognizer in `models/indic-ocr-mlx-8bit` (`hari31416/indic-ocr-mlx-8bit`; set `BODHAN_MLX_REPO=hari31416/indic-ocr-mlx-bf16` for BF16, which read the same text on 56 of 57 measured crops). Layout stays the official PyTorch checkpoint in `models/indic-ocr`. On the RTX 5070, set `BODHAN_RUNTIME=pytorch` so the official CUDA kernels run. Accept the model license on the Hub before the first download. The MLX lane keeps one copy of the weights and one continuous batch: crops from every page in flight join it as slots free (`BODHAN_MLX_BATCH`, default 16), and `BODHAN_MLX_PAGES` (default 6, or `--bodhan-workers`) pages are in flight. Layout and the batch take turns on the GPU one decode step at a time. Measured on a fanless M4 MacBook Air: 0.42 s per crop, 4.5–5.8 dense pages per minute (1.8 before continuous batching), lower when the chip is hot. Cost is ₹0. Bodhan still has no language field.

## Known limits

- **Scanned pages are not rotated automatically yet.** Upright detection uses the text layer.
- **Same-script languages are told apart by function words.** Hindi, Marathi and Nepali are reliable on prose. Dogri and Bodo have no word lists yet, and fall back to the `-l` hint or Hindi. Text with almost no words, such as an all-number table on its own page, takes the document's majority language.
- **Text-layer number checks work only where the PDF has a usable text layer.** On scanned pages, the formula checks, the comparison between engines, a re-read and the vision model are what remain.
- **Rounding tolerance can hide a last-digit error.** A sum check allows half a unit per summed term, so a one-unit error in a large total can pass.
- **The agent only decides a join when the rules call it ambiguous.** Its decisions and reasons are listed in `report.md` for review.
