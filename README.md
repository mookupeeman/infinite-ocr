# Infinite-OCR

OCR for PDFs of any length in English and the Indian languages, with a checker that decides how much to trust each table cell. Two OCR engines read every document, **Sarvam Vision** and **Bodhan IndicOCR**. Free checks and, optionally, vision models settle the places where they disagree. A local web app runs it all on your own machine.

## Key features

- **Two readers, one checker.** Their tables are lined up cell by cell. Differences are settled by printed formulas, ruling lines in the page image and the PDF's own text layer. A vision model is asked only about cells still in doubt and can never invent a value. Each table ends as *verified*, *repaired* or *needs review*.
- **Built for hard Indian documents:** sideways budget tables, tables that continue across pages, legacy-font PDFs, scans, and several languages on one page. Languages are identified per block, with no hint needed.
- **Three effort levels:** Fast, Medium (`normal`) and High (`think`). See the table below.
- **Runs locally.** Bodhan runs on your own GPU: an NVIDIA GPU through vLLM, or Apple Silicon through MLX. There is no per-page fee and no rate limit. On an RTX 5070 laptop it reads 23 to 42 pages a minute, against 8 through the hosted API.
- **Local web app.** Upload a PDF or a photo, pick the effort, watch progress, and inspect every page: the original and the digitised text side by side, with the detected blocks. The model sleeps between documents, so the GPU is free when you are not reading.
- **Safe to interrupt and repeat.** Runs resume after a crash, spending can be capped, and every result is cached, so a re-run costs nothing.

## Setup

Needs Python 3.12. The hosted Sarvam API is used in every setup.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then fill in the keys below
```

| Key in `.env` | Needed for |
|---|---|
| `SARVAM_API_KEY` | Sarvam Vision, the second reader (₹0.50 a page) |
| `OPENROUTER_API_KEY` | the image check of Medium and High |
| `BODHAN_API_KEY` | only if you use Bodhan's hosted API instead of running it locally |

**Run Bodhan locally.** Choose one:

- **NVIDIA GPU (Linux), recommended.** Install PyTorch for your CUDA version, then the model libraries, and vLLM in its own environment. vLLM pins its own PyTorch, so it must not share one with the rest.
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  pip install transformers accelerate
  python3.12 -m venv .venv-vllm && .venv-vllm/bin/pip install vllm==0.32.0
  ```
  In `.env`: `BODHAN_BACKEND=local` and `BODHAN_RUNTIME=vllm`. The model weights (`bodhan-ai/indic-ocr`) download on first use into `models/indic-ocr`. Accept the model's licence on Hugging Face and run `hf auth login` first.
- **Apple Silicon.** `pip install mlx-vlm torch torchvision transformers accelerate`, then in `.env`: `BODHAN_BACKEND=local` and `BODHAN_RUNTIME=mlx`.
- **Hosted API.** Leave `BODHAN_BACKEND=api` and set `BODHAN_API_KEY`. It is limited to 8 requests a minute.

## Run
**Initialize the vLLM local model for CUDA. Not needed for MLX model**

```bash
./serve-bodhan-vllm.sh                  # Start vLLM server on http://127.0.0.1:8000
```
**Web app:**

```bash
./start-web.sh                  # opens http://127.0.0.1:8765
```

Drop a file, choose the effort, press *Start reading*. The model loads onto the GPU only then (about a minute the first time) and sleeps when the job ends.

**Command line:**

```bash
python -m src.run samples/marathi-fiscal-report.pdf --mode normal
python -m src.run FILE.pdf --pages 10-30 --max-cost-inr 20 --lead bodhan
python -m src.run --help
```
--mode decides the effort level. --lead decides which out of Bodhan or Sarvam will be the lead engines. --pages allows you to run the pipeline on specific pages.  --max-cost-inr caps the total cost of running. 
Results go to `out/docs/<name>-<hash>/`: `doc.md` (the document), `tables/` (HTML, CSV and JSON for each table), `figures/` and `report.md` (what was checked, settled and left for review).

## Effort levels

| CLI mode | Web | Who reads | Checking |
|---|---|---|---|
| `fast` | Fast | one engine reads every page, the other reads table pages | free checks only, no model calls |
| `normal` | Medium | same | adds a vision-model check of cells still in doubt, and a few audits |
| `think` | High | both engines read every page | adds a third model on scans, more audits, and a word-by-word comparison of the two readings |

With a local GPU, Bodhan is the engine that reads every page (`--lead bodhan`, the default there). Otherwise Sarvam leads.

## Project layout

```
src/        the pipeline
  run.py                       command line; stage 1: triage, engine lanes, blocks
  triage.py, page_class.py     free per-page checks: rotation, scripts, tables, scanned or digital
  lanes.py, manifest.py        the Sarvam and Bodhan lanes, and the SQLite state behind resume
  sarvam_digitise.py, bodhan_ocr.py, indic_local.py, vllm_recognizer.py, mlx_recognizer.py   engine clients
  assemble.py                  stage 2: builds doc.md and the report
  tables.py, table_*.py        table parsing, cell-by-cell checking, image check, joining across pages
  langid.py, text_qa.py        language identification, comparison of the two engines' prose
  modes.py, estimate.py        the effort levels, and the cost and time estimate
web/        the local web app: server.py (API), jobs.py, gpu.py (model sleep and wake), static/ (the page)
bench/      our own table benchmark (tables we wrote, so the answers are known), and its scoring
tests/      quick code checks: no API calls, no GPU
samples/    public PDFs used for testing
docs/       architecture report (PDF)
serve-bodhan-vllm.sh   starts the local model server (the web app does this for you)
```

## Benchmark (`bench/`) and tests (`tests/`)

**Why we built our own benchmark.** To score an OCR tool you need the right answer to compare against. For most Indian-language PDFs we did not have one: no clean, checked copy of the table exists. So `bench/` makes its own.

- **We write the table first.** `bench/gen.py` invents small tables (names, amounts, dates, merged headers, and so on) in Hindi, Marathi, Tamil, Telugu, Bengali and English. Because we wrote every cell, we know the correct answer for every cell.
- **Then we turn it into a PDF.** `bench/render.py` draws the table as a real-looking PDF, and also as a scan-like copy, so we can see how much a scan hurts.
- **Then we run the pipeline and mark it.** `bench/metrics.py` compares what the pipeline read against what we wrote, and gives a score for how many cells are right and how well the table shape was kept.
- **Pretend readers, no cost.** `bench/engines.py` can stand in for Sarvam and Bodhan and plant known mistakes in their answers. This lets us test the checker for free, including the case where both readers make the same mistake, which agreement alone cannot catch. `python -m bench.run_bench` runs this.
- **Real readers, a small paid run.** `bench/real.py` sends the benchmark pages to the real Sarvam and Bodhan. `bench/calibrate.py` then compares the real mistakes with what the pretend readers assumed, and `bench/pilot.py` helps pick the vision model that answers the checker's questions best for the price.
- **A real book as the answer key.** `bench/wikisource_eval.py` scores a finished run against a proofread Wikisource book, which gives a true answer for real scanned pages.

Results are saved in `bench/results/`. Benchmark PDFs are in `bench/real/`. Only `bench/real.py` and `bench/pilot.py` spend money, and both have a `--dry-run` that shows the cost first.

**What `tests/` is for.** `tests/` are quick checks that the code still works. They use no API calls and no GPU, so they cost nothing and can run any time. They are not a measure of OCR accuracy. `bench/` does that.

```bash
python -m unittest discover -s tests -t .
```

## Credits and licences

- Built with IndicOCR from Bodhan AI / AI4Bharat, under the Indic Open Model License.
- Sarvam Vision is the second reader.
- Fonts: Montserrat and Noto Sans (SIL Open Font License, in `web/static/fonts`).
