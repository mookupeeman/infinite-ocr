"""Language handling: identification, OCR mode at triage, the English-mode safety net."""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pymupdf

from src import cache, sarvam_digitise
from src.langid import detect, page_languages, strip_markup
from src.lanes import Settings
from src.manifest import Manifest
from src.run import Run
from src.triage import triage_page
from tests.fakes import FakeBodhan, FakeSarvam, make_pdf

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"

HINDI = "इसके अतिरिक्त, यदि दो राशियाँ x और y के सापेक्ष परिवर्तित हो रही हैं तो यह नियम लागू होता है।"
MARATHI = "कर्मचाऱ्यांची संख्या व वेतनावरील खर्च यांचे विवरणपत्र आहे आणि ही माहिती मार्च अखेरीच्या आकडेवारीनुसार आहे."
NEPALI = "यो योजना गर्न सरकारले पनि निर्णय गरेको छ र काम सुरु भएको छ।"


class DetectTest(unittest.TestCase):
    def test_same_script_languages(self):
        self.assertEqual(detect(HINDI).dominant, "hi-IN")
        self.assertEqual(detect(MARATHI).dominant, "mr-IN")
        self.assertEqual(detect(NEPALI).dominant, "ne-IN")

    def test_scripts_with_one_language_and_letter_cues(self):
        self.assertEqual(detect("அத்தியாயம் 1 - ஆடித்திருநாள்").dominant, "ta-IN")
        self.assertEqual(detect("মোট ৰাজহ আৰু ৱাৰ্ষিক").dominant, "as-IN")  # Assamese ৰ ৱ
        self.assertEqual(detect("মোট রাজস্ব এবং বার্ষিক").dominant, "bn-IN")
        self.assertEqual(detect("ڪراچي ۾ سنڌي ٻولي").dominant, "sd-IN")  # Sindhi letters
        self.assertEqual(detect("اردو زبان میں کتاب").dominant, "ur-IN")

    def test_markup_is_not_english(self):
        self.assertEqual(strip_markup("<table><tr><td>एकूण</td></tr></table> $\\frac{dy}{dx}$").split(), ["एकूण"])
        self.assertEqual(detect("<table><tr><td>तमिल</td></tr></table>").shares, {"hi-IN": 1.0})

    def test_mixed_page_block_by_block(self):
        texts = ["and its defense and realization have been a primary goal",
                 "அத்தியாயம் 1 - ஆடித்திருநாள்", HINDI, "$$ \\frac{dy}{dx} $$",
                 "कर्मचाऱ्यांची संख्या व वेतनावरील खर्च यांचे विवरणपत्र",   # Marathi caption: 2 markers
                 "<table><tr><td>शासन</td><td>४९१४६६</td></tr></table>",    # table: no markers
                 "टीप - वरील माहिती वर्ष २०२१-२२ च्या मार्च अखेरीच्या आकडेवारीनुसार"]
        kinds = ["text", "heading", "text", "math", "caption", "table", "footnote"]
        per, page = page_languages(texts, kinds=kinds)
        self.assertEqual([p.dominant for p in per], ["en-IN", "ta-IN", "hi-IN", None, "mr-IN", "mr-IN", "mr-IN"])
        self.assertEqual(set(page.present(0.05)), {"en-IN", "ta-IN", "hi-IN", "mr-IN"})


class TriageModeTest(unittest.TestCase):
    def mode(self, name, language, page=0):
        flags = triage_page(pymupdf.open(SAMPLES / name)[page], language)
        return flags.ocr_mode, flags.mode_source, flags.language

    def test_modes(self):
        self.assertEqual(self.mode("Patrick-Deneen-sample.pdf", "auto"), ("en", "text_layer", "en-IN"))
        self.assertEqual(self.mode("mixed-language-test-page.pdf", "auto"), ("indic", "text_layer", "auto"))
        self.assertEqual(self.mode("mixed-language-test-page-scanned.pdf", "auto"), ("indic", "default", "auto"))
        self.assertEqual(self.mode("mixed-language-test-page-scanned.pdf", "en-IN"), ("en", "user", "en-IN"))
        # A wrong -l en-IN cannot push a Tamil page into English mode.
        self.assertEqual(self.mode("Ponniyan-Selvan.pdf", "en-IN"), ("indic", "text_layer", "auto"))
        self.assertEqual(self.mode("marathi-fiscal-report.pdf", "mr-IN"), ("indic", "text_layer", "mr-IN"))


class SubmitValidationTest(unittest.TestCase):
    def test_invalid_codes_never_reach_the_api(self):
        with mock.patch.object(sarvam_digitise, "_client") as client:
            for bad in ("auto", "hi-IN,en-IN", "mul", "hindi"):
                with self.assertRaises(ValueError):
                    sarvam_digitise.submit([("a.pdf", b"%PDF")], bad)
            client.assert_not_called()

    def test_none_omits_the_field(self):
        with mock.patch.object(sarvam_digitise, "_client") as client:
            client.return_value.doc_ai.digitise.return_value.job_id = "j1"
            self.assertEqual(sarvam_digitise.submit([("a.pdf", b"%PDF")], None), "j1")
            self.assertNotIn("language", client.return_value.doc_ai.digitise.call_args.kwargs)


class ManifestMigrationTest(unittest.TestCase):
    def test_old_run_database_gets_language_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.sqlite"
            db = sqlite3.connect(path)
            db.execute("CREATE TABLE pages(page INTEGER PRIMARY KEY, rotate INTEGER, language TEXT)")
            db.execute("INSERT INTO pages(page, rotate, language) VALUES(1, 0, 'mr-IN')")
            db.commit()
            db.close()
            m = Manifest(path)
            info = m.page_info(1)
            self.assertIn("ocr_mode", info)
            self.assertEqual(info["mode_retried"], 0)
            m.update_detected(1, "sarvam", {"shares": {"mr-IN": 1.0}})
            self.assertEqual(m.page_info(1)["detected"], {"sarvam": {"shares": {"mr-IN": 1.0}}})
            m.close()


class ModeRetryTest(unittest.TestCase):
    def test_english_mode_page_that_comes_back_hindi_is_reread_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(cache, "CACHE_DIR", root / "cache"):
                pdf = make_pdf(root / "scan.pdf", 2)  # almost no text layer: mode comes from -l
                sarvam = FakeSarvam(body_text=HINDI)
                run = Run(pdf, Settings(language="en-IN", policy="sarvam-only"), out_root=root / "out",
                          sarvam=sarvam, bodhan_backend=FakeBodhan(), progress_every=999, verbose=False)
                run.prepare()
                summary = run.execute()
                self.assertTrue(summary["complete"], summary)
                self.assertEqual([s["language"] for s in sarvam.submissions], ["en-IN", None])  # None = field omitted (Indic mode)
                m = Manifest(Path(summary["out_dir"]) / "manifest.sqlite")
                pages = m.all_pages()
                self.assertTrue(all(p["ocr_mode"] == "indic" and p["mode_retried"] for p in pages))
                self.assertEqual(pages[0]["detected"]["sarvam"]["dominant"], "hi-IN")
                self.assertEqual(m.extras_by_kind()["sarvam_mode_retry"]["calls"], 2)
                m.close()


if __name__ == "__main__":
    unittest.main()
