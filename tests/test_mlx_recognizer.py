"""The MLX continuous batch, with the model replaced by a fake generator (no weights, no Metal)."""

import threading
import time
import unittest
from types import SimpleNamespace

from src.mlx_recognizer import MIN_TOKENS, _Engine, token_cap

STOP = 0
TABLE_PROMPT = "Convert this table image to HTML. Output only the HTML table."
TEXT_PROMPT = "Transcribe the text in this image. Write any mathematical expressions in LaTeX."


class FakeGen:
    """Each crop's prompt is its answer; one token per character, then the stop token."""

    def __init__(self, log: list) -> None:
        self.log = log
        self.seqs: dict[int, list[int]] = {}
        self.uid = 0
        self.max_live = 0
        self.limits: list[int] = []

    def insert(self, prompts, max_tokens, prompt_kwargs):
        self.limits.extend(max_tokens)
        uid = self.uid
        self.uid += 1
        self.seqs[uid] = list(prompt_kwargs[0]["answer"]) + [STOP]
        return [uid]

    def next(self):
        self.log.append(("step", time.perf_counter()))
        self.max_live = max(self.max_live, len(self.seqs))
        out = []
        for uid, rest in list(self.seqs.items()):
            token = rest.pop(0)
            done = token == STOP
            out.append(SimpleNamespace(uid=uid, token=token, finish_reason="stop" if done else None))
            if done:
                del self.seqs[uid]
        time.sleep(0.001)
        return [], out

    def close(self) -> None:
        pass


class FakeEngine(_Engine):
    def __init__(self, slots: int) -> None:
        self.log: list = []
        self.fake = FakeGen(self.log)
        detok = SimpleNamespace(text="")
        detok.reset = lambda: setattr(detok, "text", "")
        detok.add_token = lambda t: setattr(detok, "text", detok.text + chr(t))
        detok.finalize = lambda: None
        processor = SimpleNamespace(
            tokenizer=SimpleNamespace(eos_token_id=STOP, encode=lambda s, add_special_tokens=False: [STOP]),
            detokenizer=detok,
        )
        super().__init__(model=None, processor=processor, max_tokens=64, slots=slots)

    def _generator(self):
        return self.fake

    def _embed(self, item):
        if item.prompt == "bad":
            raise RuntimeError("bad crop")
        return SimpleNamespace(squeeze=lambda _: SimpleNamespace(tolist=lambda: [1])), {
            "answer": [ord(c) for c in item.prompt]
        }


def req(text: str, size: tuple[int, int] = (256, 256)):
    return SimpleNamespace(image=SimpleNamespace(size=size), prompt=text)


class EngineTests(unittest.TestCase):
    def test_pages_get_their_own_texts_in_order(self) -> None:
        engine = FakeEngine(slots=4)
        results = {}

        def page(name, texts):
            results[name] = engine.submit([req(t) for t in texts])

        threads = [
            threading.Thread(target=page, args=("a", ["one", "two", "three"])),
            threading.Thread(target=page, args=("b", ["x" * 40, "y", "zz"])),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(5)
        self.assertEqual(results["a"], ["one", "two", "three"])
        self.assertEqual(results["b"], ["x" * 40, "y", "zz"])
        self.assertLessEqual(engine.fake.max_live, 4)

    def test_short_crops_pass_a_long_one(self) -> None:
        engine = FakeEngine(slots=4)
        finished = {}

        def page(name, texts):
            engine.submit([req(t) for t in texts])
            finished[name] = time.perf_counter()

        long_page = threading.Thread(target=page, args=("long", ["L" * 300]))
        long_page.start()
        time.sleep(0.02)
        short = threading.Thread(target=page, args=("short", ["s", "t"]))
        short.start()
        short.join(5)
        long_page.join(5)
        self.assertLess(finished["short"], finished["long"])

    def test_a_bad_crop_fails_only_its_page(self) -> None:
        engine = FakeEngine(slots=4)
        with self.assertRaises(RuntimeError):
            engine.submit([req("ok"), req("bad")])
        self.assertEqual(engine.submit([req("fine")]), ["fine"])

    def test_layout_never_overlaps_a_step(self) -> None:
        engine = FakeEngine(slots=2)
        done = threading.Event()

        def page():
            engine.submit([req("p" * 200)])
            done.set()

        threading.Thread(target=page).start()
        time.sleep(0.02)
        engine.begin_layout()
        start = time.perf_counter()
        time.sleep(0.05)
        end = time.perf_counter()
        engine.end_layout()
        self.assertTrue(done.wait(5))
        steps = [t for kind, t in engine.log if kind == "step"]
        self.assertFalse([t for t in steps if start <= t <= end])


class TokenCapTests(unittest.TestCase):
    def test_small_crops_stop_at_the_floor(self) -> None:
        page_number = SimpleNamespace(size=(256, 256))
        self.assertEqual(token_cap(TEXT_PROMPT, page_number, 8192), MIN_TOKENS)
        self.assertEqual(token_cap(TABLE_PROMPT, page_number, 8192), MIN_TOKENS)

    def test_large_crops_get_more_and_tables_most(self) -> None:
        big = SimpleNamespace(size=(1536, 1536))
        text, table = token_cap(TEXT_PROMPT, big, 8192), token_cap(TABLE_PROMPT, big, 8192)
        self.assertGreater(text, 2048)
        self.assertEqual(table, 8192)

    def test_ceiling_wins_over_floor(self) -> None:
        self.assertEqual(token_cap(TEXT_PROMPT, SimpleNamespace(size=(256, 256)), 64), 64)

    def test_the_engine_gives_each_crop_its_own_limit(self) -> None:
        engine = FakeEngine(slots=1)
        engine.max_tokens = 8192
        engine.submit([req("a"), req("b", (1536, 1536))])
        self.assertEqual(engine.fake.limits, [MIN_TOKENS, 24 * 1536 * 1536 // 10_000])

    def test_prompts_still_match_the_vendored_contract(self) -> None:
        try:
            from idp_contract import prompt_for
        except ImportError:
            self.skipTest("models/indic-ocr is not on the path")
        crop = SimpleNamespace(size=(1536, 1536))
        self.assertEqual(token_cap(prompt_for("Table"), crop, 8192), 8192)
        self.assertLess(token_cap(prompt_for("Text"), crop, 8192), 8192)
        self.assertLess(token_cap(prompt_for("Equation"), crop, 8192), 8192)


class CloseTests(unittest.TestCase):
    def test_close_stops_the_thread_and_unblocks_waiters(self) -> None:
        engine = FakeEngine(slots=1)
        outcome = {}

        def page():
            try:
                engine.submit([req("p" * 5000)])
            except RuntimeError as exc:
                outcome["error"] = str(exc)

        waiter = threading.Thread(target=page)
        waiter.start()
        time.sleep(0.05)
        engine.close()
        waiter.join(5)
        self.assertFalse(engine._thread.is_alive())
        self.assertIn("closed", outcome["error"])
        engine.close()  # idempotent


if __name__ == "__main__":
    unittest.main()
