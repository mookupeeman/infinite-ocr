"""The MLX continuous batch, with the model replaced by a fake generator (no weights, no Metal)."""

import threading
import time
import unittest
from types import SimpleNamespace

from src.mlx_recognizer import _Engine

STOP = 0


class FakeGen:
    """Each crop's prompt is its answer; one token per character, then the stop token."""

    def __init__(self, log: list) -> None:
        self.log = log
        self.seqs: dict[int, list[int]] = {}
        self.uid = 0
        self.max_live = 0

    def insert(self, prompts, max_tokens, prompt_kwargs):
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


def req(text: str):
    return SimpleNamespace(image=None, prompt=text)


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


if __name__ == "__main__":
    unittest.main()
