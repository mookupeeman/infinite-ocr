"""vLLM recognizer against a stub OpenAI server. No vLLM, no GPU, no weights."""

import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

from src.vllm_recognizer import VllmRecognizer, VllmUnavailable, invention_starts, token_cap


class Req:
    def __init__(self, prompt: str, width: int = 8, height: int = 8) -> None:
        self.image = Image.new("RGB", (width, height), "white")
        self.prompt = prompt


class Stub(BaseHTTPRequestHandler):
    seen: list = []
    reply: dict | None = None  # a fixed reply instead of echoing the prompt

    def log_message(self, *args) -> None:
        pass

    def _send(self, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        self._send({"data": [{"id": "indic-ocr"}]})

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Stub.seen.append(body)
        prompt = body["messages"][0]["content"][1]["text"]
        # Later requests answer first, so order can only come from the client.
        time.sleep(0.05 if prompt == "first" else 0.0)
        reply = Stub.reply or {"content": f" {prompt} \n"}
        choice = {"finish_reason": reply.get("finish", "stop"), "message": {"content": reply["content"]}}
        for key in ("token_ids", "logprobs"):
            if key in reply:
                choice[key] = reply[key]
        self._send({"choices": [choice]})


class VllmRecognizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/v1"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()

    def test_texts_come_back_in_request_order_and_stripped(self) -> None:
        rec = VllmRecognizer(self.url, max_tokens=4096)
        texts = rec.transcribe([Req("first"), Req("second"), Req("third")])
        self.assertEqual(texts, ["first", "second", "third"])
        self.assertEqual(rec.last_n_crops(), 3)
        rec.close()

    def test_request_carries_image_then_prompt_and_greedy_cap(self) -> None:
        Stub.seen.clear()
        rec = VllmRecognizer(self.url, max_tokens=4096)
        rec.transcribe([Req("only", width=2048, height=1024)])
        body = Stub.seen[0]
        content = body["messages"][0]["content"]
        self.assertEqual([part["type"] for part in content], ["image_url", "text"])
        self.assertTrue(content[0]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual((body["model"], body["max_tokens"], body["temperature"]), ("indic-ocr", 4096, 0.0))
        rec.close()

    def test_token_cap_follows_crop_size(self) -> None:
        self.assertEqual(token_cap(256, 256, 8192), 512)  # page number: a loop stops early
        self.assertEqual(token_cap(1475, 429, 8192), 2468)  # measured 1312
        self.assertEqual(token_cap(1329, 1776, 8192), 8192)  # measured 6402
        self.assertEqual(token_cap(256, 256, 300), 300)

    def test_text_is_decoded_from_token_ids_not_taken_from_the_server(self) -> None:
        """vLLM's own detokenizer wrote U+FFFD for a multi-byte character; the ids decode correctly."""

        class Tok:
            def decode(self, ids, skip_special_tokens):
                assert skip_special_tokens
                return "8\u00d7A100" if ids == [7, 8] else "?"

        Stub.reply = {"content": "8\ufffdA100", "token_ids": [7, 8]}
        self.addCleanup(setattr, Stub, "reply", None)
        rec = VllmRecognizer(self.url, tokenizer=Tok())
        self.assertEqual(rec.transcribe([Req("only")]), ["8\u00d7A100"])
        self.assertTrue(Stub.seen[-1]["return_token_ids"])
        rec.close()
        # no ids in the reply (an older server) or no tokenizer: the server's text is used as it is
        rec = VllmRecognizer(self.url, tokenizer=Tok())
        Stub.reply = {"content": "plain"}
        self.assertEqual(rec.transcribe([Req("only")]), ["plain"])
        rec.close()

    def test_invented_text_is_cut_where_the_models_confidence_drops(self) -> None:
        sure = [0.0] * 50
        self.assertEqual(invention_starts(sure + [-1.2, -0.8, -1.4, -0.9, -1.1, -0.7, -0.2]), 50)
        self.assertEqual(invention_starts(sure + [-0.6, -0.9, -0.1, -0.8, -1.0, -0.5]), 50)
        self.assertIsNone(invention_starts(sure))
        self.assertIsNone(invention_starts(sure[:20] + [-0.9] + sure[20:]))  # one unsure character is not invention
        self.assertIsNone(invention_starts([-2.0, -2.0]))  # fewer tokens than one window

    def test_a_text_block_that_ran_to_the_cap_is_trimmed_but_a_table_is_not(self) -> None:
        class Tok:
            def decode(self, ids, skip_special_tokens):
                return "".join(chr(97 + i % 26) for i in ids)

        ids = list(range(60))
        logprobs = [{"logprob": 0.0}] * 40 + [{"logprob": -1.0}] * 20
        Stub.reply = {"content": "x", "token_ids": ids, "logprobs": {"content": logprobs}, "finish": "length"}
        self.addCleanup(setattr, Stub, "reply", None)
        rec = VllmRecognizer(self.url, tokenizer=Tok())
        text_prompt = "Transcribe the text in this image."
        self.assertEqual(len(rec.transcribe([Req(text_prompt)])[0]), 40)  # cut at the first unsure token
        self.assertTrue(Stub.seen[-1]["logprobs"])
        table = rec.transcribe([Req("Convert this table image to HTML.")])[0]
        self.assertEqual(len(table), 60)  # a table that ran to the cap is left as it is
        self.assertNotIn("logprobs", Stub.seen[-1])
        Stub.reply = {"content": "x", "token_ids": ids, "logprobs": {"content": logprobs}, "finish": "stop"}
        self.assertEqual(len(rec.transcribe([Req(text_prompt)])[0]), 60)  # finished by itself: never trimmed
        rec.close()

    def test_wrong_model_name_is_unavailable(self) -> None:
        with self.assertRaises(VllmUnavailable):
            VllmRecognizer(self.url, model="something-else")

    def test_no_server_is_unavailable_and_stops_the_lane(self) -> None:
        from unittest import mock

        from src import indic_local
        from src.lanes import BodhanLocalBackend
        from src.rate_limit import FatalError

        with self.assertRaises(VllmUnavailable):
            VllmRecognizer("http://127.0.0.1:9/v1")

        def down(image, mime, table_format):
            raise VllmUnavailable("no server")

        with mock.patch.object(indic_local, "local_ocr", down):
            with self.assertRaises(FatalError) as caught:
                BodhanLocalBackend("html").ocr(b"png", "image/png", None)
        self.assertEqual(caught.exception.outcome.reason, "missing_runtime")


if __name__ == "__main__":
    unittest.main()
