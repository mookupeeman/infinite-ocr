"""Local IndicOCR payload matches the block reader. No weights and no torch."""

import unittest

from src.blocks import from_bodhan
from src.indic_local import payload_from_page


class PayloadTests(unittest.TestCase):
    def test_blocks_keep_pixel_boxes(self) -> None:
        page = {
            "markdown": "Hello",
            "blocks": [
                {
                    "order": 0,
                    "label": "Paragraph",
                    "type": "Text",
                    "bbox_xyxy": [10, 20, 110, 80],
                    "conf": 0.9,
                    "text": "Hello",
                }
            ],
        }
        payload = payload_from_page(page, table_format="html", mime="image/png")
        self.assertEqual(payload["choices"][0]["message"]["content"], "Hello")
        self.assertEqual(payload["_infinite"]["backend"], "local")
        blocks = from_bodhan(payload, 1, 200, 400)
        self.assertEqual(blocks[0].kind, "text")
        self.assertEqual(blocks[0].text, "Hello")
        self.assertEqual(blocks[0].bbox, (0.05, 0.05, 0.55, 0.2))

    def test_word_boundary_mark_becomes_a_space(self) -> None:
        page = {
            "markdown": "भारत▁सरकार",
            "blocks": [{"order": 0, "label": "Paragraph", "text": "वित्त▁मंत्रालय"}],
        }
        payload = payload_from_page(page, table_format="html", mime="image/png")
        self.assertEqual(payload["choices"][0]["message"]["content"], "भारत सरकार")
        self.assertEqual(payload["blocks"][0]["text"], "वित्त मंत्रालय")


class BackendTests(unittest.TestCase):
    """The lane's local backend, with the model replaced by a stub (no weights, no MLX)."""

    def test_a_successful_page_is_returned_and_reported(self) -> None:
        from unittest import mock

        from src import indic_local
        from src.lanes import BodhanLocalBackend
        from src.rate_limit import Verdict

        page = {"markdown": "Hello", "blocks": [{"order": 0, "label": "Paragraph", "bbox_xyxy": [0, 0, 10, 10], "text": "Hello"}]}
        seen = []
        with mock.patch.object(indic_local, "local_ocr",
                               lambda image, mime, table_format: payload_from_page(page, table_format=table_format, mime=mime)):
            payload = BodhanLocalBackend("html").ocr(b"png", "image/png", seen.append)
        self.assertEqual(payload["blocks"][0]["text"], "Hello")
        self.assertEqual([o.verdict for o in seen], [Verdict.OK])

    def test_a_missing_runtime_stops_the_lane(self) -> None:
        from unittest import mock

        from src import indic_local
        from src.lanes import BodhanLocalBackend
        from src.rate_limit import FatalError

        def broken(*args, **kwargs):
            raise ImportError("No module named 'mlx_vlm'")

        with mock.patch.object(indic_local, "local_ocr", broken):
            with self.assertRaises(FatalError) as ctx:
                BodhanLocalBackend("html").ocr(b"png", "image/png", None)
        self.assertEqual(ctx.exception.outcome.reason, "missing_runtime")


if __name__ == "__main__":
    unittest.main()
