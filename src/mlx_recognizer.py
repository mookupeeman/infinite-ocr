"""Stage 2 recognizer for Apple Silicon.

One BF16 MLX model and one continuous batch, like the vLLM path upstream. Every
page's crops join the running batch as soon as a slot frees; a crop leaves the
batch when it stops. Short crops never wait for a long table to finish.

Layout (PyTorch on Metal) and this batch take turns on the GPU, one decode step
(~60 ms) at a time. Running both frameworks at once slowed layout 4x and
sometimes failed with a Metal out-of-memory error.

Measured on an M4 (57 crops, Hindi annual report): one crop at a time 1.56 s per
crop; 16 slots 0.57 s per crop; 56 of 57 texts identical to one at a time.
`mlx_vlm.batch_generate` is not used: it groups crops by exact pixel size, and
block crops almost never share one, so its batches ran one crop at a time.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from pathlib import Path

from src.config import BODHAN_MLX_BATCH

logger = logging.getLogger("mlx_recognizer")


class _Job:
    def __init__(self, n: int) -> None:
        self.n = n
        self.texts = [""] * n
        self.done = 0
        self.error: BaseException | None = None
        self.event = threading.Event()
        self.lock = threading.Lock()


class _Item:
    def __init__(self, image: object, prompt: str, job: _Job, index: int) -> None:
        self.image = image
        self.prompt = prompt
        self.job = job
        self.index = index


class _Engine:
    """One thread owns the model. It admits crops into free slots and steps the batch."""

    def __init__(self, model: object, processor: object, max_tokens: int, slots: int) -> None:
        self.model = model
        self.processor = processor
        self.max_tokens = max_tokens
        self.slots = max(1, slots)
        self._state = threading.Condition()
        self._q: deque[_Item] = deque()
        self._gpu = threading.Lock()
        self._layouts_waiting = 0
        self._gen = None
        self._active: dict[int, tuple[_Item, list[int]]] = {}
        self._stops = self._stop_ids()
        self._thread = threading.Thread(target=self._loop, name="mlx-engine", daemon=True)
        self._thread.start()

    def begin_layout(self) -> None:
        with self._state:
            self._layouts_waiting += 1
        self._gpu.acquire()
        with self._state:
            self._layouts_waiting -= 1

    def end_layout(self) -> None:
        self._gpu.release()
        with self._state:
            self._state.notify_all()

    def submit(self, requests: list) -> list[str]:
        if not requests:
            return []
        job = _Job(len(requests))
        with self._state:
            for index, req in enumerate(requests):
                self._q.append(_Item(req.image, req.prompt, job, index))
            self._state.notify_all()
        job.event.wait()
        if job.error is not None:
            raise job.error
        return job.texts

    def _loop(self) -> None:
        while True:
            with self._state:
                while self._layouts_waiting or (not self._q and not self._active):
                    self._state.wait(timeout=0.5)
                admit = []
                while self._q and len(self._active) + len(admit) < self.slots:
                    item = self._q.popleft()
                    if item.job.error is None:
                        admit.append(item)
                    else:
                        self._finish(item, error=item.job.error)
            with self._gpu:
                for item in admit:
                    self._admit(item)
                if not self._active:
                    continue
                try:
                    self._step()
                except BaseException as exc:  # noqa: BLE001 - every waiting page must unblock
                    logger.exception("mlx batch step failed")
                    self._reset(exc)

    def _stop_ids(self) -> set[int]:
        """Stop on the chat end token. The checkpoint EOS is an ordinary word piece."""
        tok = self.processor.tokenizer
        found: list[int] = []
        eos = getattr(tok, "eos_token_id", None)
        if isinstance(eos, int):
            found.append(eos)
        elif isinstance(eos, (list, tuple)):
            found.extend(int(item) for item in eos)
        for piece in ("<|im_end|>", "<|endoftext|>"):
            encoded = tok.encode(piece, add_special_tokens=False)
            if encoded:
                found.append(int(encoded[-1]))
        return set(found)

    def _generator(self):
        if self._gen is None:
            from mlx_vlm.generate.ar import BatchGenerator

            self._gen = BatchGenerator(
                self.model.language_model,
                self.processor,
                stop_tokens=self._stops,
                max_tokens=self.max_tokens,
                completion_batch_size=self.slots,
                prefill_batch_size=min(self.slots, 8),
                compute_logprobs=False,
            )
        return self._gen

    def _embed(self, item: _Item):
        """Chat template, image patches and vision encoder for one crop."""
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import prepare_inputs, should_add_special_tokens

        config = self.model.config
        prompt = apply_chat_template(self.processor, config, item.prompt, num_images=1)
        inputs = prepare_inputs(
            self.processor,
            images=[item.image],
            prompts=[prompt],
            image_token_index=getattr(config, "image_token_index", None),
            add_special_tokens=should_add_special_tokens(config.model_type, self.processor),
        )
        ids = inputs["input_ids"]
        data = {k: v for k, v in inputs.items() if k not in ("input_ids", "pixel_values", "attention_mask")}
        embedded = self.model.get_input_embeddings(
            ids, inputs.get("pixel_values"), mask=inputs.get("attention_mask"), **data
        )
        return ids, {**data, **{k: v for k, v in embedded.to_dict().items() if v is not None}}

    def _admit(self, item: _Item) -> None:
        try:
            ids, kwargs = self._embed(item)
            (uid,) = self._generator().insert(
                [ids.squeeze(0).tolist()], max_tokens=self.max_tokens, prompt_kwargs=[kwargs]
            )
        except BaseException as exc:  # noqa: BLE001 - one bad crop fails its page only
            self._finish(item, error=exc)
            return
        self._active[uid] = (item, [])

    def _step(self) -> None:
        _, responses = self._generator().next()
        for r in responses:
            entry = self._active.get(r.uid)
            if entry is None:
                continue
            item, tokens = entry
            if r.finish_reason != "stop" and r.token is not None:
                tokens.append(int(r.token.item() if hasattr(r.token, "item") else r.token))
            if r.finish_reason is not None:
                del self._active[r.uid]
                if r.finish_reason == "length":
                    logger.info("mlx crop hit max_tokens=%d", self.max_tokens)
                self._finish(item, text=self._detok(tokens))

    def _detok(self, tokens: list[int]) -> str:
        detok = self.processor.detokenizer
        detok.reset()
        for token in tokens:
            detok.add_token(token)
        detok.finalize()
        return detok.text.strip()

    def _reset(self, exc: BaseException) -> None:
        for item, _ in self._active.values():
            self._finish(item, error=exc)
        self._active.clear()
        if self._gen is not None:
            try:
                self._gen.close()
            except Exception:  # noqa: BLE001
                pass
        self._gen = None
        import mlx.core as mx

        mx.clear_cache()

    def _finish(self, item: _Item, text: str | None = None, error: BaseException | None = None) -> None:
        job = item.job
        with job.lock:
            if error is not None and job.error is None:
                job.error = error
            if text is not None:
                job.texts[item.index] = text
            job.done += 1
            if job.error is not None or job.done == job.n:
                job.event.set()


class MlxRecognizer:
    """Transcribe block crops through one shared continuous batch."""

    def __init__(self, model_path: str | Path, max_tokens: int = 2048, batch_size: int | None = None) -> None:
        from mlx_vlm import load

        self.model_path = Path(model_path)
        self.max_tokens = max_tokens
        self._tls = threading.local()
        self._layout = threading.Lock()
        logger.info("Loading MLX recognizer from %s", self.model_path)
        started = time.perf_counter()
        model, processor = load(str(self.model_path))
        self._engine = _Engine(
            model,
            processor,
            max_tokens,
            BODHAN_MLX_BATCH if batch_size is None else batch_size,
        )
        logger.info("MLX recognizer ready in %.1f s", time.perf_counter() - started)

    def begin_layout(self) -> None:
        # Layouts take turns with each other, then with the batch, one step at a time.
        self._layout.acquire()
        try:
            self._engine.begin_layout()
        except BaseException:
            self._layout.release()
            raise

    def end_layout(self) -> None:
        try:
            self._engine.end_layout()
        finally:
            self._layout.release()

    def transcribe(self, requests: list) -> list[str]:
        # Plain prompts: the engine applies the chat template once.
        self._tls.n_crops = len(requests)
        return self._engine.submit(requests)

    def last_n_crops(self) -> int:
        return int(getattr(self._tls, "n_crops", 0))

    def close(self) -> None:
        self._engine.model = None
        self._engine.processor = None
