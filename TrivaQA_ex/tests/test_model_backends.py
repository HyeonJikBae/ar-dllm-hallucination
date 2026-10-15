from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from model_backends import LoadedBackend, _decode_continuation, format_prompt


class FakeTokens:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return list(self.values)


class FakeTokenizer:
    eos_token_id = [2, 99]
    chat_template = "fake"

    def decode(self, token_ids, skip_special_tokens=True):
        words = {10: "Paris", 11: "suffix", 2: "<eos>", 99: "<end>"}
        values = [words[token_id] for token_id in token_ids]
        if skip_special_tokens:
            values = [value for value in values if not value.startswith("<")]
        return " ".join(values)

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.last_messages = messages
        return "<chat>"


class ModelBackendTest(unittest.TestCase):
    def test_first_eos_truncates_scored_but_preserves_raw(self) -> None:
        backend = LoadedBackend(
            name="dream",
            model_id="test/dream",
            model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=2)),
            tokenizer=FakeTokenizer(),
            prompt_mode="plain",
        )
        result = _decode_continuation(backend, FakeTokens([10, 2, 11]))
        self.assertEqual(result.raw_output, "Paris suffix")
        self.assertEqual(result.scored_output, "Paris")
        self.assertEqual(result.first_eos_index, 1)
        self.assertEqual(result.continuation_token_ids, (10, 2, 11))

    def test_qwen_chat_template_is_explicit(self) -> None:
        tokenizer = FakeTokenizer()
        backend = LoadedBackend(
            name="qwen",
            model_id="test/qwen",
            model=object(),
            tokenizer=tokenizer,
            prompt_mode="chat",
        )
        prompt, add_special_tokens = format_prompt(backend, "A question?")
        self.assertEqual(prompt, "<chat>")
        self.assertFalse(add_special_tokens)
        self.assertIn("A question?", tokenizer.last_messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
