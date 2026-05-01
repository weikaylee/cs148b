from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.token_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}

    @staticmethod
    def _decode_token_field(token_field):
        if isinstance(token_field, dict):
            if "hex" in token_field:
                return bytes.fromhex(token_field["hex"])
            if "utf8" in token_field:
                return token_field["utf8"].encode("utf-8")
            raise ValueError("Unsupported token object format in vocab JSON.")

        if isinstance(token_field, str):
            return token_field.encode("utf-8")

        raise ValueError("Unsupported token format in vocab JSON.")

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        Class method that constructs and returns a Tokenizer from a serialized vocabulary
        and list of merges (in the same format that your BPE training code output) and
        (optionally) a list of special tokens. This method should accept the following additional
        parameters:
        """
        vocab_path = Path(vocab_filepath)
        merges_path = Path(merges_filepath)

        with vocab_path.open("r", encoding="utf-8") as f:
            raw_vocab = json.load(f)

        vocab = {}

        # Supported format A (train_tinystories_bpe.py):
        # {
        #   "0": {"utf8": "...", "hex": "...", "length_bytes": ...},
        #   ...
        # }
        if all(str(k).isdigit() for k in raw_vocab.keys()):
            for token_id_str, token_field in raw_vocab.items():
                vocab[int(token_id_str)] = cls._decode_token_field(token_field)
        else:
            # Supported format B (GPT-like):
            # {
            #   "token_text": token_id,
            #   ...
            # }
            for token_text, token_id in raw_vocab.items():
                vocab[int(token_id)] = cls._decode_token_field(token_text)

        merges = []
        with merges_path.open("r", encoding="utf-8") as f:
            for line in f:
                cleaned = line.strip()
                if not cleaned:
                    continue
                left_str, right_str = cleaned.split()

                # Primary format: hex pairs.
                try:
                    left = bytes.fromhex(left_str)
                    right = bytes.fromhex(right_str)
                except ValueError:
                    # Fallback format: raw UTF-8 token strings.
                    left = left_str.encode("utf-8")
                    right = right_str.encode("utf-8")

                merges.append((left, right))

        if special_tokens:
            vocab_values = set(vocab.values())
            for special_token in special_tokens:
                b = special_token.encode("utf-8")
                if b not in vocab_values:
                    vocab[len(vocab)] = b
                    vocab_values.add(b)

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode input text into token IDs using regex pre-tokenization + ordered BPE merges."""
        token_ids: list[int] = []

        if self.special_tokens:
            # Keep delimiters so special tokens are preserved as single tokens.
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            split_pattern = "(" + "|".join(map(re.escape, sorted_specials)) + ")"
            chunks = re.split(split_pattern, text)
        else:
            chunks = [text]

        for chunk in chunks:
            if not chunk:
                continue

            if chunk in self.special_tokens:
                token_bytes = chunk.encode("utf-8")
                token_id = self.token_to_id.get(token_bytes)
                if token_id is None:
                    raise ValueError(f"Special token not found in vocabulary: {chunk}")
                token_ids.append(token_id)
                continue

            for m in re.finditer(PAT, chunk):
                pretoken = m.group().encode("utf-8")
                symbols = [bytes([b]) for b in pretoken]

                # Apply merges in order of creation, independently within each pre-token.
                for left, right in self.merges:
                    if len(symbols) < 2:
                        break

                    merged_symbols: list[bytes] = []
                    i = 0
                    while i < len(symbols):
                        if i + 1 < len(symbols) and symbols[i] == left and symbols[i + 1] == right:
                            merged_symbols.append(left + right)
                            i += 2
                        else:
                            merged_symbols.append(symbols[i])
                            i += 1
                    symbols = merged_symbols

                for token_bytes in symbols:
                    token_id = self.token_to_id.get(token_bytes)
                    if token_id is None:
                        raise ValueError(f"Token bytes not found in vocabulary: {token_bytes!r}")
                    token_ids.append(token_id)

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings, lazily yield token IDs."""
        for chunk in iterable:
            for token_id in self.encode(chunk):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text.

        Concatenates token bytes from the vocabulary and decodes to Unicode using
        replacement for malformed byte sequences.
        """
        out = bytearray()
        replacement = "\uFFFD".encode("utf-8")

        for token_id in ids:
            token_bytes = self.vocab.get(token_id)
            if token_bytes is None:
                out.extend(replacement)
            else:
                out.extend(token_bytes)

        return bytes(out).decode("utf-8", errors="replace")