from __future__ import annotations
from collections import defaultdict
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BPETrainer():
    """
    Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    def __init__(self, corpus, vocab_size, special_tokens):
        self.merges = []
        self.vocab = {}

        self.vocab_size = vocab_size
        self.corpus = corpus
        self.special_tokens = special_tokens

        # State used during BPE training.
        self.word_freq: dict[tuple[bytes, ...], int] = {}
        self.pair_to_freq: dict[tuple[bytes, bytes], int] = {}
        self.pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}
    
    def init_vocab(self):
        """Add all 256 bytes, and special tokens, to the vocab."""
        self.vocab = {i: bytes([i]) for i in range(256)}

        for tok in self.special_tokens:
            self.vocab[len(self.vocab)] = tok.encode("utf-8")

    def _iter_pairs(self, word: tuple[bytes, ...]):
        for i in range(len(word) - 1):
            yield (word[i], word[i + 1])

    def _merge_word(self, word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
        """Merge every non-overlapping occurrence of `pair` in `word` from left to right."""
        merged = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                merged.append(word[i] + word[i + 1])
                i += 2
            else:
                merged.append(word[i])
                i += 1
        return tuple(merged)

    def init_pretokens(self, corpus):
        """Split corpus into pre-tokens and aggregate frequencies by UTF-8 byte sequence."""
        pretoken_freq: dict[bytes, int] = defaultdict(int)
        combined = "\n".join(corpus)

        # Split on special tokens to avoid merges across special-token boundaries.
        if self.special_tokens:
            pattern = "|".join(map(re.escape, self.special_tokens))
            chunks = re.split(pattern, combined)
        else:
            chunks = [combined]

        for chunk in chunks:
            if not chunk:
                continue

            for m in re.finditer(PAT, chunk):
                pretoken_freq[m.group().encode("utf-8")] += 1

        self.word_freq = {}
        for pretoken, freq in pretoken_freq.items():
            word = tuple(bytes([b]) for b in pretoken)
            self.word_freq[word] = self.word_freq.get(word, 0) + freq

    def init_pair_stats(self):
        self.pair_to_freq = defaultdict(int)
        self.pair_to_words = defaultdict(set)

        for word, freq in self.word_freq.items():
            for pair in self._iter_pairs(word):
                self.pair_to_freq[pair] += freq
                self.pair_to_words[pair].add(word)

    def merge_tokens(self, pair: tuple[bytes, bytes]):
        affected_words = list(self.pair_to_words.get(pair, set()))
        updates: list[tuple[tuple[bytes, ...], int, tuple[bytes, ...]]] = []

        # Snapshot current frequencies so newly created words are not reprocessed in this merge step.
        for old_word in affected_words:
            freq = self.word_freq.get(old_word, 0)
            if freq <= 0:
                continue
            updates.append((old_word, freq, self._merge_word(old_word, pair)))

        # Phase 1: remove contributions of old words.
        for old_word, freq, _ in updates:
            remaining = self.word_freq.get(old_word, 0) - freq
            if remaining > 0:
                self.word_freq[old_word] = remaining
            else:
                self.word_freq.pop(old_word, None)

            for old_pair in self._iter_pairs(old_word):
                new_count = self.pair_to_freq.get(old_pair, 0) - freq
                if new_count > 0:
                    self.pair_to_freq[old_pair] = new_count
                else:
                    self.pair_to_freq.pop(old_pair, None)

                words_for_pair = self.pair_to_words.get(old_pair)
                if words_for_pair is not None:
                    words_for_pair.discard(old_word)
                    if not words_for_pair:
                        self.pair_to_words.pop(old_pair, None)

        # Phase 2: add contributions of merged words.
        for _, freq, new_word in updates:
            self.word_freq[new_word] = self.word_freq.get(new_word, 0) + freq

            for new_pair in self._iter_pairs(new_word):
                self.pair_to_freq[new_pair] = self.pair_to_freq.get(new_pair, 0) + freq
                self.pair_to_words.setdefault(new_pair, set()).add(new_word)

        self.pair_to_freq.pop(pair, None)
        self.pair_to_words.pop(pair, None)
        self.merges.append(pair)

    def train_bpe(self):
        self.init_vocab()
        self.init_pretokens(self.corpus)
        self.init_pair_stats()

        while len(self.vocab) < self.vocab_size and self.pair_to_freq:
            # Deterministic tie break: lexicographically greater pair wins.
            most_frequent = max(self.pair_to_freq.items(), key=lambda item: (item[1], item[0]))[0]
            merged_token = most_frequent[0] + most_frequent[1]
            self.merge_tokens(most_frequent)
            self.vocab[len(self.vocab)] = merged_token