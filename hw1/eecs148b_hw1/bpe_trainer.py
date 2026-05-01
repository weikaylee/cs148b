from __future__ import annotations
from collections import defaultdict
import regex as re


class Node:
    def __init__(self, value: bytes):
        self.value = value
        self.prev: Node | None = None
        self.next: Node | None = None
        self.owner: DoublyLinkedList | None = None
        self.alive = True


class DoublyLinkedList:
    def __init__(self, values: list[bytes] | None = None):
        self.head: Node | None = None
        self.tail: Node | None = None
        self.size = 0

        if values is not None:
            for value in values:
                self.append(value)

    def append(self, value: bytes) -> Node:
        node = Node(value)
        node.owner = self

        if self.head is None:
            self.head = node
            self.tail = node
        else:
            assert self.tail is not None
            self.tail.next = node
            node.prev = self.tail
            self.tail = node

        self.size += 1
        return node


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
        self.pretoken_to_freq = defaultdict(int)
        self.pretoken_to_ll = {}
        self.ll_to_freq = {}

        self.pair_to_freq = defaultdict(int)
        self.pair_to_node = defaultdict(set)

        self.merges = []
        self.vocab = {}

        self.vocab_size = vocab_size
        self.corpus = corpus
        self.special_tokens = special_tokens
    
    def init_vocab(self):
        """Add all 256 bytes, and special tokens, to the vocab."""
        self.vocab = {i: bytes([i]) for i in range(256)}

        for tok in self.special_tokens:
            self.vocab[len(self.vocab)] = tok.encode("utf-8")

    def init_pretokens(self, corpus):
        """Split the corpus into pretokens and convert each unique pretoken into a linked list."""
        combined = "\n".join(corpus)

        # split on special tokens
        if self.special_tokens:
            pattern = "|".join(map(re.escape, self.special_tokens))
            chunks = re.split(pattern, combined)
        else:
            chunks = [combined]

        for chunk in chunks:
            if not chunk:
                continue

            for m in re.finditer(PAT, chunk):
                text = m.group()
                b = bytes(text, "utf-8")
                self.pretoken_to_freq[b] += 1

        for pretoken, freq in self.pretoken_to_freq.items():
            byte_tokens = [bytes([x]) for x in pretoken]
            ll = DoublyLinkedList(byte_tokens)
            self.pretoken_to_ll[pretoken] = ll
            self.ll_to_freq[ll] = freq

    def _add_pair_occurrence(self, pair: tuple[bytes, bytes], first_node: Node, weight: int):
        self.pair_to_node[pair].add(first_node)
        self.pair_to_freq[pair] = self.pair_to_freq.get(pair, 0) + weight

    def _remove_pair_occurrence(self, pair: tuple[bytes, bytes], first_node: Node, weight: int):
        nodes = self.pair_to_node.get(pair)
        if nodes is not None:
            nodes.discard(first_node)
            if not nodes:
                self.pair_to_node.pop(pair, None)

        if pair in self.pair_to_freq:
            new_count = self.pair_to_freq[pair] - weight
            if new_count > 0:
                self.pair_to_freq[pair] = new_count
            else:
                self.pair_to_freq.pop(pair, None)
            
    def init_pair_to_node_and_pair_to_freq(self):        
        for pretoken, ll in self.pretoken_to_ll.items():
            node = ll.head
            freq = self.pretoken_to_freq[pretoken]

            while node is not None and node.next is not None:
                pair = (node.value, node.next.value)
                self._add_pair_occurrence(pair, node, freq)
                node = node.next
    
    def merge_tokens(self, pair: tuple[bytes, bytes]):
        candidates = list(self.pair_to_node.get(pair, set()))

        for node in candidates:
            if not node.alive:
                continue

            second = node.next
            if second is None or not second.alive:
                continue
            if second.prev is not node:
                continue
            if (node.value, second.value) != pair:
                continue

            ll = node.owner
            assert ll is not None
            weight = self.ll_to_freq[ll]

            left = node.prev
            right = second.next

            if left is not None:
                self._remove_pair_occurrence((left.value, node.value), left, weight)
            self._remove_pair_occurrence(pair, node, weight)
            if right is not None:
                self._remove_pair_occurrence((second.value, right.value), second, weight)

            new_node = Node(pair[0] + pair[1])
            new_node.owner = ll
            new_node.prev = left
            new_node.next = right

            if left is not None:
                left.next = new_node
            else:
                ll.head = new_node

            if right is not None:
                right.prev = new_node
            else:
                ll.tail = new_node

            ll.size -= 1

            node.alive = False
            second.alive = False
            node.prev = None
            node.next = None
            second.prev = None
            second.next = None

            if left is not None:
                self._add_pair_occurrence((left.value, new_node.value), left, weight)
            if right is not None:
                self._add_pair_occurrence((new_node.value, right.value), new_node, weight)

        self.pair_to_node.pop(pair, None)
        self.pair_to_freq.pop(pair, None)
        self.merges.append(pair)

    def train_bpe(self):
        # strip all special tokens from the corpus
        # initialize pre-tokens, by mapping each seuquence of bytes to frequency. each pre-token should be a doubly linkedlist!!! 
        # initialie counts of each pair of tokens, by itearting throgh. map pair to linked list node!!! where each node contains the first byte in the pair
        # merge! get the most freuquently occuring pair. updat etheir neighbors, by iterating through the dict mapping pair to nodes. get the left and right neighors, and update thier counts (decrement by..).
        # after their counts are updated, update the pre-otken ll by replacing the most fruequently occuring pair with a merged node, and update the neighbors.

        self.init_vocab()
        self.init_pretokens(self.corpus)
        self.init_pair_to_node_and_pair_to_freq()

        while len(self.vocab) < self.vocab_size and self.pair_to_freq:
            most_frequent = max(
                self.pair_to_freq,
                key=lambda pair: (self.pair_to_freq[pair], pair),
            )
            self.merge_tokens(most_frequent)
            self.vocab[len(self.vocab)] = bytes(most_frequent[0] + most_frequent[1])
            