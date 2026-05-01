from collections import defaultdict

from eecs148b_hw1.bpe_trainer import BPETrainer, DoublyLinkedList, Node

def ll_values(ll: DoublyLinkedList) -> list[int]:
	vals: list[int] = []
	cur = ll.head
	while cur is not None:
		vals.append(cur.value)
		cur = cur.next
	return vals


def test_node_init():
	node = Node(42)
	assert node.value == 42
	assert node.prev is None
	assert node.next is None


def test_doubly_linked_list_init_with_values():
	ll = DoublyLinkedList([1, 2, 3])

	assert ll.size == 3
	assert ll.head is not None and ll.head.value == 1
	assert ll.tail is not None and ll.tail.value == 3
	assert ll_values(ll) == [1, 2, 3]
	assert ll.head.next is not None and ll.head.next.prev is ll.head


def test_doubly_linked_list_append_updates_links():
	ll = DoublyLinkedList()

	n1 = ll.append(7)
	n2 = ll.append(8)

	assert ll.size == 2
	assert ll.head is n1
	assert ll.tail is n2
	assert n1.next is n2
	assert n2.prev is n1


def test_bpe_trainer_init_uses_defaultdicts():
	trainer = BPETrainer(corpus=["hello"], vocab_size=300, special_tokens=[])

	assert isinstance(trainer.pretoken_to_freq, defaultdict)
	assert isinstance(trainer.pair_to_freq, defaultdict)
	assert isinstance(trainer.pair_to_node, defaultdict)
	assert trainer.vocab_size == 300
	assert trainer.corpus == ["hello"]
	assert trainer.special_tokens == []


def test_init_vocab_adds_bytes_and_special_tokens():
	trainer = BPETrainer(corpus=["hello"], vocab_size=300, special_tokens=["<|endoftext|>"])
	trainer.init_vocab()

	assert len(trainer.vocab) == 257
	assert trainer.vocab[0] == b"\x00"
	assert trainer.vocab[255] == b"\xff"
	assert b"<|endoftext|>" in trainer.vocab.values()


def test_init_pretokens_builds_freq_and_linked_lists():
	trainer = BPETrainer(corpus=["hello hello"], vocab_size=300, special_tokens=[])
	trainer.init_pretokens(trainer.corpus)

	assert len(trainer.pretoken_to_freq) > 0
	assert len(trainer.pretoken_to_ll) > 0

	for pretoken, freq in trainer.pretoken_to_freq.items():
		assert freq > 0
		ll = trainer.pretoken_to_ll[pretoken]
		assert ll_values(ll) == list(pretoken)


def test_init_pair_to_node_populates_pair_maps():
	trainer = BPETrainer(corpus=["banana"], vocab_size=300, special_tokens=[])
	trainer.init_pretokens(trainer.corpus)
	trainer.init_pair_to_node_and_pair_to_freq()

	assert len(trainer.pair_to_node) > 0
	assert len(trainer.pair_to_freq) > 0

	for pair, nodes in trainer.pair_to_node.items():
		assert len(nodes) > 0
		assert trainer.pair_to_freq[pair] == len(nodes)
		for node in nodes:
			assert node.next is not None
			assert (node.value, node.next.value) == pair


def test_merge_tokens_updates_neighbors_and_merges_list():
	trainer = BPETrainer(corpus=[], vocab_size=300, special_tokens=[])
	ll = DoublyLinkedList([4, 5, 6])

	first = ll.head
	assert first is not None
	second = first.next
	assert second is not None

	trainer.pair_to_node[(4, 5)].append(first)
	trainer.pair_to_node[(5, 6)].append(second)
	trainer.pair_to_freq[(4, 5)] = 1
	trainer.pair_to_freq[(5, 6)] = 1

	trainer.merge_tokens((4, 5))

	assert trainer.merges == [(4, 5)]
	assert second.next is not None
	assert second.next.prev is not None
	assert second.next.prev.value == 9
	assert (9, 6) in trainer.pair_to_node
	assert trainer.pair_to_freq[(9, 6)] == 1


def test_train_bpe_runs_and_grows_vocab_once():
	trainer = BPETrainer(corpus=["aa"], vocab_size=257, special_tokens=[])
	trainer.train_bpe()

	assert len(trainer.vocab) == 257
	assert len(trainer.merges) == 1

def test_init_pair_to_node_and_pair_to_freq_real():
	trainer = BPETrainer(["this", "is", "a", "test!", "test", "test"], vocab_size=260, special_tokens=[])
	trainer.train_bpe()
	# trainer.init_pretokens(trainer.corpus)
	# # pretokens: dict_keys([b'this', b'\n', b'is', b'a', b'test', b'!'])
	# trainer.init_pair_to_node_and_pair_to_freq()
	# print(f"this is the pair to freq: {trainer.pretoken_to_ll}")




#     for pretoken, ll in self.pretoken_to_ll.items():
#         node = ll.head
#         freq = self.pretoken_to_freq[pretoken]

#         while node is not None and node.next is not None:
#             pair = (node.value, node.next.value)

#             if pair not in self.pair_to_node:
#                 self.pair_to_node[pair] = []
#             self.pair_to_node[pair].append(node)
            
#             self.pair_to_freq[pair] = self.pair_to_freq.get(pair, 0) + freq
#             node = node.next

	 
# def init_pair_to_node_and_pair_to_freq(self):        
#     for pretoken, ll in self.pretoken_to_ll.items():
#         node = ll.head
#         freq = self.pretoken_to_freq[pretoken]

#         while node is not None and node.next is not None:
#             pair = (node.value, node.next.value)

#             if pair not in self.pair_to_node:
#                 self.pair_to_node[pair] = []
#             self.pair_to_node[pair].append(node)
            
#             self.pair_to_freq[pair] = self.pair_to_freq.get(pair, 0) + freq
#             node = node.next
