import unittest

from vocab import build_vocab


class VocabularyTests(unittest.TestCase):
    def test_build_vocab_reserves_period_for_sequence_boundaries(self):
        stoi, itos = build_vocab(["ada", "alan"])

        self.assertEqual(stoi["."], 0)
        self.assertEqual(itos[0], ".")
        self.assertEqual(set(stoi), {".", "a", "d", "l", "n"})

    def test_build_vocab_is_deterministic(self):
        first = build_vocab(["zara", "ada"])
        second = build_vocab(["zara", "ada"])

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
