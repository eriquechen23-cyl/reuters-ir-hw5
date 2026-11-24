import tempfile
import unittest
from collections import Counter
from pathlib import Path

from ir import Search, build_index, read_index


class IndexTests(unittest.TestCase):
    def test_spill_merge_preserves_frequency_and_produces_equal_rankings(self):
        docs = [Counter("oil oil crude".split()), Counter("oil crude".split()),
                Counter("oil wheat".split()), Counter("palm oil".split()), Counter()]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / f"{mode}.tsv" for mode in ("BSBI", "SPIMI")]
            stats = [build_index(docs, mode, path, capacity=2, fan_in=2)
                     for mode, path in zip(("BSBI", "SPIMI"), paths)]
            self.assertEqual(paths[0].read_bytes(), paths[1].read_bytes())
            self.assertTrue(all(s.blocks == 4 and s.merge_passes == 2 for s in stats))
            for path in paths:
                search = Search(read_index(path), len(docs))
                self.assertEqual(search.index["oil"], [(1, 2), (2, 1), (3, 1), (4, 1)])
                self.assertEqual(search.boolean(["crude", "wheat"], "AND"), [])
                self.assertEqual(search.boolean(["crude", "wheat"], "OR"), [1, 2, 3])
                self.assertEqual([hit.doc for hit in search.related(1)], [2, 3, 4])
                top = search.related(1)[0]
                self.assertAlmostEqual(top.score, sum(c.score for c in top.terms))
                self.assertEqual(search.related(5), [])


if __name__ == "__main__":
    unittest.main()
