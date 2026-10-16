"""No PyTorch required: test experimental graph-control invariants."""
import unittest
from collections import Counter

from training.topology_ablation import degrees, filter_heldout, rewire_scaffold


class TopologyAblationTests(unittest.TestCase):
    @staticmethod
    def example_graph():
        # 20 vertices; degree-regular bipartite digraph with 20 unique edges.
        rows = [
            {"src": i, "dst": 10 + i, "weight": 0.2 + i / 100, "syn_count": i + 1}
            for i in range(10)
        ]
        rows += [
            {"src": i, "dst": 10 + (i + 2) % 10, "weight": 0.5 + i / 100, "syn_count": i + 11}
            for i in range(10)
        ]
        return {"dataset": "male-cns:v1.0", "n_nodes": 20,
                "root_ids": list(range(20)), "edges": rows}

    def test_preserves_directed_degree_and_weight_multiset(self):
        original = self.example_graph()
        randomized = rewire_scaffold(original, seed=10042, swaps_per_edge=3)
        self.assertEqual(degrees(original["edges"]), degrees(randomized["edges"]))
        self.assertEqual(sorted(e["weight"] for e in original["edges"]),
                         sorted(e["weight"] for e in randomized["edges"]))
        self.assertEqual(sorted(e["syn_count"] for e in original["edges"]),
                         sorted(e["syn_count"] for e in randomized["edges"]))
        self.assertEqual(len(randomized["edges"]), len(original["edges"]))
        self.assertNotEqual(
            {(e["src"], e["dst"]) for e in original["edges"]},
            {(e["src"], e["dst"]) for e in randomized["edges"]},
        )
        self.assertEqual(original["dataset"], "male-cns:v1.0")
        self.assertEqual(randomized["dataset"], "degree-preserving-rewired-control")

    def test_deterministic_control_given_seed(self):
        self.assertEqual(
            rewire_scaffold(self.example_graph(), 57),
            rewire_scaffold(self.example_graph(), 57),
        )

    def test_exact_overlap_exclusion(self):
        cases = [
            {"input": "Hello  THERE", "route": "general"},
            {"input": "fresh question", "route": "code"},
            {"input": "Fresh Question", "route": "code"},
        ]
        filtered, removed = filter_heldout(cases, {"hello there"})
        self.assertEqual(filtered, [{"input": "fresh question", "route": "code"}])
        self.assertEqual(removed, 2)

    def test_rejects_duplicate_edges(self):
        graph = self.example_graph()
        graph["edges"].append(dict(graph["edges"][0]))
        with self.assertRaises(ValueError):
            rewire_scaffold(graph, 42)


if __name__ == "__main__":
    unittest.main()
