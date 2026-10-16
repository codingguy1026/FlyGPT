"""Scientific invariants: disjoint groups and nested training fractions."""
import unittest

from training.natural_learning_curves import (
    group_near_duplicates, grouped_three_way_split,
    ordered_nested_training, fraction_training,
)
from training.train_router import Example


class NaturalLearningCurvesTests(unittest.TestCase):
    def examples(self):
        values = []
        for route in ("code", "math", "general"):
            for i in range(25):
                values.append(Example(f"distinct {route} prompt number {i:03d} with topic {i*7}", route))
        return values

    def test_all_sets_disjoint_and_complete(self):
        rows = self.examples()
        training, val, test, audit = grouped_three_way_split(rows, 42)
        originals = {x.text for x in rows}
        a, b, c = [set(x.text for x in group) for group in (training, val, test)]
        self.assertEqual(originals, a | b | c)
        self.assertFalse(a & b or a & c or b & c)
        self.assertEqual(audit["train"] + audit["validation"] + audit["test"], len(rows))

    def test_nested_train_sets_are_reproducible(self):
        train, _, _, _ = grouped_three_way_split(self.examples(), 42)
        groups = ordered_nested_training(train, 42)
        low = fraction_training(groups, .25)
        mid = fraction_training(groups, .5)
        high = fraction_training(groups, 1.)
        self.assertTrue(set(x.text for x in low) <= set(x.text for x in mid))
        self.assertTrue(set(x.text for x in mid) <= set(x.text for x in high))
        self.assertEqual(len(high), len(train))
        self.assertEqual(
            [x.text for x in fraction_training(ordered_nested_training(train, 42), .25)],
            [x.text for x in low],
        )

    def test_paraphrase_cluster_kept_together(self):
        rows = self.examples()
        rows += [
            Example("Please write this exact sample code", "code"),
            Example("Please write this exact sample code!", "code"),
        ]
        clusters = group_near_duplicates(rows)
        pair = [cluster for cluster in clusters if 75 in cluster or 76 in cluster]
        self.assertEqual(len(pair), 1)
        self.assertEqual(set(pair[0]), {75, 76})
        train, val, test, _ = grouped_three_way_split(rows, 33)
        locations = {
            x.text: split_id
            for split_id, group in enumerate((train, val, test))
            for x in group
        }
        self.assertEqual(
            locations["Please write this exact sample code"],
            locations["Please write this exact sample code!"],
        )

    def test_invalid_fraction_rejected(self):
        train, _, _, _ = grouped_three_way_split(self.examples(), 1)
        by_route = ordered_nested_training(train, 1)
        with self.assertRaises(ValueError):
            fraction_training(by_route, 0)
        with self.assertRaises(ValueError):
            fraction_training(by_route, 1.25)


if __name__ == "__main__":
    unittest.main()
