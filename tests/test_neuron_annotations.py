"""Official MaleCNS neuron role labels: conservative mapping and security tests."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from connectome import MaleCNSConnectome
from neuron_annotations import annotation_record, categorize_superclass


class AnnotationKindTests(unittest.TestCase):
    def test_known_official_superclasses_only(self):
        examples = {
            "vnc_motor": "motor",
            "cb_motor": "motor",
            "ol_sensory": "sensory",
            "vnc_sensory": "sensory",
            "sensory_ascending": "sensory",
            "sensory_descending": "sensory",
            "cb_intrinsic": "interneuron",
            "vnc_intrinsic": "interneuron",
            "ol_intrinsic": "interneuron",
            "ascending_neuron": "projection",
            "descending_neuron": "projection",
            "visual_projection": "projection",
            "visual_centrifugal": "projection",
            "vnc_efferent": "other",  # never infer motor from efferent
            "cb_endocrine": "other",
            "vnc_tbc": "other",
            "unknown_new_superclass": "other",
            "": "unclassified",
            None: "unclassified",
        }
        for superclass, expected in examples.items():
            with self.subTest(superclass=superclass):
                self.assertEqual(categorize_superclass(superclass), expected)

    def test_unannotated_neurons_never_get_inferred_roles(self):
        record = annotation_record({
            "body_id": 12781, "superclass": None,
            "type": "DNge104", "instance": "DNge104_R",
            "consensus_nt": float("nan"),
        })
        self.assertEqual(record["kind"], "unclassified")
        self.assertIsNone(record["superclass"])
        self.assertIsNone(record["consensus_nt"])
        self.assertEqual(record["body_id"], "12781")

    def test_missing_neurotransmitter_not_fabricated(self):
        record = annotation_record({
            "body_id": 1, "superclass": "vnc_motor",
            "consensus_nt": None, "type": None, "instance": None,
        })
        self.assertEqual(record["kind"], "motor")
        self.assertIsNone(record["consensus_nt"])


class FakeClient:
    def __init__(self):
        self.calls = []

    def fetch_custom(self, query):
        self.calls.append(query)
        return pd.DataFrame([
            {
                "body_id": 10009, "superclass": "vnc_motor",
                "type": "TEST_MOTOR", "instance": "TEST_MOTOR_R",
                "consensus_nt": "glutamate",
            },
            {
                "body_id": 12781, "superclass": "descending_neuron",
                "type": "DNge104", "instance": "DNge104_R",
                "consensus_nt": "acetylcholine",
            },
        ])


class BatchQueryTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeClient()
        self.mcns = MaleCNSConnectome(client=self.fake)

    def test_one_query_for_selected_neurons(self):
        data = self.mcns.neuron_annotations([12781, 10009, 12781])
        self.assertEqual(len(self.fake.calls), 1)
        self.assertIn("n.bodyId IN [10009, 12781]", self.fake.calls[0])
        self.assertEqual(data["10009"]["kind"], "motor")
        self.assertEqual(data["12781"]["kind"], "projection")
        self.assertEqual(data["12781"]["superclass"], "descending_neuron")
        self.assertEqual(data["12781"]["consensus_nt"], "acetylcholine")

    def test_unsafe_ids_rejected_without_query(self):
        for body_ids in ([0], [-1], [12781, "1) RETURN n"], [True], list(range(1, 514))):
            with self.subTest(body_ids=str(body_ids)[:70]):
                with self.assertRaises(ValueError):
                    self.mcns.neuron_annotations(body_ids)
        self.assertEqual(self.fake.calls, [])

    def test_empty_ids_dont_query(self):
        self.assertEqual(self.mcns.neuron_annotations([]), {})
        self.assertEqual(self.fake.calls, [])


class EndpointTests(unittest.TestCase):
    def test_annotations_require_auth_before_loading_router(self):
        import app as api

        with patch.object(api, "_require_user",
                          side_effect=api.HTTPException(status_code=401, detail="Login")):
            with patch.object(api, "get_router") as router:
                with self.assertRaises(api.HTTPException) as caught:
                    api.router_annotations(object())
        self.assertEqual(caught.exception.status_code, 401)
        router.assert_not_called()

    def test_missing_neuprint_shows_service_error_not_fake_labels(self):
        import app as api

        router = type("Router", (), {
            "model_path": "test.pt",
            "graph": lambda self: {"nodes": [{"root_id": "12781"}]},
        })()
        with patch.object(api, "_require_user", return_value={"id": "test"}), \
             patch.object(api, "get_router", return_value=router), \
             patch.object(api, "get_connectome", side_effect=RuntimeError("no token")):
            with self.assertRaises(api.HTTPException) as caught:
                api.router_annotations(object())
        self.assertEqual(caught.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
