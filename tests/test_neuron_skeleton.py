"""Offline tests for real MaleCNS neuron geometry and endpoint safety."""
from __future__ import annotations

import math
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from connectome import MaleCNSConnectome


class SkeletonGeometryTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.mcns = MaleCNSConnectome(client=self.client)
        self.client.fetch_skeleton.return_value = pd.DataFrame([
            {"rowId": 1, "link": -1, "x": 10, "y": 20, "z": 30},
            {"rowId": 2, "link": 1, "x": 11, "y": 20, "z": 30},
            {"rowId": 3, "link": 2, "x": 12, "y": 21, "z": 31},
            {"rowId": 4, "link": -1, "x": 30, "y": 40, "z": 50},
            {"rowId": 5, "link": 4, "x": 31, "y": 41, "z": 51},
        ])

    def test_only_actual_parent_links_are_drawn(self):
        shape = self.mcns.skeleton_geometry(12781)
        self.client.fetch_skeleton.assert_called_once_with(
            12781, format="pandas", heal=False
        )
        self.assertEqual(shape["body_id"], "12781")
        self.assertEqual(shape["total_segments"], 3)
        self.assertEqual(shape["shown_segments"], 3)
        self.assertEqual(shape["segments"][0], [11.0, 20.0, 30.0, 10.0, 20.0, 30.0])
        self.assertEqual(shape["segments"][2], [31.0, 41.0, 51.0, 30.0, 40.0, 50.0])
        self.assertEqual(shape["bounds"]["min"], [10.0, 20.0, 30.0])
        self.assertEqual(shape["bounds"]["max"], [31.0, 41.0, 51.0])

    def test_sampling_is_bounded_and_does_not_invent_segments(self):
        shape = self.mcns.skeleton_geometry(12781, max_segments=2)
        self.assertEqual(shape["total_segments"], 3)
        self.assertEqual(shape["shown_segments"], 2)
        self.assertTrue(shape["sampled"])

    def test_invalid_id_fails_before_any_network_query(self):
        for bad in (-1, 0, True, "12781", 2**63):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.mcns.skeleton_geometry(bad)
        self.client.fetch_skeleton.assert_not_called()

    def test_nan_coordinates_are_not_rendered(self):
        df = self.client.fetch_skeleton.return_value.copy()
        df.loc[1, "x"] = math.nan
        self.client.fetch_skeleton.return_value = df
        shape = self.mcns.skeleton_geometry(12781)
        self.assertEqual(shape["total_segments"], 1)

    def test_missing_schema_refuses_fake_morphology(self):
        self.client.fetch_skeleton.return_value = pd.DataFrame([{"foo": "bar"}])
        with self.assertRaises(ValueError):
            self.mcns.skeleton_geometry(12781)


class SkeletonEndpointTests(unittest.TestCase):
    def test_auth_happens_before_neuprint_access(self):
        import app as api
        with patch.object(api, "_require_user", side_effect=api.HTTPException(401)):
            with patch.object(api, "get_connectome") as get_data:
                with self.assertRaises(api.HTTPException) as err:
                    api.neuron_skeleton(object(), 12781)
        self.assertEqual(err.exception.status_code, 401)
        get_data.assert_not_called()

    def test_service_failure_does_not_return_fake_skeleton(self):
        import app as api
        with patch.object(api, "_require_user", return_value={"id": "test"}):
            with patch.object(api, "get_connectome", side_effect=RuntimeError("no token")):
                with self.assertRaises(api.HTTPException) as err:
                    api.neuron_skeleton(object(), 12781)
        self.assertEqual(err.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
