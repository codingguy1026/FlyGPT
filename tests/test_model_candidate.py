from __future__ import annotations

import importlib
import os
import unittest
from pathlib import Path
from unittest.mock import patch


class ModelCandidateTests(unittest.TestCase):
    def test_malecns_v05_is_first_default_candidate(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FLYGPT_MODEL_PATH", None)
            import app
            importlib.reload(app)

        self.assertEqual(
            app._model_candidates[0],
            "artifacts/fly_router_malecns_v0_5.pt",
        )
        self.assertEqual(
            Path(app.MODEL_PATH),
            Path("artifacts/fly_router_malecns_v0_5.pt"),
        )


if __name__ == "__main__":
    unittest.main()
