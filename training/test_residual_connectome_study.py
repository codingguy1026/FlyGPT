"""Invariants for the experimental residual MaleCNS router."""
import unittest

import torch

from training.residual_connectome_study import ResidualFlyGraphRouter


class ResidualConnectomeTests(unittest.TestCase):
    @staticmethod
    def router():
        n = 16
        src = torch.tensor([i for i in range(n)], dtype=torch.long)
        dst = torch.tensor([(i+1) % n for i in range(n)], dtype=torch.long)
        base = torch.ones(n, dtype=torch.float32)
        return ResidualFlyGraphRouter(128, n, 6, src, dst, base)

    def test_gate_starts_at_prespecified_value(self):
        self.assertAlmostEqual(self.router().graph_share().item(), 0.2, places=5)

    def test_zero_gate_equals_text_only_path(self):
        net = self.router()
        x = torch.randn(3, 128)
        with torch.no_grad():
            net.graph_logit.fill_(-40.0)
            obtained = net(x)
            expected = net.output(net.norm(torch.tanh(net.encoder(x))))
        torch.testing.assert_close(obtained, expected)

    def test_graph_edges_can_affect_the_predictions(self):
        net = self.router()
        x = torch.randn(2, 128)
        with torch.no_grad():
            net.graph_logit.fill_(40.0)
            graph_scores = net(x)
            net.graph_logit.fill_(-40.0)
            text_scores = net(x)
        self.assertFalse(torch.allclose(graph_scores, text_scores))

    def test_graph_gate_can_learn(self):
        net = self.router()
        x = torch.randn(5, 128)
        logits = net(x)
        logits.square().mean().backward()
        self.assertIsNotNone(net.graph_logit.grad)
        self.assertTrue(torch.isfinite(net.graph_logit.grad).all().item())

    def test_does_not_change_base_router_architecture(self):
        net = self.router()
        self.assertEqual(net.steps, 3)
        self.assertEqual(net.src.numel(), 16)
        self.assertEqual(net.dst.numel(), 16)


if __name__ == "__main__":
    unittest.main()
