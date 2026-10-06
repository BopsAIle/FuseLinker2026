"""CPU regression checks: python -m unittest discover -s task_degree_edge -v."""
import unittest
import tempfile
from pathlib import Path
import task_setup  # noqa: F401
import dgl
import numpy as np
import torch
from model_base4 import (FusionGate, LinkPredict, PPRBranch, EdgeTypeClassifier,
                        DegreeClassifier, _PPRDiffusion)
from ssl_eval import encode_full_train_graph


class SSLRegressionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        torch.set_num_threads(1)

    def test_fusion_starts_at_requested_prior(self):
        a, b = torch.randn(3, 4), torch.randn(3, 4)
        gate = FusionGate(4)
        torch.testing.assert_close(gate(a, b, w=0.75), .75 * a + .25 * b)

    def test_feature_ppr_matches_dense_reference_and_preserves_gradients(self):
        g = dgl.graph(([0, 1, 1, 2], [1, 0, 2, 1]), num_nodes=4)
        branch = PPRBranch(4, num_layers=1, c=0.15, num_iter=12)
        with torch.no_grad():
            branch.layers[0].linear.weight.copy_(torch.eye(4))
            branch.layers[0].linear.bias.zero_()
            branch.mass_projection.weight.zero_()
        x = torch.randn(4, 4, requires_grad=True)
        a = torch.eye(4)
        src, dst = g.edges()
        a[dst, src] += 1
        inv = a.sum(1).rsqrt()
        transition = inv[:, None] * a * inv[None, :]
        expected = .85 * torch.linalg.solve(torch.eye(4) - .15 * transition, x)
        actual = branch(g, x)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        actual.square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        branch.eval()
        torch.testing.assert_close(branch(g, x), actual)

    def test_ppr_adjoint_on_directed_graph(self):
        g = dgl.graph(([0, 1], [1, 2]), num_nodes=3)
        x = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda z: _PPRDiffusion.apply(z, g, .85, 8), (x,)))

    def test_structural_mass_is_diffusion_of_ones(self):
        g = dgl.graph(([0, 1, 0, 2], [1, 0, 2, 0]), num_nodes=3)
        branch = PPRBranch(4)
        x = torch.randn(3, 4)
        joint = branch.diffuse(g, x)
        expected = _PPRDiffusion.apply(torch.ones(3, 1), g, branch.c, branch.num_iter)
        torch.testing.assert_close(joint[:, -1:], expected)
        self.assertGreater(expected.std().item(), 0)

    def test_single_node_keeps_batch_dimension(self):
        m = LinkPredict(3, 4, 1, num_bases=2, use_ppr=False)
        g = dgl.graph(([], []), num_nodes=1)
        h = m(g, torch.tensor([[2]]), torch.empty(0, dtype=torch.long), torch.empty(0, 1))
        self.assertEqual(h.shape, (1, 4))
        h.square().sum().backward()

    def test_degree_full_graph_can_train_encoder(self):
        m = LinkPredict(3, 4, 1, num_bases=2, use_ppr=False)
        g = dgl.graph(([0, 1], [1, 0]), num_nodes=3)
        h = encode_full_train_graph(m, g, np.array([0, 1]), np.ones(3, np.float32),
                                    3, torch.device('cpu'))
        self.assertTrue(h.requires_grad, 'Degree loss must reach encoder during training')
        h.square().sum().backward()
        self.assertIsNotNone(m.rgcn.layers[0].domain_projector.net[0].weight.grad)
        m.eval()
        h = encode_full_train_graph(m, g, np.array([0, 1]), np.ones(3, np.float32),
                                    3, torch.device('cpu'))
        self.assertFalse(h.requires_grad)

    def test_joint_ssl_loss_reaches_both_branches(self):
        m = LinkPredict(3, 4, 1, num_bases=2)
        g = dgl.graph(([0, 1], [1, 0]), num_nodes=3)
        h = m(g, torch.arange(3), torch.tensor([0, 1]), torch.ones(2, 1))
        edge, degree = EdgeTypeClassifier(4, 2), DegreeClassifier(4, 2)
        le = torch.nn.functional.cross_entropy(edge(h[[0, 1]], h[[1, 2]]), torch.tensor([0, 1]))
        ld = torch.nn.functional.cross_entropy(degree(h), torch.tensor([0, 1, 0]))
        (le + ld).backward()
        for p in (m.ppr_branch.layers[0].linear.weight,
                  m.ppr_fusion.fc.weight, m.rgcn.layers[0].fusion_gate.fc.weight,
                  edge.interaction[-1].weight):
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.abs().sum().item(), 0)

    def test_isolated_nodes_keep_restart_path(self):
        m = LinkPredict(3, 4, 1, num_bases=2).eval()
        g = dgl.graph(([], []), num_nodes=3)
        ids = torch.arange(3)
        rel, norm = torch.empty(0, dtype=torch.long), torch.empty(0, 1)
        with torch.no_grad():
            x = m.rgcn.encode_input(ids)
            main = m.rgcn(g, ids, rel, norm)
            expected = m.ppr_fusion(main, m.ppr_branch(g, x))
            actual = m(g, ids, rel, norm)
        torch.testing.assert_close(actual, expected)

    def test_masked_encoder_does_not_consume_full_graph_ppr(self):
        m = LinkPredict(3, 4, 1, num_bases=2).eval()
        g = dgl.graph(([0, 1], [1, 0]), num_nodes=3)
        # A supplied global auxiliary graph can contain the held-out pair 0--2.
        leaked = dgl.graph(([0, 2], [2, 0]), num_nodes=3)
        ids, rel, norm = torch.arange(3).view(-1, 1), torch.tensor([0, 1]), torch.ones(2, 1)
        with torch.no_grad():
            expected = m(g, ids, rel, norm)
            actual = m(g, ids, rel, norm, ppr_graph=leaked,
                       ppr_edge_weight=torch.ones(2), all_node_ids=ids)
        torch.testing.assert_close(actual, expected)

    def test_global_context_removes_all_target_directions(self):
        m = LinkPredict(3, 4, 1, num_bases=2)
        visible = dgl.graph(([0, 1], [1, 0]), num_nodes=3)
        # Include duplicate target edges to check removal across all relations.
        context = dgl.graph(([0, 1, 1, 2, 1], [1, 0, 2, 1, 2]), num_nodes=3)
        ids, rel, norm = torch.arange(3), torch.tensor([0, 1]), torch.ones(2, 1)
        with self.assertRaisesRegex(ValueError, 'masked_pairs'):
            m(visible, ids, rel, norm, context_graph=context)
        actual = m(visible, ids, rel, norm, context_graph=context,
                   masked_pairs=torch.tensor([[1, 2]]))
        expected = m(visible, ids, rel, norm, context_graph=visible,
                     masked_pairs=torch.empty(0, 2, dtype=torch.long))
        torch.testing.assert_close(actual, expected)
        self.assertEqual(context.num_edges(), 5)
        actual.square().sum().backward()
        self.assertTrue(torch.isfinite(m.ppr_branch.mass_projection.weight.grad).all())

    def test_degree_prior_uses_training_counts_and_survives_roundtrip(self):
        head = DegreeClassifier(4, 3, dropout=0)
        for p in head.parameters():
            torch.nn.init.zeros_(p)
        head.configure_sampling_prior(torch.ones(4), torch.tensor([0, 0, 0, 1]),
                                      torch.ones(3), 2, 4, full_graph=True)
        x = torch.zeros(1, 4)
        torch.testing.assert_close(head(x).softmax(-1), torch.tensor([[.75, .25, 0.]]))
        head.eval()
        torch.testing.assert_close(head(x).softmax(-1), torch.tensor([[.5, .5, 0.]]))
        restored = DegreeClassifier(4, 3, dropout=0).eval()
        restored.load_state_dict(head.state_dict())
        torch.testing.assert_close(restored(x), head(x))

    def test_benchmark_preserves_original_entity_order(self):
        from benchmark import load_data
        from data_loader import Data
        import pandas as pd
        frames = [pd.DataFrame([['b', 'r', 'a'], ['c', 's', 'b']]),
                  pd.DataFrame([['a', 's', 'd']]), pd.DataFrame([['e', 'r', 'c']])]
        reference = Data(pd.concat(frames), *frames)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            for split, frame in zip(('train', 'valid', 'test'), frames):
                frame.to_csv(folder / (split + '.tsv'), sep='\t', header=False, index=False)
            n, r, arrays = load_data(folder)
        self.assertEqual((n, r), (reference.num_nodes, reference.num_rels))
        for actual, expected in zip(arrays, (reference.train_data, reference.valid_data,
                                            reference.test_data)):
            np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
