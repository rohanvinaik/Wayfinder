"""Tests for ternary quantization -- values in {-1, 0, +1} and STE backward finite."""

import unittest

import torch

from src.ternary_decoder import TernaryDecoder, TernaryLinear, ternary_quantize


class TestTernaryQuantize(unittest.TestCase):
    def test_output_values_only_ternary(self):
        """Quantized weights must be exactly {-1, 0, +1}."""
        w = torch.randn(32, 64)
        q = ternary_quantize(w)
        unique = sorted(q.unique().tolist())
        for v in unique:
            self.assertIn(v, [-1.0, 0.0, 1.0])

    def test_large_positive_quantizes_to_plus_one(self):
        """Weights well above threshold should become +1."""
        w = torch.tensor([[10.0, 10.0, 10.0, 10.0]])
        q = ternary_quantize(w)
        # All values identical and positive => threshold = 0.7 * mean(abs) = 7.0
        # All > 7.0, so all should be +1
        self.assertEqual(q.tolist(), [[1.0, 1.0, 1.0, 1.0]])

    def test_large_negative_quantizes_to_minus_one(self):
        """Weights well below -threshold should become -1."""
        w = torch.tensor([[-10.0, -10.0, -10.0, -10.0]])
        q = ternary_quantize(w)
        self.assertEqual(q.tolist(), [[-1.0, -1.0, -1.0, -1.0]])

    def test_zero_weights_stay_zero(self):
        """All-zero weights should quantize to zero."""
        w = torch.zeros(4, 4)
        q = ternary_quantize(w)
        self.assertEqual(q.tolist(), torch.zeros(4, 4).tolist())

    def test_mixed_signs_correct(self):
        """Mixed large pos/neg values quantize to correct signs."""
        w = torch.tensor([[5.0, -5.0, 5.0, -5.0]])
        q = ternary_quantize(w)
        # threshold = 0.7 * 5.0 = 3.5; all |values| > 3.5
        self.assertEqual(q.tolist(), [[1.0, -1.0, 1.0, -1.0]])

    def test_shape_preserved(self):
        """Output shape must match input shape."""
        for shape in [(4, 8), (1, 1), (16, 32)]:
            w = torch.randn(*shape)
            q = ternary_quantize(w)
            self.assertEqual(q.shape, w.shape)

    def test_ste_backward_finite(self):
        """STE backward through quantization must produce finite gradients."""
        w = torch.randn(16, 32, requires_grad=True)
        q = ternary_quantize(w)
        loss = q.sum()
        loss.backward()
        self.assertIsNotNone(w.grad)
        self.assertEqual(torch.isnan(w.grad).any().item(), False)
        self.assertEqual(torch.isinf(w.grad).any().item(), False)

    def test_ste_gradient_is_identity(self):
        """STE gradient should be 1.0 for all weights (straight-through)."""
        w = torch.randn(8, 16, requires_grad=True)
        q = ternary_quantize(w)
        loss = q.sum()
        loss.backward()
        # STE: d(quantized)/d(weights) = 1.0 via straight-through
        self.assertEqual(w.grad.tolist(), torch.ones_like(w).tolist())

    def test_quantize_deterministic(self):
        """Same input should produce same output."""
        w = torch.randn(8, 16)
        q1 = ternary_quantize(w)
        q2 = ternary_quantize(w)
        self.assertEqual(q1.tolist(), q2.tolist())


class TestTernaryLinearForward(unittest.TestCase):
    """Tests for TernaryLinear.forward — exact-value assertions on quantized linear."""

    def test_output_shape(self):
        """Output shape must be (batch, out_features)."""
        layer = TernaryLinear(in_features=8, out_features=4, bias=True)
        x = torch.randn(3, 8)
        out = layer(x)
        self.assertEqual(out.shape, (3, 4))

    def test_output_shape_no_bias(self):
        """Output shape correct when bias=False."""
        layer = TernaryLinear(in_features=16, out_features=5, bias=False)
        x = torch.randn(2, 16)
        out = layer(x)
        self.assertEqual(out.shape, (2, 5))
        self.assertIsNone(layer.bias)

    def test_known_weight_exact_output(self):
        """With manually set weights that quantize predictably, verify exact output."""
        layer = TernaryLinear(in_features=3, out_features=2, bias=False)
        # Set weights to large values so quantization is deterministic:
        # Row 0: [5, -5, 5] -> quantizes to [1, -1, 1]
        # Row 1: [-5, 5, -5] -> quantizes to [-1, 1, -1]
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[5.0, -5.0, 5.0], [-5.0, 5.0, -5.0]]))
        x = torch.tensor([[1.0, 2.0, 3.0]])
        out = layer(x)
        # Expected: x @ q_weight.T
        # q_weight = [[1, -1, 1], [-1, 1, -1]]
        # out[0,0] = 1*1 + 2*(-1) + 3*1 = 1 - 2 + 3 = 2
        # out[0,1] = 1*(-1) + 2*1 + 3*(-1) = -1 + 2 - 3 = -2
        expected = torch.tensor([[2.0, -2.0]])
        self.assertEqual(out.tolist(), expected.tolist())

    def test_known_weight_with_bias(self):
        """With known weights and bias, verify exact output includes bias."""
        layer = TernaryLinear(in_features=2, out_features=2, bias=True)
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[10.0, 10.0], [-10.0, -10.0]]))
            layer.bias.copy_(torch.tensor([0.5, -0.5]))
        x = torch.tensor([[1.0, 1.0]])
        out = layer(x)
        # q_weight = [[1, 1], [-1, -1]]
        # out[0,0] = 1*1 + 1*1 + 0.5 = 2.5
        # out[0,1] = 1*(-1) + 1*(-1) + (-0.5) = -2.5
        expected = torch.tensor([[2.5, -2.5]])
        self.assertEqual(out.tolist(), expected.tolist())

    def test_forward_weights_are_ternary(self):
        """During forward, the effective weights used must be in {-1, 0, +1}."""
        layer = TernaryLinear(in_features=16, out_features=8)
        # Quantize the weights directly to inspect
        q = ternary_quantize(layer.weight)
        unique = set(q.unique().tolist())
        self.assertTrue(unique.issubset({-1.0, 0.0, 1.0}))

    def test_batch_dimension_preserved(self):
        """Various batch sizes should all produce correct output shapes."""
        layer = TernaryLinear(in_features=4, out_features=3)
        for batch_size in [1, 5, 32]:
            x = torch.randn(batch_size, 4)
            out = layer(x)
            self.assertEqual(out.shape, (batch_size, 3))

    def test_gradient_flow(self):
        """output.sum().backward() must not error and must produce finite gradients."""
        layer = TernaryLinear(in_features=8, out_features=4)
        x = torch.randn(2, 8)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(layer.weight.grad)
        self.assertFalse(torch.isnan(layer.weight.grad).any().item())
        self.assertFalse(torch.isinf(layer.weight.grad).any().item())

    def test_gradient_flow_to_input(self):
        """Gradients must flow back to the input tensor."""
        layer = TernaryLinear(in_features=4, out_features=2)
        x = torch.randn(1, 4, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any().item())


class TestTernaryDecoderForward(unittest.TestCase):
    """Tests for TernaryDecoder.forward — output shape, keys, value range, gradient flow."""

    def test_output_keys_both_tiers(self):
        """When both tier1 and tier2 vocab sizes are set, both keys must appear."""
        dec = TernaryDecoder(input_dim=16, hidden_dim=32, tier1_vocab_size=10, tier2_vocab_size=20)
        x = torch.randn(2, 16)
        result = dec(x)
        self.assertIn("tier1_logits", result)
        self.assertIn("tier2_logits", result)

    def test_output_keys_tier1_only(self):
        """When only tier1 vocab is set, only tier1_logits should appear."""
        dec = TernaryDecoder(input_dim=16, hidden_dim=32, tier1_vocab_size=10, tier2_vocab_size=0)
        x = torch.randn(2, 16)
        result = dec(x)
        self.assertIn("tier1_logits", result)
        self.assertNotIn("tier2_logits", result)

    def test_output_keys_tier2_only(self):
        """When only tier2 vocab is set, only tier2_logits should appear."""
        dec = TernaryDecoder(input_dim=16, hidden_dim=32, tier1_vocab_size=0, tier2_vocab_size=20)
        x = torch.randn(2, 16)
        result = dec(x)
        self.assertNotIn("tier1_logits", result)
        self.assertIn("tier2_logits", result)

    def test_output_keys_no_heads(self):
        """When both vocab sizes are 0, result dict should be empty."""
        dec = TernaryDecoder(input_dim=16, hidden_dim=32, tier1_vocab_size=0, tier2_vocab_size=0)
        x = torch.randn(2, 16)
        result = dec(x)
        self.assertEqual(len(result), 0)

    def test_output_shape_tier1(self):
        """tier1_logits shape must be (batch, tier1_vocab_size)."""
        dec = TernaryDecoder(input_dim=8, hidden_dim=16, tier1_vocab_size=5, tier2_vocab_size=0)
        x = torch.randn(3, 8)
        result = dec(x)
        self.assertEqual(result["tier1_logits"].shape, (3, 5))

    def test_output_shape_tier2(self):
        """tier2_logits shape must be (batch, tier2_vocab_size)."""
        dec = TernaryDecoder(input_dim=8, hidden_dim=16, tier1_vocab_size=0, tier2_vocab_size=12)
        x = torch.randn(4, 8)
        result = dec(x)
        self.assertEqual(result["tier2_logits"].shape, (4, 12))

    def test_output_values_finite(self):
        """All output logits must be finite (no NaN/Inf)."""
        dec = TernaryDecoder(input_dim=16, hidden_dim=32, tier1_vocab_size=10, tier2_vocab_size=20)
        x = torch.randn(5, 16)
        result = dec(x)
        for key in ("tier1_logits", "tier2_logits"):
            self.assertFalse(torch.isnan(result[key]).any().item(), f"NaN in {key}")
            self.assertFalse(torch.isinf(result[key]).any().item(), f"Inf in {key}")

    def test_gradient_flow_both_heads(self):
        """Backward through both heads must produce finite gradients on all params."""
        dec = TernaryDecoder(input_dim=16, hidden_dim=32, tier1_vocab_size=10, tier2_vocab_size=20)
        x = torch.randn(2, 16)
        result = dec(x)
        loss = result["tier1_logits"].sum() + result["tier2_logits"].sum()
        loss.backward()
        for name, param in dec.named_parameters():
            self.assertIsNotNone(param.grad, f"No gradient for {name}")
            self.assertFalse(torch.isnan(param.grad).any().item(), f"NaN gradient in {name}")

    def test_gradient_flow_to_input(self):
        """Gradients must flow back to the input tensor."""
        dec = TernaryDecoder(input_dim=8, hidden_dim=16, tier1_vocab_size=5, tier2_vocab_size=10)
        x = torch.randn(1, 8, requires_grad=True)
        result = dec(x)
        loss = result["tier1_logits"].sum() + result["tier2_logits"].sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any().item())

    def test_ternary_disabled_uses_linear(self):
        """With ternary_enabled=False, layers should be nn.Linear, not TernaryLinear."""
        dec = TernaryDecoder(input_dim=8, hidden_dim=16, tier1_vocab_size=5, ternary_enabled=False)
        for module in dec.layers:
            if isinstance(module, (torch.nn.Linear, TernaryLinear)):
                self.assertNotIsInstance(module, TernaryLinear)

    def test_partial_ternary_heads_are_linear(self):
        """With partial_ternary=True, heads should be nn.Linear, hidden layers TernaryLinear."""
        dec = TernaryDecoder(
            input_dim=8,
            hidden_dim=16,
            tier1_vocab_size=5,
            tier2_vocab_size=10,
            ternary_enabled=True,
            partial_ternary=True,
        )
        # Hidden layers should contain TernaryLinear
        has_ternary_hidden = any(isinstance(m, TernaryLinear) for m in dec.layers)
        self.assertTrue(has_ternary_hidden)
        # Heads should be plain nn.Linear, not TernaryLinear
        self.assertNotIsInstance(dec.tier1_head, TernaryLinear)
        self.assertNotIsInstance(dec.tier2_head, TernaryLinear)

    def test_known_input_exact_output_single_layer(self):
        """With 1 hidden layer and known weights, verify exact output values."""
        dec = TernaryDecoder(
            input_dim=2,
            hidden_dim=2,
            tier1_vocab_size=2,
            tier2_vocab_size=0,
            num_layers=1,
            ternary_enabled=True,
        )
        # Set hidden layer weights to large values for predictable quantization
        # layers[0] is the TernaryLinear, layers[1] is ReLU
        hidden_linear = dec.layers[0]
        with torch.no_grad():
            # Weight: [[10, 10], [-10, 10]] -> quantizes to [[1, 1], [-1, 1]]
            hidden_linear.weight.copy_(torch.tensor([[10.0, 10.0], [-10.0, 10.0]]))
            hidden_linear.bias.copy_(torch.zeros(2))
            # Head weight: [[10, -10], [-10, 10]] -> quantizes to [[1, -1], [-1, 1]]
            dec.tier1_head.weight.copy_(torch.tensor([[10.0, -10.0], [-10.0, 10.0]]))
            dec.tier1_head.bias.copy_(torch.zeros(2))

        x = torch.tensor([[1.0, 1.0]])
        result = dec(x)
        # Hidden: x @ q_hidden_weight.T + bias
        # q_hidden = [[1,1],[-1,1]]
        # h = [1*1+1*1, 1*(-1)+1*1] = [2, 0]
        # After ReLU: [2, 0]
        # Head: h @ q_head_weight.T + bias
        # q_head = [[1,-1],[-1,1]]
        # out = [2*1+0*(-1), 2*(-1)+0*1] = [2, -2]
        expected = torch.tensor([[2.0, -2.0]])
        self.assertEqual(result["tier1_logits"].tolist(), expected.tolist())


class TestTernaryLinearExactForward(unittest.TestCase):
    """P0: Exact-value assertions for TernaryLinear.forward with known weights."""

    def test_exact_output_bias_false_3x2(self):
        """Known weights [5,-5,0.1],[-5,5,-5] with bias=False, verify exact output."""
        torch.manual_seed(42)
        layer = TernaryLinear(in_features=3, out_features=2, bias=False)
        with torch.no_grad():
            # Row 0: [5, -5, 0.1] => mean(abs)=3.367, threshold=2.357
            #   5 > 2.357 => +1; -5 < -2.357 => -1; 0.1 < 2.357 and > -2.357 => 0
            # Row 1: [-5, 5, -5] => mean(abs)=5, threshold=3.5
            #   -5 < -3.5 => -1; 5 > 3.5 => +1; -5 < -3.5 => -1
            layer.weight.copy_(torch.tensor([[5.0, -5.0, 0.1], [-5.0, 5.0, -5.0]]))
        x = torch.tensor([[2.0, 3.0, 4.0]])
        out = layer(x)
        # q_weight = [[1, -1, 0], [-1, 1, -1]]
        # out[0,0] = 2*1 + 3*(-1) + 4*0 = 2 - 3 + 0 = -1
        # out[0,1] = 2*(-1) + 3*1 + 4*(-1) = -2 + 3 - 4 = -3
        expected = torch.tensor([[-1.0, -3.0]])
        self.assertTrue(
            torch.allclose(out, expected, atol=1e-6),
            f"Expected {expected.tolist()}, got {out.tolist()}",
        )

    def test_exact_output_bias_true_2x3(self):
        """Known weights with bias=True, verify bias is added to quantized matmul."""
        torch.manual_seed(42)
        layer = TernaryLinear(in_features=2, out_features=3, bias=True)
        with torch.no_grad():
            # All large values => quantize cleanly
            # Row 0: [8, 8] => threshold=0.7*8=5.6 => [+1, +1]
            # Row 1: [-8, 8] => threshold=0.7*8=5.6 => [-1, +1]
            # Row 2: [8, -8] => threshold=0.7*8=5.6 => [+1, -1]
            layer.weight.copy_(torch.tensor([[8.0, 8.0], [-8.0, 8.0], [8.0, -8.0]]))
            layer.bias.copy_(torch.tensor([1.0, -1.0, 0.5]))
        x = torch.tensor([[3.0, 7.0]])
        out = layer(x)
        # q_weight = [[1,1],[-1,1],[1,-1]]
        # out[0,0] = 3*1 + 7*1 + 1.0 = 11.0
        # out[0,1] = 3*(-1) + 7*1 + (-1.0) = 3.0
        # out[0,2] = 3*1 + 7*(-1) + 0.5 = -3.5
        expected = torch.tensor([[11.0, 3.0, -3.5]])
        self.assertTrue(
            torch.allclose(out, expected, atol=1e-6),
            f"Expected {expected.tolist()}, got {out.tolist()}",
        )

    def test_exact_output_multi_batch(self):
        """Multi-row batch with known weights, verify each row independently."""
        torch.manual_seed(42)
        layer = TernaryLinear(in_features=2, out_features=2, bias=False)
        with torch.no_grad():
            # [[10, -10], [10, 10]] => q=[[1,-1],[1,1]]
            layer.weight.copy_(torch.tensor([[10.0, -10.0], [10.0, 10.0]]))
        x = torch.tensor([[1.0, 2.0], [3.0, -1.0], [0.0, 0.0]])
        out = layer(x)
        # q = [[1,-1],[1,1]]
        # row0: [1*1+2*(-1), 1*1+2*1] = [-1, 3]
        # row1: [3*1+(-1)*(-1), 3*1+(-1)*1] = [4, 2]
        # row2: [0, 0]
        expected = torch.tensor([[-1.0, 3.0], [4.0, 2.0], [0.0, 0.0]])
        self.assertTrue(
            torch.allclose(out, expected, atol=1e-6),
            f"Expected {expected.tolist()}, got {out.tolist()}",
        )

    def test_bias_false_has_no_bias_parameter(self):
        """bias=False should register bias as None, not a zero tensor."""
        layer = TernaryLinear(in_features=4, out_features=3, bias=False)
        self.assertIsNone(layer.bias)
        # Verify it doesn't appear in parameters
        param_names = [n for n, _ in layer.named_parameters()]
        self.assertNotIn("bias", param_names)

    def test_bias_true_adds_to_output(self):
        """Difference between bias=True and bias=False should be exactly the bias vector."""
        torch.manual_seed(42)
        layer_bias = TernaryLinear(in_features=3, out_features=2, bias=True)
        layer_nobias = TernaryLinear(in_features=3, out_features=2, bias=False)
        weights = torch.tensor([[10.0, -10.0, 10.0], [-10.0, 10.0, -10.0]])
        bias_val = torch.tensor([0.25, -0.75])
        with torch.no_grad():
            layer_bias.weight.copy_(weights)
            layer_bias.bias.copy_(bias_val)
            layer_nobias.weight.copy_(weights)
        x = torch.tensor([[1.0, 2.0, 3.0]])
        out_bias = layer_bias(x)
        out_nobias = layer_nobias(x)
        diff = out_bias - out_nobias
        self.assertTrue(
            torch.allclose(diff, bias_val.unsqueeze(0), atol=1e-6),
            f"Expected bias diff {bias_val.tolist()}, got {diff.tolist()}",
        )


class TestTernaryLinearWeightQuantization(unittest.TestCase):
    """P0: Verify quantized weights are exactly {-1, 0, +1} with known sign patterns."""

    def test_known_weights_sign_pattern(self):
        """Set specific weights and verify the exact ternary sign pattern after quantization."""
        layer = TernaryLinear(in_features=4, out_features=2, bias=False)
        with torch.no_grad():
            # Row 0: [10, -10, 0.01, 3] => mean(abs)=5.7525, threshold=4.027
            #   10>4.027 => +1; -10<-4.027 => -1; |0.01|<4.027 => 0; |3|<4.027 => 0
            # Row 1: [-3, -3, -3, -3] => mean(abs)=3, threshold=2.1
            #   all -3 < -2.1 => all -1
            layer.weight.copy_(torch.tensor([[10.0, -10.0, 0.01, 3.0], [-3.0, -3.0, -3.0, -3.0]]))
        q = ternary_quantize(layer.weight)
        expected_pattern = torch.tensor([[1.0, -1.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
        self.assertEqual(q.tolist(), expected_pattern.tolist())

    def test_all_values_in_ternary_set_after_forward(self):
        """After a forward pass, the quantized weight values used are in {-1,0,+1}."""
        torch.manual_seed(42)
        layer = TernaryLinear(in_features=16, out_features=8, bias=True)
        x = torch.randn(4, 16)
        _ = layer(x)  # trigger forward
        q = ternary_quantize(layer.weight)
        unique_vals = set(q.unique().tolist())
        self.assertTrue(
            unique_vals.issubset({-1.0, 0.0, 1.0}),
            f"Found non-ternary values: {unique_vals - {-1.0, 0.0, 1.0}}",
        )

    def test_continuous_weights_unchanged_after_forward(self):
        """The underlying fp32 shadow weights should NOT be modified by forward pass."""
        torch.manual_seed(42)
        layer = TernaryLinear(in_features=8, out_features=4, bias=False)
        weight_before = layer.weight.data.clone()
        x = torch.randn(2, 8)
        _ = layer(x)
        self.assertTrue(
            torch.equal(layer.weight.data, weight_before),
            "Shadow weights were modified by forward pass",
        )

    def test_near_threshold_quantization(self):
        """Values near the threshold boundary quantize correctly."""
        layer = TernaryLinear(in_features=4, out_features=1, bias=False)
        with torch.no_grad():
            # [1.0, 1.0, 1.0, 1.0] => mean(abs)=1.0, threshold=0.7
            # All values=1.0 > 0.7 => all +1
            layer.weight.copy_(torch.tensor([[1.0, 1.0, 1.0, 1.0]]))
        q = ternary_quantize(layer.weight)
        self.assertEqual(q.tolist(), [[1.0, 1.0, 1.0, 1.0]])

    def test_below_threshold_quantizes_to_zero(self):
        """Values below the threshold should quantize to 0."""
        layer = TernaryLinear(in_features=4, out_features=1, bias=False)
        with torch.no_grad():
            # [10, 0.1, -0.1, -10] => mean(abs)=5.05, threshold=3.535
            # 10>3.535 => +1; |0.1|<3.535 => 0; |-0.1|<3.535 => 0; -10<-3.535 => -1
            layer.weight.copy_(torch.tensor([[10.0, 0.1, -0.1, -10.0]]))
        q = ternary_quantize(layer.weight)
        self.assertEqual(q.tolist(), [[1.0, 0.0, 0.0, -1.0]])


class TestTernaryDecoderExactValues(unittest.TestCase):
    """P0: TernaryDecoder forward with seeded weights, exact key/shape/value checks."""

    def test_seeded_decoder_exact_keys_and_shapes_ternary_enabled(self):
        """Seed=42, ternary_enabled=True: verify exact output keys and shapes."""
        torch.manual_seed(42)
        dec = TernaryDecoder(
            input_dim=4,
            hidden_dim=8,
            tier1_vocab_size=3,
            tier2_vocab_size=5,
            num_layers=1,
            ternary_enabled=True,
        )
        x = torch.randn(2, 4)
        result = dec(x)
        # Exact keys
        self.assertEqual(set(result.keys()), {"tier1_logits", "tier2_logits"})
        # Exact shapes
        self.assertEqual(result["tier1_logits"].shape, (2, 3))
        self.assertEqual(result["tier2_logits"].shape, (2, 5))

    def test_seeded_decoder_exact_keys_and_shapes_ternary_disabled(self):
        """Seed=42, ternary_enabled=False: verify exact output keys and shapes."""
        torch.manual_seed(42)
        dec = TernaryDecoder(
            input_dim=4,
            hidden_dim=8,
            tier1_vocab_size=3,
            tier2_vocab_size=5,
            num_layers=1,
            ternary_enabled=False,
        )
        x = torch.randn(2, 4)
        result = dec(x)
        self.assertEqual(set(result.keys()), {"tier1_logits", "tier2_logits"})
        self.assertEqual(result["tier1_logits"].shape, (2, 3))
        self.assertEqual(result["tier2_logits"].shape, (2, 5))

    def test_seeded_decoder_output_values_reproducible(self):
        """Same seed + same input => identical output values."""

        def run_once():
            torch.manual_seed(42)
            dec = TernaryDecoder(
                input_dim=4,
                hidden_dim=8,
                tier1_vocab_size=3,
                tier2_vocab_size=0,
                num_layers=1,
                ternary_enabled=True,
            )
            torch.manual_seed(99)
            x = torch.randn(1, 4)
            return dec(x)["tier1_logits"]

        out1 = run_once()
        out2 = run_once()
        self.assertTrue(
            torch.allclose(out1, out2, atol=1e-7),
            f"Non-reproducible outputs: {out1.tolist()} vs {out2.tolist()}",
        )

    def test_exact_forward_known_weights_ternary_enabled(self):
        """Manual weights on small decoder, verify exact tier1_logits values."""
        torch.manual_seed(42)
        dec = TernaryDecoder(
            input_dim=2,
            hidden_dim=2,
            tier1_vocab_size=2,
            tier2_vocab_size=0,
            num_layers=1,
            ternary_enabled=True,
        )
        with torch.no_grad():
            # Hidden layer: layers[0] is TernaryLinear
            dec.layers[0].weight.copy_(torch.tensor([[10.0, -10.0], [-10.0, -10.0]]))
            dec.layers[0].bias.copy_(torch.tensor([0.0, 0.0]))
            # Head: tier1_head is TernaryLinear
            dec.tier1_head.weight.copy_(torch.tensor([[10.0, 10.0], [-10.0, 10.0]]))
            dec.tier1_head.bias.copy_(torch.tensor([0.0, 0.0]))

        x = torch.tensor([[3.0, 1.0]])
        result = dec(x)
        # Hidden q_weight = [[1,-1],[-1,-1]]
        # h = [3*1+1*(-1), 3*(-1)+1*(-1)] = [2, -4]
        # After ReLU: [2, 0]
        # Head q_weight = [[1,1],[-1,1]]
        # out = [2*1+0*1, 2*(-1)+0*1] = [2, -2]
        expected = torch.tensor([[2.0, -2.0]])
        self.assertTrue(
            torch.allclose(result["tier1_logits"], expected, atol=1e-6),
            f"Expected {expected.tolist()}, got {result['tier1_logits'].tolist()}",
        )

    def test_exact_forward_known_weights_ternary_disabled(self):
        """ternary_enabled=False uses nn.Linear (no quantization), verify exact output."""
        torch.manual_seed(42)
        dec = TernaryDecoder(
            input_dim=2,
            hidden_dim=2,
            tier1_vocab_size=2,
            tier2_vocab_size=0,
            num_layers=1,
            ternary_enabled=False,
        )
        with torch.no_grad():
            # Hidden layer: layers[0] is nn.Linear (continuous weights)
            dec.layers[0].weight.copy_(torch.tensor([[0.5, -0.5], [-0.25, 0.75]]))
            dec.layers[0].bias.copy_(torch.tensor([0.0, 0.0]))
            # Head: nn.Linear
            dec.tier1_head.weight.copy_(torch.tensor([[1.0, 2.0], [-1.0, 1.0]]))
            dec.tier1_head.bias.copy_(torch.tensor([0.0, 0.0]))

        x = torch.tensor([[4.0, 2.0]])
        result = dec(x)
        # Hidden: [4*0.5+2*(-0.5), 4*(-0.25)+2*0.75] = [1.0, 0.5]
        # After ReLU: [1.0, 0.5]
        # Head: [1.0*1.0+0.5*2.0, 1.0*(-1.0)+0.5*1.0] = [2.0, -0.5]
        expected = torch.tensor([[2.0, -0.5]])
        self.assertTrue(
            torch.allclose(result["tier1_logits"], expected, atol=1e-6),
            f"Expected {expected.tolist()}, got {result['tier1_logits'].tolist()}",
        )

    def test_ternary_vs_disabled_differ(self):
        """Same weights but ternary_enabled=True vs False should produce different outputs."""
        weights_hidden = torch.tensor([[0.5, -0.8], [0.3, 0.6]])
        weights_head = torch.tensor([[0.9, -0.4], [-0.7, 0.2]])

        def build(ternary_on):
            dec = TernaryDecoder(
                input_dim=2,
                hidden_dim=2,
                tier1_vocab_size=2,
                tier2_vocab_size=0,
                num_layers=1,
                ternary_enabled=ternary_on,
            )
            with torch.no_grad():
                dec.layers[0].weight.copy_(weights_hidden)
                dec.layers[0].bias.copy_(torch.zeros(2))
                dec.tier1_head.weight.copy_(weights_head)
                dec.tier1_head.bias.copy_(torch.zeros(2))
            return dec

        x = torch.tensor([[1.0, 2.0]])
        out_ternary = build(True)(x)["tier1_logits"]
        out_continuous = build(False)(x)["tier1_logits"]
        self.assertFalse(
            torch.allclose(out_ternary, out_continuous, atol=1e-6),
            "Ternary and continuous outputs should differ for non-integer weights",
        )


class TestSTEGradientFlow(unittest.TestCase):
    """P0: Verify STE gradient correctness — gradients pass through quantization as identity."""

    def test_ste_grad_equals_continuous_grad(self):
        """Gradient of loss w.r.t. quantized weights equals continuous weights.

        The STE trick means d(loss)/d(w_continuous) = d(loss)/d(w_quantized) * 1 (identity).
        Since loss = sum(quantized), d(loss)/d(quantized_ij) = 1 for all i,j.
        So d(loss)/d(w_ij) = 1 for all i,j via STE.
        """
        w = torch.randn(4, 8, requires_grad=True)
        q = ternary_quantize(w)
        loss = q.sum()
        loss.backward()
        expected_grad = torch.ones_like(w)
        self.assertTrue(
            torch.allclose(w.grad, expected_grad, atol=1e-7),
            f"STE grad should be all ones, got: {w.grad}",
        )

    def test_ste_grad_with_scaling(self):
        """Gradient through STE with a scalar loss: d(2*sum(q))/d(w) should be 2."""
        w = torch.randn(3, 5, requires_grad=True)
        q = ternary_quantize(w)
        loss = 2.0 * q.sum()
        loss.backward()
        expected_grad = 2.0 * torch.ones_like(w)
        self.assertTrue(
            torch.allclose(w.grad, expected_grad, atol=1e-7),
            f"STE grad with 2x scaling should be all 2s, got: {w.grad}",
        )

    def test_ternary_linear_weight_grad_via_ste(self):
        """TernaryLinear weight gradients match what we'd expect from STE."""
        layer = TernaryLinear(in_features=3, out_features=2, bias=False)
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[10.0, -10.0, 10.0], [-10.0, 10.0, -10.0]]))
        x = torch.tensor([[1.0, 1.0, 1.0]], requires_grad=False)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        # d(loss)/d(out) = [1,1] for each output
        # d(out)/d(q_weight) via F.linear: grad_q_weight[i,j] = x[0,j] for each output i
        # STE: grad_weight = grad_q_weight (identity)
        # So grad_weight[i,j] = x[0,j] = 1.0 for all i,j
        expected_grad = torch.ones(2, 3)
        self.assertTrue(
            torch.allclose(layer.weight.grad, expected_grad, atol=1e-6),
            f"Expected uniform grad of 1.0, got: {layer.weight.grad}",
        )

    def test_ternary_linear_input_grad_matches_quantized_weights(self):
        """Input gradient should reflect the quantized weight values, not continuous."""
        layer = TernaryLinear(in_features=3, out_features=2, bias=False)
        with torch.no_grad():
            # q_weight will be [[1,-1,1],[-1,1,-1]]
            layer.weight.copy_(torch.tensor([[10.0, -10.0, 10.0], [-10.0, 10.0, -10.0]]))
        x = torch.tensor([[2.0, 3.0, 4.0]], requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        # d(loss)/d(out) = [1,1]
        # d(out)/d(x) = q_weight.T summed over output dim
        # x.grad[j] = sum_i q_weight[i,j]
        # q_weight = [[1,-1,1],[-1,1,-1]]
        # x.grad = [1+(-1), -1+1, 1+(-1)] = [0, 0, 0]
        expected_grad = torch.tensor([[0.0, 0.0, 0.0]])
        self.assertTrue(
            torch.allclose(x.grad, expected_grad, atol=1e-6),
            f"Expected x.grad={expected_grad.tolist()}, got: {x.grad.tolist()}",
        )

    def test_ternary_linear_bias_grad_is_standard(self):
        """Bias gradient should be standard (not affected by STE)."""
        layer = TernaryLinear(in_features=2, out_features=3, bias=True)
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[10.0, 10.0], [-10.0, 10.0], [10.0, -10.0]]))
            layer.bias.copy_(torch.tensor([1.0, 2.0, 3.0]))
        x = torch.tensor([[1.0, 1.0]])
        out = layer(x)
        loss = out.sum()
        loss.backward()
        # d(loss)/d(bias) = [1, 1, 1] (standard linear gradient)
        expected_bias_grad = torch.tensor([1.0, 1.0, 1.0])
        self.assertTrue(
            torch.allclose(layer.bias.grad, expected_bias_grad, atol=1e-6),
            f"Expected bias grad {expected_bias_grad.tolist()}, got {layer.bias.grad.tolist()}",
        )

    def test_grad_accumulation_through_two_forward_passes(self):
        """Gradients accumulate correctly across two forward passes without zeroing."""
        layer = TernaryLinear(in_features=2, out_features=1, bias=False)
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[10.0, -10.0]]))  # q = [[1, -1]]

        x1 = torch.tensor([[1.0, 2.0]])
        out1 = layer(x1)
        loss1 = out1.sum()
        loss1.backward()
        grad_after_first = layer.weight.grad.clone()

        x2 = torch.tensor([[3.0, 4.0]])
        out2 = layer(x2)
        loss2 = out2.sum()
        loss2.backward()
        grad_after_second = layer.weight.grad.clone()

        # First: grad = x1 = [1, 2]
        self.assertTrue(
            torch.allclose(grad_after_first, torch.tensor([[1.0, 2.0]]), atol=1e-6),
            f"After first pass: expected [1,2], got {grad_after_first.tolist()}",
        )
        # Accumulated: grad = x1 + x2 = [4, 6]
        self.assertTrue(
            torch.allclose(grad_after_second, torch.tensor([[4.0, 6.0]]), atol=1e-6),
            f"After accumulation: expected [4,6], got {grad_after_second.tolist()}",
        )


if __name__ == "__main__":
    unittest.main()
