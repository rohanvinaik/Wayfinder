"""Mutation-prescribed tests for nav_model_factory (VALUE + SWAP + TYPE)."""

import unittest

from src.nav_model_factory import resolve_model_config


class TestResolveModelConfig(unittest.TestCase):
    """VALUE + SWAP + TYPE prescriptions."""

    def _base_config(self):
        return {
            "model": {
                "encoder": {"model_name": "base-encoder", "embed_dim": 384},
                "goal_analyzer": {"feature_dim": 256},
                "bridge": {"bridge_dim": 128},
                "navigator": {"hidden_dim": 256, "num_anchors": 300, "num_layers": 2},
            }
        }

    def test_no_checkpoint_returns_config_model(self):
        config = self._base_config()
        result = resolve_model_config(config, checkpoint=None)
        self.assertEqual(result["encoder"]["model_name"], "base-encoder")

    def test_checkpoint_overrides_config(self):
        config = self._base_config()
        ckpt = {
            "config": {
                "model": {
                    "encoder": {"model_name": "ckpt-encoder", "embed_dim": 768},
                    "goal_analyzer": {"feature_dim": 512},
                    "bridge": {"bridge_dim": 64},
                    "navigator": {"hidden_dim": 128, "num_anchors": 100, "num_layers": 1},
                }
            }
        }
        result = resolve_model_config(config, ckpt)
        self.assertEqual(result["encoder"]["model_name"], "ckpt-encoder")
        self.assertEqual(result["encoder"]["embed_dim"], 768)

    def test_swap_config_checkpoint(self):
        """SWAP: checkpoint model != config model produces different result."""
        config = self._base_config()
        ckpt = {"config": {"model": {"encoder": {"model_name": "different"}}}}
        with_ckpt = resolve_model_config(config, ckpt)
        without_ckpt = resolve_model_config(config, None)
        self.assertNotEqual(with_ckpt, without_ckpt)

    def test_returns_deepcopy(self):
        """VALUE: result is a copy, not a reference to original."""
        config = self._base_config()
        result = resolve_model_config(config, None)
        result["encoder"]["model_name"] = "mutated"
        self.assertEqual(config["model"]["encoder"]["model_name"], "base-encoder")

    def test_checkpoint_without_config_key_falls_back(self):
        """TYPE: checkpoint with no 'config' key uses base config."""
        config = self._base_config()
        ckpt = {"step": 1000, "modules": {}}
        result = resolve_model_config(config, ckpt)
        self.assertEqual(result["encoder"]["model_name"], "base-encoder")

    def test_checkpoint_config_not_dict_falls_back(self):
        """TYPE: checkpoint['config'] is not a dict → fall back."""
        config = self._base_config()
        ckpt = {"config": "not a dict"}
        result = resolve_model_config(config, ckpt)
        self.assertEqual(result["encoder"]["model_name"], "base-encoder")

    def test_checkpoint_config_no_model_falls_back(self):
        """TYPE: checkpoint['config'] has no 'model' key → fall back."""
        config = self._base_config()
        ckpt = {"config": {"training": {"lr": 0.001}}}
        result = resolve_model_config(config, ckpt)
        self.assertEqual(result["encoder"]["model_name"], "base-encoder")


if __name__ == "__main__":
    unittest.main()
