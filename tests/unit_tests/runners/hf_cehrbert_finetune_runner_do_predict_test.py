import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch
from torch.utils.data import DataLoader

from cehrbert.runners.hf_cehrbert_finetune_runner import do_predict


class StubClassifier(torch.nn.Module):
    """Stands in for a fine-tuned classifier, returning logits of shape (batch_size, 1)."""

    def forward(self, classifier_label, **kwargs):
        logits = classifier_label.float() * 2 - 1
        return SimpleNamespace(loss=torch.tensor(0.5), logits=logits)


def make_samples(num_samples, with_index_date=True):
    samples = []
    for i in range(num_samples):
        sample = {"person_id": torch.tensor(i), "classifier_label": torch.tensor([i % 2])}
        if with_index_date:
            sample["index_date"] = torch.tensor(1_700_000_000.0 + i * 86_400, dtype=torch.float64)
        samples.append(sample)
    return samples


class DoPredictTest(unittest.TestCase):

    def run_do_predict(self, samples, batch_size):
        with tempfile.TemporaryDirectory() as output_dir:
            with patch(
                "cehrbert.runners.hf_cehrbert_finetune_runner.load_finetuned_model", return_value=StubClassifier()
            ):
                do_predict(
                    DataLoader(samples, batch_size=batch_size),
                    SimpleNamespace(use_lora=False),
                    SimpleNamespace(output_dir=output_dir),
                )
            predictions = pd.read_parquet(Path(output_dir) / "test_predictions")
            metrics = json.loads((Path(output_dir) / "test_results.json").read_text())
        return predictions.sort_values("subject_id").reset_index(drop=True), metrics

    def test_last_batch_with_a_single_sample(self):
        # 3 samples in batches of 2 leave a final batch of one sample
        predictions, metrics = self.run_do_predict(make_samples(3), batch_size=2)
        self.assertEqual([0, 1, 2], predictions.subject_id.tolist())
        self.assertEqual([False, True, False], predictions.boolean_value.tolist())
        self.assertFalse(predictions.prediction_time.isna().any())
        self.assertIn("test_loss", metrics)

    def test_batch_size_one(self):
        predictions, _ = self.run_do_predict(make_samples(3), batch_size=1)
        self.assertEqual([0, 1, 2], predictions.subject_id.tolist())
        self.assertFalse(predictions.prediction_time.isna().any())

    def test_single_sample_batch_without_index_date(self):
        predictions, _ = self.run_do_predict(make_samples(3, with_index_date=False), batch_size=2)
        self.assertEqual([0, 1, 2], predictions.subject_id.tolist())
        self.assertTrue(predictions.prediction_time.isna().all())


if __name__ == "__main__":
    unittest.main()
