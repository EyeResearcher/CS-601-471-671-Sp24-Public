import unittest
from unittest.mock import patch

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import BertConfig, BertModel

import base_classification
import classification


class Accuracy:
    def __init__(self):
        self.correct = 0
        self.total = 0

    def add_batch(self, predictions, references):
        self.correct += (predictions == references).sum().item()
        self.total += len(references)

    def compute(self):
        return {"accuracy": self.correct / self.total}


class ClassificationTests(unittest.TestCase):
    def make_model(self, tuning_type):
        config = BertConfig(
            vocab_size=32, hidden_size=16, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=32,
            max_position_embeddings=32, hidden_dropout_prob=0,
            attention_probs_dropout_prob=0,
        )
        with patch.object(classification.AutoModel, "from_pretrained", return_value=BertModel(config)):
            return classification.CustomModelforSequenceClassification(
                "unused", type=tuning_type, prefix_length=3
            )

    def test_tokenizer_uses_model_specific_pair_format(self):
        class Tokenizer:
            def encode_plus(self, question, **kwargs):
                self.question = question
                self.kwargs = kwargs
                return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}

        for module in (base_classification, classification):
            tokenizer = Tokenizer()
            dataset = module.BoolQADataset(["passage"], ["question"], [True], tokenizer, 8)
            self.assertEqual(dataset[0]["labels"].item(), 1)
            self.assertEqual(tokenizer.question, "question")
            self.assertEqual(tokenizer.kwargs["text_pair"], "passage")
            self.assertEqual(tokenizer.kwargs["truncation"], "only_second")

    def test_tuning_modes_update_only_intended_parameters(self):
        ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        for tuning_type in ("full", "head", "prefix"):
            with self.subTest(tuning_type=tuning_type):
                model = self.make_model(tuning_type)
                logits = model(ids, mask)["logits"]
                self.assertEqual(tuple(logits.shape), (2, 2))
                logits.sum().backward()
                self.assertIsNotNone(model.classifier.weight.grad)
                self.assertEqual(model.prefix.grad is not None, tuning_type == "prefix")
                self.assertEqual(
                    model.model.embeddings.word_embeddings.weight.grad is not None,
                    tuning_type == "full",
                )

    def test_lora_training_updates_adapter_and_head(self):
        model = self.make_model("full")
        model.model = get_peft_model(model.model, LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, r=2, lora_alpha=4,
            target_modules=["query", "value"],
        ))
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        batch = {
            "input_ids": torch.tensor([[1, 2, 0], [3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
            "labels": torch.tensor([0, 1]),
        }
        with patch.object(classification.evaluate, "load", side_effect=lambda _: Accuracy()), \
                patch.object(classification, "plot"):
            classification.train(model, 1, [batch], [batch], [batch], "cpu", 0.01, "model")
        changed = {name for name, p in model.named_parameters() if not torch.equal(before[name], p)}
        self.assertTrue(any("lora_" in name for name in changed))
        self.assertTrue(any(name.startswith("classifier.") for name in changed))
        self.assertFalse(any("base_layer" in name for name in changed))


if __name__ == "__main__":
    unittest.main()
