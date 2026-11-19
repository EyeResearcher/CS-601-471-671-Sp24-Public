"""LoRA fine-tuning of the BoolQ classifier."""

import argparse

from peft import LoraConfig, TaskType, get_peft_model
from transformers import set_seed

from classification import evaluate_model, pre_process, print_gpu_memory, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small_subset", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="roberta-base")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    args = parser.parse_args()
    if args.rank < 1:
        parser.error("--rank must be positive")

    print(f"Specified arguments: {args}")
    set_seed(args.seed)
    model, train_loader, validation_loader, test_loader = pre_process(
        args.model, args.batch_size, args.device, args.small_subset,
        type="full", seed=args.seed,
    )
    config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=["query", "value"],
    )
    model.model = get_peft_model(model.model, config)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    train(model, args.num_epochs, train_loader, validation_loader, test_loader,
          args.device, args.lr, f"{args.model}_lora")
    print_gpu_memory()
    print(f" - Average DEV metrics: accuracy={evaluate_model(model, validation_loader, args.device)}")
    print(f" - Average TEST metrics: accuracy={evaluate_model(model, test_loader, args.device)}")


if __name__ == "__main__":
    main()
