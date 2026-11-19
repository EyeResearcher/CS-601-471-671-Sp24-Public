"""Evaluate balanced eight-shot prompting on a small BoolQ validation sample."""

import argparse
import random

from datasets import load_dataset
from openai import OpenAI


def format_example(example, include_answer=True):
    text = f"Passage: {example['passage']}\nQuestion: {example['question']}\nAnswer:"
    if include_answer:
        text += " Yes" if example['answer'] else " No"
    return text


def build_prompt(demonstrations, query):
    return (
        "Answer each question using its passage. Reply with Yes or No only.\n\n"
        + "\n\n".join(format_example(example) for example in demonstrations)
        + "\n\n"
        + format_example(query, include_answer=False)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_examples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="gpt-3.5-turbo-instruct")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    data = load_dataset("boolq")
    train = list(data["train"])
    validation = list(data["validation"])
    yes = rng.sample([x for x in train if x["answer"]], 4)
    no = rng.sample([x for x in train if not x["answer"]], 4)
    demonstrations = [example for pair in zip(yes, no) for example in pair]
    queries = rng.sample(validation, args.num_examples)
    client = OpenAI()
    correct = 0
    for index, query in enumerate(queries, 1):
        response = client.completions.create(
            model=args.model,
            prompt=build_prompt(demonstrations, query),
            max_tokens=3,
            temperature=0,
        )
        answer = response.choices[0].text.strip().lower().split()
        prediction = answer[0].rstrip(".,!") if answer else ""
        gold = "yes" if query["answer"] else "no"
        correct += prediction == gold
        print(f"{index}/{len(queries)} prediction={prediction!r} gold={gold} correct={correct}")
    print(f"Accuracy: {correct}/{len(queries)} = {correct / len(queries):.4f}")


if __name__ == "__main__":
    main()
