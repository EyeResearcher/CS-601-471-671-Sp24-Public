# CS 601.471/671: Self-supervised Models
## Homework 6: Finetuning, PEFT, and In-context Learning

### Overview
In this programming homework, we will finetune encoder LMs for a classification task with yes/no questions. In
particular, we will
- Implement full-parameter finetuning.
- Explore various parameter-efficient finetuning strategies.

Additionally, we will also experiment with in-context learning using GPT-3.5 models via OpenAI APIs
### Setup
create a new environment for this homework:
```
conda create -n ssm_hw6 python=3.10.13
```

And install the required packages:
```
cd hw6
conda activate ssm_hw6
pip install -r requirements.txt
```

### Run the code
You can create your own sbatch scripts to run the code.

Both classification scripts use `--seed 42` by default, so the 8,000-example
training split and held-out test split are reproducible across runs. For example:

```
python base_classification.py --small_subset --device cpu --batch_size 10 --lr 1e-4 --num_epochs 20
python classification.py --type prefix --device cuda --num_epochs 5
python classification_lora.py --device cuda --rank 8 --alpha 16
python openai_gpt-boolq.py
```

The LoRA script targets the query and value projections in BERT/RoBERTa-style
models and reports the number of trainable parameters. The OpenAI script requires
`OPENAI_API_KEY` and incurs API charges. Training runs download the BoolQ data
and model weights and may require a GPU; the assignment's plots and reported
accuracies must be obtained from actual runs.

### Submission
Please follow the instructions in the homework pdf.
