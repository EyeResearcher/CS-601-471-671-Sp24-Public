# HW3 training

`train_hw3.py` runs all 12 experiments already implemented in `main.py`,
using the original hyperparameters and 20 epochs per experiment. It fixes
the random seed at 42 and reuses the same sentence features across sweeps.
Training uses four CPU threads. Models are selected by development accuracy;
each selected checkpoint is then evaluated on the test set.

The local `data` directory contains the existing downloaded IMDB train/test
Arrow files, GloVe Twitter 50-dimensional embeddings, and NLTK Punkt tokenizer.
The GloVe archive was verified against Gensim's published cached checksum:
`c168f18641f8c8a00fe30984c4799b2b`. No new download was needed for these assets.
The official train split supplies 20,000 training and 5,000 development
examples; 10,000 examples from the official test split are used for testing,
matching `mlp.py`.

From this directory, run:

```powershell
.\.venv\Scripts\python.exe -u train_hw3.py --output runs/manual-run
```

Use a new output directory for each run. Each run saves model checkpoints,
comparison plots, per-epoch metrics in JSON, package versions, and `status.json`.
Background launches also save stdout and stderr logs in that directory.
To launch a new run in the background with a timestamped output folder:

```powershell
.\start_training.ps1
```

`runs/latest.txt` records the latest run directory; its `training.log` contains
epoch evaluations, while `progress.log` contains progress bars and warnings.

The Python 3.10.13 virtual environment uses `requirements.txt`, plus
`scipy==1.12.0` and `pyarrow==14.0.2` for compatibility with the assignment's
older Gensim and Datasets versions. Setup command:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements.txt scipy==1.12.0 pyarrow==14.0.2
```

The assignment PDF is not present in this checkout; the run follows `hw3.md`
and the implemented configurations in `main.py`.
