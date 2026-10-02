"""Run the implemented HW3 experiments with local data and persistent results."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import random
import traceback

ROOT = Path(__file__).resolve().parent
os.environ['MPLBACKEND'] = 'Agg'
os.environ['MPLCONFIGDIR'] = str(ROOT / 'data' / 'matplotlib')
os.environ['NLTK_DATA'] = str(ROOT / 'data' / 'nltk_data')
os.environ['HF_HOME'] = str(ROOT / 'data' / 'huggingface')
os.environ['HF_DATASETS_OFFLINE'] = '1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(output)

    def status(**fields):
        temporary = output / 'status.tmp'
        temporary.write_text(json.dumps(fields, indent=2), encoding='utf-8')
        temporary.replace(output / 'status.json')

    try:
        status(state='preparing', pid=os.getpid())
        import numpy as np
        import torch
        from datasets import Dataset
        from gensim.models import KeyedVectors
        import main as experiments

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        versions = {p: importlib.metadata.version(p) for p in
                    ['torch', 'numpy', 'scipy', 'datasets', 'gensim', 'nltk', 'matplotlib']}
        (output / 'environment.json').write_text(json.dumps(versions, indent=2))
        print('Runtime:', versions, flush=True)
        print('Device: CPU; seed: 42; threads: 4', flush=True)
        train = Dataset.from_file(str(ROOT / 'data/imdb/imdb-train.arrow'))
        test = Dataset.from_file(str(ROOT / 'data/imdb/imdb-test.arrow'))
        assert len(train) == 25000 and len(test) == 25000
        train = train.shuffle(seed=42, keep_in_memory=True)
        test = test.shuffle(seed=42, keep_in_memory=True)
        dev_data, train_data, test_data = train[:5000], train[5000:25000], test[:10000]
        print('IMDB splits: train=20000, dev=5000, test=10000', flush=True)
        print('Loading verified glove-twitter-50 embeddings...', flush=True)
        embeddings = KeyedVectors.load_word2vec_format(str(ROOT / 'data/glove-twitter-50.gz'))
        assert embeddings.vector_size == 50

        # The same fixed features are reused across the homework's three sweeps.
        feature_cache = {}
        featurize = experiments.create_tensor_dataset

        def cached_features(raw, emb):
            if id(raw) not in feature_cache:
                feature_cache[id(raw)] = featurize(raw, emb)
                print('Feature rows:', len(feature_cache[id(raw)]), flush=True)
            return feature_cache[id(raw)]

        experiments.create_tensor_dataset = cached_features
        original_run = experiments.run_mlp
        results = []

        def recorded_run(config, *datasets):
            name = Path(config.save_path).stem
            status(state='training', pid=os.getpid(), experiment=name,
                   completed=len(results), total=12)
            print('\nExperiment:', dict(config), flush=True)
            values = original_run(config, *datasets)
            record = {'config': dict(config)}
            for key, value in zip(['train_loss', 'train_accuracy', 'dev_loss',
                                   'dev_accuracy', 'test_loss', 'test_accuracy'], values):
                record[key] = [float(v) for v in value] if isinstance(value, list) else float(value)
            record['best_epoch'] = int(np.argmax(record['dev_accuracy']))
            record['best_dev_accuracy'] = max(record['dev_accuracy'])
            (output / (name + '.json')).write_text(json.dumps(record, indent=2))
            results.append(record)
            (output / 'results.json').write_text(json.dumps(results, indent=2))
            return values

        experiments.run_mlp = recorded_run
        for sweep in [experiments.explore_mlp_structures,
                      experiments.explore_mlp_activations,
                      experiments.explore_mlp_learning_rates]:
            sweep(dev_data, train_data, test_data, embeddings)
        status(state='complete', pid=os.getpid(), completed=len(results), total=12)
        print('All 12 experiments completed.', flush=True)
    except BaseException:
        status(state='failed', pid=os.getpid(), error=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
