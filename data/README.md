# About the data

Since some of the datasets are already pre-processed to enable better handling with the `wuenlp` package, the datasets are already included in this repo.

### Structure of the data

Each dataset has one of the following structures:

```
data/<task_name>/(<dataset_name>)/<split>/<file_name>.json|xmi

data/<task_name>/(<dataset_name>)/<split>.json|xmi
```

Where `<task_name>` is one of the following:

- `absa` (Aspect-based Sentiment Analysis)
- `ssa` (Structured Sentiment Analysis)
- `sre` (Sentiment Relationship Extraction)
- `deft` (Definition Extraction from Free Text)
- `spaceeval` (Space Evaluation)
- `gabsa` (German Aspect-based Sentiment Analysis)
- `gner` (German Named Entity Recognition)

And `<split>` has to be all of the following:

- `train`
- `valid`
- `test`


## About the ABSA datasets

The data can also be re-downloaded from the original Paper repo: https://github.com/yhcc/BARTABSA/tree/main/data 

To simplify the process, you can just run the following command (in an environment with the `wuenlp` package installed):

```bash
python data/scripts/download_and_preprocess.py
```