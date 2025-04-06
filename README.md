# Chunking Evaluation Task

This repository has the following folders:

- `data`: Contains the data files used for the evaluation task.
- `src`: Contains the source code.
- `results`: Contains the results of the evaluation task.


## Data

For an evaluation I chose [state_of_the_union.md](./data/corpora/state_of_the_union.md) corpus but, of course, you can use any other corpus. You can specify it in the config in [pipeline.py](./src/pipeline.py).

## src
Detailed explanation of the source code can be found in the [report.pdf](./results/report.pdf).

## Results
In this folder you can find metrics and results of the evaluation. It contains dataframe with metrics ([metrics.csv](./results/metrics.csv)), notebook with experiments ([experiments.ipynb](./results/experiments.ipynb)) and [report.pdf](./results/report.pdf) with all my thoughts and conclusions.