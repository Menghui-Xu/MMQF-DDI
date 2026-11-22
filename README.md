# MMQF-DDI

Multi-Scale Molecular Query Fusion for Drug–Drug Interaction Prediction.

## Environment

The provided environment targets Linux and an NVIDIA GPU compatible with CUDA 12.4.

```bash
conda env create -f environment_server.yaml
conda activate MMQF
```

## Reproduction

Run these commands from the repository root:

```bash
# Main experiments: five folds on Deng and Lin
python Run.py

# Cold-start experiments: five folds on Deng and Lin
python Run_cold.py
```

Dataset-specific parameters are configured in the launchers. Edit `gpu`, `datasets`, or `folds` to select the device, datasets, and folds.

Cold-start training uses known–known drug pairs. Each trained model is evaluated separately on S1 / Task 2 (Old–New) and S2 / Task 3 (New–New).

Results are appended to `result/`. Main-experiment files are named `{dataset}_five_fold.csv`. Cold-start files are named `{gpu}_{dataset}_cold_task2.csv` and `{gpu}_{dataset}_cold_task3.csv`. Report the mean of the five final-fold results. Repeated runs append additional rows.

## Key files

| File or directory | Description |
| --- | --- |
| `model.py` | Main model `MMQF` and cold-start model `MMQF_cold`. |
| `main.py`, `Run.py` | Single-fold training and five-fold launcher for the main experiments. |
| `main_cold.py`, `Run_cold.py` | Shared training and separate Task 2 / Task 3 evaluation, plus the five-fold launcher. |
| `environment_server.yaml` | Conda environment and dependency versions. |
| `Deng/`, `Lin/` | Main-experiment splits (`0/`–`4/`), precomputed MolFormer features, BRICS data, and similarity matrices. |
| `cold/Deng/`, `cold/Lin/` | Drug-level cold-start splits (`0/`–`4/`), containing `train.csv`, `test_task2.csv`, and `test_task3.csv`. |

DDI files use zero-based `Drug1`, `Drug2`, and `Label` indices. Keep drug ordering consistent across features and similarity matrices. Deng contains 65 event classes and Lin contains 100. Precomputed molecular features are supplied.
