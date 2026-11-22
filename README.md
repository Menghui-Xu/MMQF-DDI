# MMQF-DDI

Multi-Scale Molecular Query Fusion for Drug–Drug Interaction Prediction.

The main model (`MMQF`) combines MolFormer features, BRICS fragment encoding, multi-query cross-attention, an augmented relation graph, and cross-view contrastive learning. The cold-start model (`MMQF_cold`) uses molecular, target, and enzyme representations and performs prediction without DDI graph propagation.

## Environment

The supplied environment targets Linux and an NVIDIA GPU compatible with CUDA 12.4.

```bash
conda env create -f environment_server.yaml
conda activate MMQF
```

Core versions: Python 3.8.20, PyTorch 2.4.0, PyTorch Geometric 2.5.3, RDKit 2024.3.5. Run commands from the repository root. Precomputed molecular features and fragment data are included, so MolFormer inference is not required to run the experiments.

## Data

Both datasets include five precomputed folds. CSV files use zero-based `Drug1`, `Drug2`, and `Label` indices.

```text
Deng/ or Lin/
  0/ ... 4/                 # main experiment: train.csv, test.csv
  Molformer_smiles_features.pt
  molformer_sim_matrix.csv
  brics_weighted_similarity_matrix.npy
  chem_Jacarrd_sim.csv
  enzyme_Jacarrd_sim.csv
  target_Jacarrd_sim.csv
  brics_fragments_data.pkl
  fragment_graph_data.pkl
  drug_fragment_indices.pkl
cold/Deng/ or cold/Lin/
  0/ ... 4/                 # train.csv, test_task2.csv, test_task3.csv
```

Keep the supplied drug index ordering consistent across DDI records, molecular features, similarity matrices, and fragment data. Deng has 65 event classes and Lin has 100.

**Data issue to resolve before reporting results:** the supplied Lin main-experiment splits contain 5, 4, 4, 4, and 2 identical DDI records shared between training and testing in folds 0–4. Deduplicate the source records and regenerate the folds before using these splits for an independent evaluation. The supplied splits have not been altered during code review.

## Main experiment

```bash
python Run.py
```

This runs five folds on Deng and Lin using the dataset-specific settings in `Run.py`. Change `gpu` or `datasets` there as needed. Results are appended to `result/Deng_five_fold.csv` and `result/Lin_five_fold.csv`.

For one Deng fold with the same settings:

```bash
python main.py --dataset Deng --rels 65 --fold 0 --seed 7 --gpu_id 0 --n_epochs 400 --batch_size 2048 --dropout 0.5 --graph_alpha 0.6 --min_sim 0.6 --temperature 0.2 --cl_alpha 0.7 --w_cl 0.1 --lr 0.002
```

The relation graph uses training DDI edges. Auxiliary edges are constructed from the weighted MolFormer and BRICS similarities, with overlap filtering based on training DDI pairs.

## Cold-start experiment

```bash
python Run_cold.py
```

This runs five folds on both datasets. Each fold trains once on known–known drug pairs, then separately evaluates S1 / Task 2 (known–unseen) and S2 / Task 3 (unseen–unseen). The held-out drug subsets are mutually disjoint across the five folds.

For the default `gpu=0`, results are appended to:

```text
result/0_Deng_cold_task2.csv
result/0_Deng_cold_task3.csv
result/0_Lin_cold_task2.csv
result/0_Lin_cold_task3.csv
```

A single-fold run uses `main_cold.py` with the same parameter arguments shown above. Its default model is `MMQF_cold`.

## Outputs and evaluation

Scripts evaluate the final training epoch. Each CSV row contains the model, seed, fold, settings, and six metrics: ACC, micro AUPRC, micro AUROC, macro F1, macro precision, and macro recall. AUPRC is computed by trapezoidal integration of the precision–recall curve. Report the mean over the five final-fold rows. Re-running a launcher appends additional rows to the same files.

The default training seed is 7; `--seed` controls model initialization and training-data shuffling. Contrastive learning uses the representations of all drugs at each training iteration, while the classification batch size counts drug pairs. Cold-start evaluation allows access to the held-out drugs' unlabeled molecular and attribute features during training.
