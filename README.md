# PROBE-regression

Predict continuous **energy and force error magnitudes** from frozen MACE embeddings.
Derived from [PROBE-forces](https://github.com/zhiyuwang0911/PROBE-forces), extending
[PROBE](https://github.com/isayevlab/PROBE). Original MIT license and attribution retained.

The new regression workflow has no percentile cutoff, class labels, softmax, or
cross-entropy. It estimates errors of an existing potential; it does not replace
the potential or predict signed correction vectors.

| Output | Continuous training target | Unit |
|---|---|---|
| Structure energy | `abs(E_MACE - E_reference)` (total energy, not per atom) | eV |
| Atom force | `mean_xyz(abs(F_MACE - F_reference))` | eV/Å |
| Structure force | Mean of atom force errors over real atoms | eV/Å |

The shared encoder and attention architecture are retained. Energy and atom force
heads each output a scalar through Softplus. Structure force prediction is the
masked mean of predicted atom errors. The loss is a weighted sum of three MSEs
in native units (default weights 1, 1, 1); adjust weights to balance the energy
and force scales. Padding and missing reference labels are excluded from loss
and metrics; structure force supervision requires complete atom force labels.
This predicts magnitudes, not calibrated probabilities or guaranteed bounds.

## Setup and training

```bash
conda env create -f environment_mace.yml
conda activate probe_mace
python train_mace_regression.py \
  --mace-model /path/to/MACE-OFF23_large.model \
  --train-xyz /path/to/train.xyz \
  --output-dir ./probe_regression_outputs \
  --loss-weights 1.0 1.0 0.3
```

Use reference energies in eV and forces in eV/Å. Training splits the supplied
training data into train/validation sets (seed 42, validation fraction 0.1).
Keep a separate test set for final evaluation. Frozen MACE predictions/features
are cached in RAM; no training-set percentile scan is needed.
Use `--enable-cueq` with the supplied cuEquivariance environment if appropriate.

`best_regressor.pt` is selected by weighted validation MSE. Training stops after
20 consecutive epochs without improvement by default (`--patience 20`). `last_checkpoint.pt`
saves model, optimizer, scheduler, random states, and history each epoch.
The checkpoint stores architecture settings for inference. Resume with the same
backbone, data, split, seed, and loss weights:

```bash
python train_mace_regression.py \
  --mace-model /path/to/MACE-OFF23_large.model --train-xyz /path/to/train.xyz \
  --loss-weights 1.0 1.0 0.3 --resume ./probe_regression_outputs/last_checkpoint.pt
```

## Inference

```bash
python infer_mace_regression.py \
  --mace-model /path/to/MACE-OFF23_large.model \
  --checkpoint ./probe_regression_outputs/best_regressor.pt \
  --test-xyz /path/to/test.xyz --output-dir ./regression_predictions
```

Use the same frozen MACE backbone used during training. Outputs are
`predictions_structure_regression.csv`, `predictions_atom_regression.csv`, and
`metrics_regression.json`.
CSV columns explicitly distinguish predicted error from reference error.
Metrics report MAE, RMSE, R² and sample count for each task. Missing references
are blank in CSV and excluded from metrics; undefined R² is JSON `null`.
Inference itself needs no reference energies or forces.

## Tests and scope

```bash
python -m pip install pytest
python -m pytest tests/test_regression.py
```

Tests exercise target definitions, padding, nonnegative outputs, missing labels,
metrics, gradients, training steps, and checkpoint state restoration.
Full MACE training requires your pretrained model and dataset, which are not
included. No scientific accuracy claim is made before training and test evaluation.

Original classifier scripts (including AIMNet2) remain for provenance and
comparison; they still perform classification. Use the two `*_regression.py`
entry points above for the new MACE regressor. See
[CLASSIFICATION_LEGACY.md](CLASSIFICATION_LEGACY.md) for the original documentation.
