#!/usr/bin/env python3
"""Export continuous predicted errors; references are optional at inference."""
import argparse
import csv
import json
from pathlib import Path
import torch
from probe.regression import ErrorRegressor, error_targets, regression_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mace-model', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--test-xyz', required=True)
    p.add_argument('--output-dir', default='probe_regression_inference')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--enable-cueq', action='store_true')
    args = p.parse_args()
    from probe.backends.mace import load_mace, get_z_table, load_extxyz_dataloader, process_batch_mace_multitask
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if ckpt.get('task') != 'error_regression':
        raise ValueError('Expected an error_regression checkpoint, not classifier weights')
    model = ErrorRegressor(**ckpt['model_config']).to(args.device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    extractor = load_mace(args.mace_model, args.device, enable_cueq=args.enable_cueq)
    loader = load_extxyz_dataloader(args.test_xyz, get_z_table(extractor),
                                    float(extractor.mace_model.r_max), batch_size=args.batch_size, shuffle=False)
    structures, atoms = [], []
    store = {k: [[], []] for k in ('energy', 'force_atom', 'force_structure')}
    offset = 0
    for batch in loader:
        feats, mask, pe, te, pf, tf, _ = process_batch_mace_multitask(batch, args.device, extractor)
        valid = torch.isfinite(pe) & mask.any(1)
        valid &= torch.isfinite(feats).all(-1).masked_fill(~mask, True).all(1)
        valid &= torch.isfinite(pf).all(-1).masked_fill(~mask, True).all(1)
        if not valid.all():
            raise ValueError(f'Nonfinite backbone output or empty structure in batch starting at {offset}')
        with torch.no_grad():
            predictions = model(feats.float(), mask, pe.float())
        targets = error_targets(pe, te, pf, tf, mask)
        if not all(torch.isfinite(x).all() for x in predictions):
            raise ValueError('Nonfinite regressor output')
        def number(x):
            return x.item() if torch.isfinite(x) else None
        for b in range(len(pe)):
            structures.append(dict(structure_index=offset + b,
                predicted_energy_error_eV=number(predictions[0][b]),
                reference_energy_error_eV=number(targets[0][b]),
                predicted_mean_force_error_eV_A=number(predictions[2][b]),
                reference_mean_force_error_eV_A=number(targets[2][b])))
            for a in mask[b].nonzero().flatten().tolist():
                atoms.append(dict(structure_index=offset + b, atom_index=a,
                    predicted_force_error_eV_A=number(predictions[1][b, a]),
                    reference_force_error_eV_A=number(targets[1][b, a])))
        for i, key in enumerate(store):
            pred, target = predictions[i].detach().cpu(), targets[i].detach().cpu()
            if i == 1:
                pred, target = pred[mask.cpu()], target[mask.cpu()]
            store[key][0].append(pred)
            store[key][1].append(target)
        offset += len(pe)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not structures:
        raise ValueError('No structures available for inference')
    for name, rows in [('structure', structures), ('atom', atoms)]:
        with (out / f'predictions_{name}.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    metrics = {key: regression_metrics(torch.cat(pred), torch.cat(target))
               for key, (pred, target) in store.items()}
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
