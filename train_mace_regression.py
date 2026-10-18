#!/usr/bin/env python3
"""Train continuous energy/force error regression on frozen MACE embeddings."""
import argparse
import json
from pathlib import Path
import torch
from probe.regression import ErrorRegressor, run_epoch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mace-model', required=True)
    p.add_argument('--train-xyz', required=True)
    p.add_argument('--output-dir', default='probe_regression_outputs')
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--valid-fraction', type=float, default=.1)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--enable-cueq', action='store_true')
    p.add_argument('--loss-weights', type=float, nargs=3, default=[1., 1., 1.],
                   metavar=('ENERGY', 'FORCE_ATOM', 'FORCE_STRUCTURE'))
    p.add_argument('--resume', help='Path to regression last_checkpoint.pt')
    args = p.parse_args()
    if not 0 < args.valid_fraction < 1:
        p.error('--valid-fraction must be between zero and one')
    if any(w < 0 or not __import__('math').isfinite(w) for w in args.loss_weights) or not any(args.loss_weights):
        p.error('--loss-weights must be finite, nonnegative and include a positive weight')
    from probe.backends.mace import load_mace, get_z_table, train_val_split_loader, CachedMACEProcessor
    torch.manual_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    extractor = load_mace(args.mace_model, args.device, enable_cueq=args.enable_cueq)
    train, val = train_val_split_loader(args.train_xyz, get_z_table(extractor),
                float(extractor.mace_model.r_max), args.batch_size, args.valid_fraction, seed=args.seed)
    if not len(train.dataset) or not len(val.dataset):
        raise ValueError('Both train and validation sets must contain structures')
    processor = CachedMACEProcessor(extractor, compute_force=True)
    config = {'backbone_dim': extractor.feat_dim}
    model = ErrorRegressor(**config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=.9)
    best, stale, start, history = float('inf'), 0, 0, []
    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)
        if ckpt.get('task') != 'error_regression' or ckpt['model_config'] != config:
            raise ValueError('Expected a compatible regression checkpoint')
        for key in ('mace_model', 'train_xyz', 'valid_fraction', 'seed', 'loss_weights'):
            if ckpt['args'][key] != vars(args)[key]:
                raise ValueError(f'Resume requires unchanged {key}')
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        best, stale, start = ckpt['best'], ckpt['stale'], ckpt['epoch'] + 1
        history = ckpt['history']
        torch.set_rng_state(ckpt['rng_state'].cpu())
        if torch.cuda.is_available() and ckpt.get('cuda_rng_state') is not None:
            torch.cuda.set_rng_state_all(ckpt['cuda_rng_state'])
    for epoch in range(start, args.epochs):
        train_loss, _ = run_epoch(model, processor, train, args.device, args.loss_weights, optimizer)
        val_loss, metrics = run_epoch(model, processor, val, args.device, args.loss_weights)
        scheduler.step(val_loss)
        improved = val_loss < best
        best, stale = (val_loss, 0) if improved else (best, stale + 1)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'metrics': metrics})
        ckpt = dict(task='error_regression', model_config=config, model_state_dict=model.state_dict(),
                    optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), epoch=epoch,
                    best=best, stale=stale, history=history, args=vars(args),
                    rng_state=torch.get_rng_state(),
                    cuda_rng_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        for name in (['last_checkpoint.pt', 'best_regressor.pt'] if improved else ['last_checkpoint.pt']):
            tmp = out / (name + '.tmp')
            torch.save(ckpt, tmp)
            tmp.replace(out / name)
        (out / 'history.json').write_text(json.dumps(history, indent=2, allow_nan=False))
        print(f'Epoch {epoch + 1}: train={train_loss:.6g} val={val_loss:.6g} {metrics}', flush=True)
        if stale >= args.patience:
            break


if __name__ == '__main__':
    main()
