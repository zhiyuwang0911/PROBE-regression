import torch
from probe.pca import PCAProjection, ModelWithPCA, fit_streaming_pca, restore_model


def test_projection_and_padding():
    projection = PCAProjection([1., 2., 3.], [[1., 0., 0.], [0., 1., 0.]])
    x = torch.tensor([[[3., 5., 8.], [0., 0., 0.]]])
    mask = torch.tensor([[True, False]])
    assert torch.allclose(projection(x, mask), torch.tensor([[[2., 3.], [0., 0.]]]))


def test_checkpoint_reconstructs_same_coordinates():
    from probe.model import PROBEModel
    p = PCAProjection([1., 2., 3.], [[1., 0., 0.], [0., 1., 0.]])
    probe = PROBEModel(backbone_dim=2, atom_encoder_output_dim=32,
                       mol_attention_heads=4, dropout=0.0)
    model = ModelWithPCA(probe, p).eval()
    x = torch.randn(2, 4, 3)
    mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
    energy = torch.randn(2)
    y = model(x, mask, energy=energy)
    cfg = dict(backbone_dim=2, atom_encoder_output_dim=32, mol_attention_heads=4,
               dropout=0.0)
    checkpoint = dict(model_config=cfg, pca_dim=2, model_state_dict=model.state_dict())
    restored = restore_model(PROBEModel, checkpoint, 'cpu').eval()
    assert torch.allclose(y, restored(x, mask, energy=energy), atol=1e-6)


def test_fit_train_data_only():
    batch = (torch.tensor([[[1., 2., 3.], [2., 3., 4.], [3., 4., 5.], [4., 5., 6.]]]),
             torch.tensor([[True, True, True, True]]))
    def processor(b, device):
        return b[0], b[1], None
    projection, info = fit_streaming_pca(processor, [batch], 'cpu', 2, 3)
    assert info['n_training_atoms'] == 4
    assert torch.allclose(projection.mean, torch.tensor([2.5, 3.5, 4.5]))


def test_disabled_pca_preserves_model_checkpoint():
    from probe.model import PROBEModel
    from probe.pca import make_pca
    projection, info = make_pca(None, None, 'cpu', None, 3)
    assert projection is None and info is None
    cfg = dict(backbone_dim=3, atom_encoder_output_dim=32, mol_attention_heads=4, dropout=0.)
    model = PROBEModel(**cfg).eval()
    restored = restore_model(PROBEModel, dict(model_config=cfg, pca_dim=None,
                             model_state_dict=model.state_dict()), 'cpu').eval()
    x, mask = torch.randn(2, 4, 3), torch.ones(2, 4, dtype=torch.bool)
    assert torch.allclose(model(x, mask), restored(x, mask))


def test_invalid_pca_options():
    import pytest
    from probe.pca import make_pca
    for dim in (0, -1, 4):
        with pytest.raises(ValueError):
            make_pca(None, None, 'cpu', dim, 3)


def test_regression_training_resume_inference(monkeypatch, tmp_path):
    import sys
    import train_mace_regression as training
    import infer_mace_regression as inference
    import probe.backends.mace as backend
    from probe.regression import ErrorRegressor
    from types import SimpleNamespace
    torch.manual_seed(11)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    batch = (torch.randn(2, 3, 4), mask, torch.randn(2), torch.randn(2),
             torch.randn(2, 3, 3), torch.randn(2, 3, 3), mask.sum(1))
    class Loader(list):
        dataset = [0, 1]
    loader = Loader([batch])
    extractor = SimpleNamespace(feat_dim=4, mace_model=SimpleNamespace(r_max=4.))
    monkeypatch.setattr(backend, 'load_mace', lambda *a, **k: extractor)
    monkeypatch.setattr(backend, 'get_z_table', lambda *a: None)
    monkeypatch.setattr(backend, 'train_val_split_loader', lambda *a, **k: (loader, loader))
    monkeypatch.setattr(backend, 'CachedMACEProcessor', lambda *a, **k: lambda b,d: b)
    monkeypatch.setattr(backend, 'load_extxyz_dataloader', lambda *a, **k: loader)
    monkeypatch.setattr(backend, 'process_batch_mace_multitask', lambda b,*a: b)
    def small_model(**cfg):
        return ErrorRegressor(**cfg, atom_encoder_output_dim=16, mol_attention_heads=4, dropout=0.)
    monkeypatch.setattr(training, 'ErrorRegressor', small_model)
    monkeypatch.setattr(inference, 'ErrorRegressor', small_model)
    out = tmp_path/'train'
    argv = ['train', '--mace-model', 'fake', '--train-xyz', 'fake', '--device', 'cpu',
            '--output-dir', str(out)]
    monkeypatch.setattr(sys, 'argv', argv + ['--epochs', '1', '--pca-dim', '2'])
    training.main()
    before = torch.load(out/'last_checkpoint.pt', weights_only=False)
    def no_refit(*a, **k):
        raise AssertionError('PCA must not be fitted on resume')
    monkeypatch.setattr(training, 'make_pca', no_refit)
    monkeypatch.setattr(sys, 'argv', argv + ['--epochs', '2', '--resume', str(out/'last_checkpoint.pt')])
    training.main()
    after = torch.load(out/'last_checkpoint.pt', weights_only=False)
    assert after['epoch'] == 1
    for key in ('pca.mean', 'pca.components'):
        assert torch.equal(before['model_state_dict'][key], after['model_state_dict'][key])
    monkeypatch.setattr(sys, 'argv', ['infer', '--mace-model', 'fake', '--checkpoint',
        str(out/'last_checkpoint.pt'), '--test-xyz', 'fake', '--device', 'cpu',
        '--output-dir', str(tmp_path/'infer')])
    inference.main()
    assert (tmp_path/'infer'/'predictions_structure_regression.csv').exists()


def test_regression_legacy_checkpoint_and_artifact(tmp_path):
    from probe.regression import ErrorRegressor
    from probe.pca import make_pca
    cfg = dict(backbone_dim=4, atom_encoder_output_dim=16, mol_attention_heads=4, dropout=0.)
    model = ErrorRegressor(**cfg).eval()
    restored = restore_model(ErrorRegressor, dict(model_config=cfg,
                             model_state_dict=model.state_dict()), 'cpu').eval()
    x, mask = torch.randn(2, 3, 4), torch.ones(2, 3, dtype=torch.bool)
    for a, b in zip(model(x, mask), restored(x, mask)):
        assert torch.allclose(a, b)
    path = tmp_path/'pca.pt'
    torch.save(dict(mean=torch.arange(4.), components=torch.eye(4)), path)
    p, _ = make_pca(None, None, 'cpu', 2, 4, pca_model=path)
    assert torch.equal(p.components, torch.eye(4)[:2])


def test_fit_ignores_padding_and_nonfinite_atoms():
    from probe.pca import make_pca
    x = torch.tensor([[[1., 2., 3.], [2., 3., 4.], [3., 4., 5.],
                       [4., 5., 6.], [float('nan'), 1., 2.], [999., 999., 999.]]])
    mask = torch.tensor([[True, True, True, True, True, False]])
    p, info = make_pca(lambda b,d: b, [(x, mask)], 'cpu', 2, 3, max_atoms=3)
    assert info['n_training_atoms'] == 3
    assert torch.allclose(p.mean, torch.tensor([2., 3., 4.]))
