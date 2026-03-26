from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_training_entrypoints_load_backbone_checkpoints_on_cpu():
    entrypoints = [
        'supervised.py',
        'fixmatch.py',
        'fixmatch_pascal.py',
        'fixmatch_peft.py',
        'fixmatch_rgcr.py',
        'fixmatch_rgcrv2.py',
        'fixmatch_rgcrv3.py',
        'fixmatch_rgcrv4.py',
        'fixmatch_rgcrv5.py',
        'fixmatch_rgcrv6.py',
        'fixmatch_rvsc.py',
        'unimatch_v2.py',
        'unimatch_v2_rgcr.py',
        'unimatchv2_peft.py',
        'corrmatch.py',
        'rankmatch.py',
        'dwl.py',
        'scalematch.py',
        'scalematch_peft.py',
    ]

    helper_text = (REPO_ROOT / 'util/ssl_method_utils.py').read_text()
    assert 'map_location="cpu"' in helper_text, (
        'util/ssl_method_utils.py should load checkpoints on CPU before moving to GPU.'
    )

    for name in entrypoints:
        text = (REPO_ROOT / name).read_text()
        assert (
            'load_backbone_checkpoint' in text
            or 'load_checkpoint_on_cpu' in text
            or 'build_model(' in text
            or 'maybe_load_checkpoint(' in text
        ), (
            f'{name} should rely on the shared CPU-safe checkpoint loading helpers.'
        )
