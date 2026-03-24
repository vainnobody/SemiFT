from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_training_entrypoints_load_backbone_checkpoints_on_cpu():
    files = [
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
        'unimatch.py',
        'unimatch_v2.py',
        'unimatch_v2_rgcr.py',
        'unimatchv2_peft.py',
        'corrmatch.py',
        'rankmatch.py',
        'dwl.py',
        'scalematch.py',
        'scalematch_peft.py',
        'segmind.py',
        'util/ssl_method_utils.py',
    ]

    for name in files:
        text = (REPO_ROOT / name).read_text()
        assert 'map_location="cpu"' in text, (
            f'{name} should load backbone checkpoints on CPU before moving to GPU.'
        )
