from pathlib import Path
import ast

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


DIRECT_ENTRYPOINTS = [
    'corrmatch.py',
    'supervised.py',
    'fixmatch.py',
    'dwl.py',
    'fixmatch_pascal.py',
    'fixmatch_peft.py',
    'fixmatch_rgcr.py',
    'fixmatch_rgcrv2.py',
    'fixmatch_rgcrv3.py',
    'fixmatch_rgcrv4.py',
    'fixmatch_rgcrv5.py',
    'fixmatch_rgcrv6.py',
    'fixmatch_rvsc.py',
    'rankmatch.py',
    'unimatch_v2.py',
    'unimatch_v2_rgcr.py',
    'unimatchv2_peft.py',
    'scalematch.py',
    'scalematch_peft.py',
]

HELPER_ENTRYPOINTS = ['wscl.py', 'segmind.py']


def load_checkpoint_to_cpu_function():
    source = (REPO_ROOT / 'util/ssl_method_utils.py').read_text()
    module = ast.parse(source)
    fn_node = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'checkpoint_to_cpu'
    )
    isolated_module = ast.Module(body=[fn_node], type_ignores=[])
    code = compile(isolated_module, filename='checkpoint_to_cpu', mode='exec')
    namespace = {'torch': torch}
    exec(code, namespace)
    return namespace['checkpoint_to_cpu']


def test_checkpoint_to_cpu_converts_nested_tensors_to_cpu():
    checkpoint_to_cpu = load_checkpoint_to_cpu_function()

    nested = {
        'model': {'w': torch.randn(2, 3)},
        'optimizer': {
            'state': {0: {'exp_avg': torch.randn(3), 'exp_avg_sq': torch.randn(3)}},
            'param_groups': [{'lr': 1e-3}],
        },
        'items': [torch.randn(1), ('x', torch.randn(2))],
        'epoch': 3,
    }

    out = checkpoint_to_cpu(nested)

    assert out['epoch'] == 3
    assert out['model']['w'].device.type == 'cpu'
    assert out['optimizer']['state'][0]['exp_avg'].device.type == 'cpu'
    assert out['optimizer']['state'][0]['exp_avg_sq'].device.type == 'cpu'
    assert out['items'][0].device.type == 'cpu'
    assert out['items'][1][1].device.type == 'cpu'


def test_direct_entrypoints_use_cpu_resume_and_cpu_safe_save_helpers():
    for name in DIRECT_ENTRYPOINTS:
        text = (REPO_ROOT / name).read_text()
        assert 'load_checkpoint_on_cpu' in text, f'{name} should resume checkpoints on CPU.'
        assert 'save_checkpoint_to_disk' in text, f'{name} should save checkpoints after moving tensors to CPU.'
        assert 'model.cuda(local_rank)' in text, f'{name} should bind model initialization to local_rank.'


def test_helper_entrypoints_pass_logger_and_rank_into_shared_ddp_helpers():
    for name in HELPER_ENTRYPOINTS:
        text = (REPO_ROOT / name).read_text()
        assert 'wrap_ddp(model, logger=logger, rank=rank, save_path=args.save_path)' in text
        assert 'maybe_load_checkpoint(' in text and 'logger=logger, rank=rank' in text
