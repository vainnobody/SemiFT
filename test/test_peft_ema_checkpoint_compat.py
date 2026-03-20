from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

import fixmatch_peft
import unimatchv2_peft


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)


def _state_with_prefix(model, prefix):
    return {f"{prefix}{k}": v.clone() for k, v in model.state_dict().items()}


def test_fixmatch_peft_flexible_loader_accepts_plain_and_module_prefix():
    model = Tiny()
    plain = model.state_dict()
    prefixed = _state_with_prefix(model, "module.")

    fixmatch_peft._load_state_dict_flexible(model, plain)
    fixmatch_peft._load_state_dict_flexible(model, prefixed)


def test_unimatchv2_peft_flexible_loader_accepts_plain_and_module_prefix():
    model = Tiny()
    plain = model.state_dict()
    prefixed = _state_with_prefix(model, "module.")

    unimatchv2_peft._load_state_dict_flexible(model, plain)
    unimatchv2_peft._load_state_dict_flexible(model, prefixed)
