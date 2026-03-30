from pathlib import Path
import sys
import types

stub_tb = types.ModuleType("torch.utils.tensorboard")


class StubSummaryWriter:
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass


stub_tb.SummaryWriter = StubSummaryWriter
sys.modules.setdefault("torch.utils.tensorboard", stub_tb)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scalematch_peft
import unimatchv2_peft


def test_scalematch_peft_reuses_shared_peft_helpers():
    assert scalematch_peft.resolve_peft_cfg is unimatchv2_peft.resolve_peft_cfg
    assert scalematch_peft.apply_peft is unimatchv2_peft.apply_peft
    assert (
        scalematch_peft.show_trainable_parameters
        is unimatchv2_peft.show_trainable_parameters
    )


def test_scalematch_peft_uses_scalematch_recipe_and_peft_logging():
    source = (REPO_ROOT / "scalematch_peft.py").read_text()

    assert "loader = zip(trainloader_l, trainloader_u, trainloader_u_mix)" in source
    assert 'pred["pred_joint"]' in source
    assert 'pred["pred_size"]' in source
    assert 'pred["pred_fp"]' in source
    assert 'if epoch < cfg["warm_up"]:' in source
    assert 'loss_u_s1 * 0.25 + loss_u_size * 0.25 + loss_u_w_fp * 0.5' in source
    assert "Running ScaleMatch + PEFT with method=%s, target_modules=%s, freeze_backbone=%s" in source
