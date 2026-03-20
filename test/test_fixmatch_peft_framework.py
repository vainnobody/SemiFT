from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fixmatch_peft
import unimatchv2_peft


def test_fixmatch_peft_reuses_shared_peft_helpers():
    assert fixmatch_peft.resolve_peft_cfg is unimatchv2_peft.resolve_peft_cfg
    assert fixmatch_peft.apply_peft is unimatchv2_peft.apply_peft
    assert (
        fixmatch_peft.show_trainable_parameters
        is unimatchv2_peft.show_trainable_parameters
    )
