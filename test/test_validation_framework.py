from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from util.validation import extract_validation_logits


def test_extract_validation_logits_supports_tensor_tuple_and_dict():
    tensor = torch.randn(1, 3, 4, 4)
    assert extract_validation_logits(tensor) is tensor
    assert extract_validation_logits((tensor, "feat")) is tensor
    assert extract_validation_logits({"out": tensor}) is tensor


def test_all_validation_wrappers_use_shared_impl_in_source():
    files = [
        "supervised.py",
        "rankmatch.py",
        "corrmatch.py",
        "fixmatch_rgcr.py",
        "fixmatch_rgcrv2.py",
        "fixmatch_rgcrv3.py",
        "fixmatch_rgcrv4.py",
        "fixmatch_rgcrv5.py",
        "fixmatch_rgcrv6.py",
        "fixmatch_rvsc.py",
        "segmind.py",
    ]

    for name in files:
        text = (REPO_ROOT / name).read_text()
        assert "from util.validation import validation_cpu as shared_validation_cpu" in text
        assert "def validation_cpu(cfg, model, valid_loader):" in text
        assert "return shared_validation_cpu(cfg, model, valid_loader)" in text
