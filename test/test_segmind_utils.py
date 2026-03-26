from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from util.segmind_utils import compute_contrastive_loss, init_queue_state


def test_compute_contrastive_loss_handles_flattened_feature_shapes():
    batch_size, feat_dim, height, width = 2, 8, 4, 6
    num_classes = 3
    proj_feat = torch.randn(batch_size, feat_dim, height, width)
    labels = torch.randint(0, num_classes, (batch_size, height * 2, width * 2))
    probs = torch.softmax(torch.randn(batch_size, num_classes, height * 2, width * 2), dim=1)
    queue_state = init_queue_state(num_classes, feat_dim, bank_size=16)

    loss = compute_contrastive_loss(
        proj_feat,
        labels,
        probs,
        queue_state,
        query_threshold=1.1,
        temperature=0.5,
        num_query=2,
        num_negative=2,
        ignore_index=255,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
