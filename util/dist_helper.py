import os
import subprocess
from datetime import timedelta

import torch
import torch.distributed as dist


def setup_distributed(backend="nccl", port=None, timeout_min=None):
    """AdaHessian Optimizer
    Lifted from https://github.com/BIGBALLON/distribuuuu/blob/master/distribuuuu/utils.py
    Originally licensed MIT, Copyright (c) 2020 Wei Li
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "10685"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    local_rank = int(os.environ.get("LOCAL_RANK", rank % num_gpus))
    torch.cuda.set_device(local_rank)

    if os.environ.get("SEMIFT_DDP_DEBUG_INIT", "0") == "1":
        print(
            f"[ddp-init] stage=setup_distributed rank={rank} local_rank={local_rank} current_device={torch.cuda.current_device()}",
            flush=True,
        )

    if timeout_min is None:
        timeout_min = float(os.environ.get("SEMIFT_DDP_TIMEOUT_MIN", "30"))

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=float(timeout_min)),
    )
    return rank, world_size
