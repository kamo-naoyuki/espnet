import dataclasses
import os
import socket
from typing import Optional

import torch
import torch.distributed


@dataclasses.dataclass
class DistributedOption:
    # Enable distributed Training
    distributed: bool = False
    # torch.distributed.Backend: "nccl", "mpi", "gloo", or "tcp"
    dist_backend: str = "nccl"
    # if init_method="env://",
    # env values of "MASTER_PORT", "MASTER_ADDR", "RANK", "WORLD_SIZE" are referred.
    dist_init_method: str = "env://"
    dist_world_size: int = -1
    dist_rank: Optional[int] = None
    local_rank: int = -1
    ngpu: int = 0
    dist_master_addr: Optional[str] = None
    dist_master_port: Optional[int] = None

    def init(self):
        if self.distributed:
            # About priority order:
            # If --dist_* is specified:
            #    Use the value of --dist_rank and overwrite it environ just in case.
            # elif environ is set:
            #    Use the value of environ and set it to self
            self.dist_rank = get_rank(self.dist_rank)
            self.dist_world_size = get_world_size(self.dist_world_size)
            self.local_rank = get_local_rank(self.local_rank)

            if self.local_rank != -1 and self.ngpu != 1:
                raise RuntimeError(f"Assuming 1GPU in this case: ngpu={self.ngpu}")

            # See:
            # https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/env.html
            os.environ.setdefault("NCCL_DEBUG", "INFO")

            if self.dist_init_method == "env://":
                self.dist_master_addr = get_master_addr(self.dist_master_addr)
                self.dist_master_port = get_master_port(self.dist_master_port)
                if (
                    self.dist_master_port is not None
                    and self.dist_master_port is not None
                ):
                    self.dist_init_method = (
                        f"tcp://{self.dist_master_addr}:{self.dist_master_port}"
                    )

            torch.distributed.init_process_group(
                backend=self.dist_backend,
                init_method=self.dist_init_method,
                world_size=self.dist_world_size,
                rank=self.dist_rank,
            )

            # About distributed model:
            # if self.local_rank != -1 and ngpu == 1
            #    => Distributed with n-Process and n-GPU
            # if self.local_rank == -1 and ngpu >= 1
            #    => Distributed with 1-Process and n-GPU
            if self.local_rank != -1:
                torch.cuda.set_device(self.local_rank)


def is_in_slurm_job() -> bool:
    return "SLURM_PROCID" in os.environ and "SLURM_NTASKS" in os.environ


def is_in_slurm_step() -> bool:
    return (
        is_in_slurm_job() and
        "SLURM_STEP_NUM_NODES" in os.environ and
        "SLURM_STEP_NUM_TASKS" in os.environ
    )


def _int_or_none(x: Optional[str]) -> Optional[int]:
    if x is None:
        return x
    return int(x)


def recommended_port():
    """Find free port using bind().

    The port is freed here. There are some interval until launching process and
    the other process can take this port in this time,
    so this process will be failed potentially.

    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def get_rank(prior=None, slurm_aware: bool = True) -> Optional[int]:
    if torch.distributed.is_initialized():
        prior = torch.distributed.get_rank()
    elif slurm_aware and is_in_slurm_step():
        prior = int(os.environ["SLURM_PROCID"])

    if prior is not None:
        os.environ["RANK"] = str(prior)
        return int(prior)
    else:
        return _int_or_none(os.environ.get("RANK"))


def get_world_size(prior=None, slurm_aware: bool = True) -> Optional[int]:
    if torch.distributed.is_initialized():
        prior = torch.distributed.get_world_size()
    elif slurm_aware and is_in_slurm_step():
        prior = int(os.environ["SLURM_NTASKS"])

    if prior is not None:
        os.environ["WORLD_SIZE"] = str(prior)
        return int(prior)
    else:
        return _int_or_none(os.environ.get("WORLD_SIZE"))


def get_local_rank(prior=None, slurm_aware: bool = True) -> Optional[int]:
    if slurm_aware and is_in_slurm_step():
        prior = int(os.environ["SLURM_PROCID"])

    if prior is not None:
        os.environ["LOCAL_RANk"] = str(prior)
        return int(prior)
    else:
        return _int_or_none(os.environ.get("LOCAL_RANK"))


def get_master_addr(prior=None, slurm_aware: bool = True) -> Optional[str]:
    if slurm_aware and is_in_slurm_step():
        # e.g nodelist = foo[1-10],bar[3-8] or foo4,bar[2-10] or foo[2,4-6],bar[2-10]
        nodelist = os.environ["SLURM_STEP_NODELIST"]
        prior = nodelist.split(",")[0].split("-").replace("[", "")
    if prior is not None:
        os.environ["MASTER_ADDR"] = str(prior)
        return str(prior)
    else:
        return os.environ["MASTER_ADDR"]


def get_master_port(prior=None) -> Optional[int]:
    if prior is not None:
        os.environ["MASTER_PORT"] = str(prior)
        return prior
    else:
        return _int_or_none(os.environ.get("MASTER_PORT"))


def get_node_rank(prior=None, slurm_aware: bool = True) -> Optional[int]:
    if prior is not None:
        return prior
    if slurm_aware and is_in_slurm_step():
        return int(os.environ["SLURM_NODEID"])
    # Use "RANK" as node_rank
    return _int_or_none(os.environ.get("RANK"))


def get_num_nodes(prior=None, slurm_aware: bool = True) -> Optional[int]:
    if prior is not None:
        return prior
    if slurm_aware and is_in_slurm_step():
        return int(os.environ["SLURM_STEP_NUM_NODES"])
    # Use "WORLD_SIZE" as num-nodes
    return _int_or_none(os.environ.get("WORLD_SIZE"))
