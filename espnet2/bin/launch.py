r"""

09, Jan, 2019: Copied from
https://github.com/pytorch/pytorch/blob/e7fe64f6a65cd427e503491f192c14476e18033b/torch/distributed/launch.py

"""
import socket
import sys
import subprocess
import os
from argparse import ArgumentParser, REMAINDER

import torch

from espnet2.utils.types import int_or_none
from espnet2.utils.types import str2bool
from espnet2.utils.types import str_or_none


def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(
        description="PyTorch distributed training launch "
        "helper utilty that will spawn up "
        "multiple distributed processes"
    )

    # Optional arguments for the launch helper
    parser.add_argument(
        "--nnodes",
        type=int,
        default=1,
        help="The number of nodes to use for distributed " "training",
    )
    parser.add_argument(
        "--node_rank",
        type=int,
        default=0,
        help="The rank of the node for multi-node distributed " "training",
    )
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=[1],
        nargs="*",
        help="The number of processes to launch on each node, "
        "for GPU training, this is recommended to be set "
        "to the number of GPUs in your system so that "
        "each process can be bound to a single GPU.",
    )
    parser.add_argument(
        "--master_addr",
        default="127.0.0.1",
        type=str_or_none,
        help="Master node (rank 0)'s address, should be either "
        "the IP address or the hostname of node 0, for "
        "single node multi-proc training, the "
        "--master_addr can simply be 127.0.0.1",
    )
    parser.add_argument(
        "--master_port",
        default=None,
        type=int_or_none,
        help="Master node (rank 0)'s free port that needs to "
        "be used for communciation during distributed "
        "training",
    )
    parser.add_argument(
        "--use_env",
        default=False,
        action="store_true",
        help="Use environment variable to pass "
        "'local rank'. For legacy reasons, the default value is False. "
        "If set to True, the script will not pass "
        "--local_rank as argument, and will instead set LOCAL_RANK.",
    )
    parser.add_argument(
        "-m",
        "--module",
        type=str2bool,
        default=False,
        help="Changes each process to interpret the launch script "
        "as a python module, executing with the same behavior as"
        "'python -m'.",
    )
    parser.add_argument(
        "--no_python",
        default=False,
        action="store_true",
        help='Do not prepend the training script with "python" - just exec '
        "it directly. Useful when the script is not a Python script.",
    )

    # positional
    parser.add_argument(
        "training_script",
        type=str,
        help="The full path to the single GPU training "
        "program/script to be launched in parallel, "
        "followed by all the arguments for the "
        "training script",
    )

    # rest from the training program
    parser.add_argument("training_script_args", nargs=REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()

    # Slurm supporting:
    if "SLURM_STEP_NODELIST" in os.environ:
        # # 1. Set --nproc_per_node
        # nproc_per_node = []
        # # e.g. SLURM_TASKS_PER_NODE = 3,1(x4) => (3, 1, 1, 1)
        # for x in os.environ["SLURM_TASKS_PER_NODE"].split(","):
        #     try:
        #         nproc_per_node.append(int(x))
        #     except ValueError:
        #         # e.g. 2(x4)
        #         n, m = x.replace("(", "").replace(")", "").split("x")
        #         nproc_per_node.extend([int(n) for _ in range(int(m))])
        # args.nproc_per_node = nproc_per_node

        # 2. Set --master_addr
        # e.g. node_list foo[1-10],bar[2-10]
        # -> master_addr = foo1
        node_list = os.environ["SLURM_STEP_NODELIST"]
        hostnames = subprocess.check_output(
            ["scontrol", "show", "hostnames", node_list]
        )
        args.master_addr = hostnames.split()[0].decode("utf-8")

        # 3. Set --node_rank
        args.node_rank = int(os.environ["SLURM_NODEID"])
        # 3. Set --nnodes
        args.nnodes = int(os.environ["SLURM_STEP_NUM_NODES"])

    # If distributed on single node:
    if (
        args.nnodes == 1
        and args.master_port is None
        and "MASTER_PORT" not in os.environ
    ):
        # then, the master host is always this machine, find free port using bind()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            args.master_port = sock.getsockname()[1]
        # The port is freed here. There are some interval until launching process and
        # the other process can take this port in this time,
        # so this process will be failed potentially.

    # If single number is speficied: e.g. --nproc_per_node 1
    # -> Use same number of procs for each nodes.
    if len(args.nproc_per_node) == 1:
        args.nproc_per_node = [args.nproc_per_node[0] for _ in range(args.nnodes)]

    if len(args.nproc_per_node) <= args.node_rank:
        raise RuntimeError(
            f"Invalid node_rank: {len(args.nproc_per_node)} <= {args.node_rank}"
        )

    # world size in terms of number of processes
    dist_world_size = sum(args.nproc_per_node)

    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    current_env.setdefault("NCCL_DEBUG", "INFO")

    current_env["WORLD_SIZE"] = str(dist_world_size)
    if args.master_addr is not None:
        current_env["MASTER_ADDR"] = args.master_addr
    if args.master_port is not None:
        current_env["MASTER_PORT"] = str(args.master_port)

    processes = []

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        devices = os.environ["CUDA_VISIBLE_DEVICES"]
        num_devices = len(devices.split(","))
    else:
        num_devices = torch.cuda.device_count()
    nprocs = args.nproc_per_node[args.node_rank]
    if num_devices != nprocs:
        raise RuntimeError(f"len(devices) != nprocs: {num_devices} != {nprocs}")

    if "OMP_NUM_THREADS" not in os.environ and nprocs > 1:
        current_env["OMP_NUM_THREADS"] = str(1)
        print(
            "*****************************************\n"
            "Setting OMP_NUM_THREADS environment variable for each process "
            "to be {} in default, to avoid your system being overloaded, "
            "please further tune the variable for optimal performance in "
            "your application as needed. \n"
            "*****************************************".format(
                current_env["OMP_NUM_THREADS"]
            )
        )

    for local_rank in range(0, nprocs):
        # each process's rank
        dist_rank = sum(args.nproc_per_node[: args.node_rank]) + local_rank
        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)

        # spawn the processes
        with_python = not args.no_python
        cmd = []
        if with_python:
            cmd = [sys.executable, "-u"]
            if args.module:
                cmd.append("-m")
        else:
            if not args.use_env:
                raise ValueError(
                    "When using the '--no_python' flag, "
                    "you must also set the '--use_env' flag."
                )
            if args.module:
                raise ValueError(
                    "Don't use both the '--no_python' flag "
                    "and the '--module' flag at the same time."
                )

        cmd.append(args.training_script)

        if not args.use_env:
            cmd.append("--local_rank={}".format(local_rank))

        cmd.extend(args.training_script_args)

        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)

    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)


if __name__ == "__main__":
    main()
