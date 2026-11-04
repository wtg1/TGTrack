"""Launch single-GPU or torchrun distributed TGTrack training."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", default="tgtrack", choices=["tgtrack"])
    parser.add_argument("--config", default="tgtrack_b256")
    parser.add_argument("--save_dir", default=".")
    parser.add_argument("--mode", choices=["single", "multiple"], default="single")
    parser.add_argument("--nproc_per_node", type=int, default=1)
    parser.add_argument("--use_lmdb", type=int, choices=[0, 1], default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable]
    if args.mode == "multiple":
        command += ["-m", "torch.distributed.run", "--standalone",
                    f"--nproc_per_node={args.nproc_per_node}"]
    command += [str(root / "lib/train/run_training.py"), "--script", args.script,
                "--config", args.config, "--save_dir", str(Path(args.save_dir).resolve()),
                "--use_lmdb", str(args.use_lmdb), "--seed", str(args.seed)]
    subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
