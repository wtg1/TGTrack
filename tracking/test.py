"""Run TGTrack on a benchmark or one selected sequence."""
import argparse
import _init_paths
from lib.test.evaluation import get_dataset, Tracker
from lib.test.evaluation.running import run_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracker_name", choices=["tgtrack"])
    parser.add_argument("tracker_param")
    parser.add_argument("--dataset", default="lasot")
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Limit a smoke run; results go to a separate directory")
    args = parser.parse_args()
    dataset = get_dataset(args.dataset)
    if args.sequence is not None:
        sequence = int(args.sequence) if args.sequence.isdigit() else args.sequence
        dataset = [dataset[sequence]]
    trackers = [Tracker(args.tracker_name, args.tracker_param, args.dataset)]
    if args.max_frames:
        from copy import deepcopy
        from pathlib import Path
        dataset = deepcopy(dataset)
        for sequence in dataset:
            sequence.frames = sequence.frames[:args.max_frames]
            sequence.ground_truth_rect = sequence.ground_truth_rect[:args.max_frames]
            if sequence.target_visible is not None:
                sequence.target_visible = sequence.target_visible[:args.max_frames]
        for tracker in trackers:
            tracker.results_dir = str(Path(tracker.results_dir) / f"smoke_{args.max_frames}frames")
    run_dataset(dataset, trackers, args.debug, args.threads, num_gpus=args.num_gpus)


if __name__ == "__main__":
    main()
