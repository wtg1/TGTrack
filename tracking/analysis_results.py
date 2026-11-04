"""Evaluate saved TGTrack results using benchmark annotations."""
import argparse
import _init_paths
from lib.test.evaluation import get_dataset, trackerlist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="tgtrack_b256")
    parser.add_argument("--dataset", default="lasot")
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Analyze a matching short execution check")
    args = parser.parse_args()
    from lib.test.analysis.plot_results import print_results
    trackers = trackerlist("tgtrack", args.config, args.dataset, display_name="TGTrack")
    dataset = get_dataset(args.dataset)
    if args.sequence is not None:
        index = int(args.sequence) if args.sequence.isdigit() else args.sequence
        dataset = [dataset[index]]
    report = f"{args.config}/{args.dataset}"
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
        report += f"/smoke_{args.max_frames}frames"
    print_results(trackers, dataset, report,
                  merge_results=True, plot_types=("success", "prec", "norm_prec"))


if __name__ == "__main__":
    main()
