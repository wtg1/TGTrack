"""Run TGTrack on LasHeR, RGBT234 or VisEvent with portable dataset paths."""
import argparse
import multiprocessing as mp
from pathlib import Path
import time
import _init_paths


def sequence_inputs(directory, dataset):
    import numpy as np
    if dataset == "VisEvent":
        rgb = sorted((directory / "vis_imgs").glob("*.bmp"))
        extra = sorted((directory / "event_imgs").glob("*.bmp"))
        boxes = np.atleast_2d(np.loadtxt(directory / "groundtruth.txt", delimiter=","))
        present = np.atleast_1d(np.loadtxt(directory / "absent_label.txt"))
        if not present.any():
            raise ValueError(f"No visible initialization frame in {directory}")
        start = int(present.argmax()) if present[0] == 0 else 0
    else:
        rgb = sorted((directory / "visible").glob("*.jpg"))
        extra = sorted((directory / "infrared").glob("*.jpg"))
        boxes = np.atleast_2d(np.loadtxt(directory / "visible.txt", delimiter=","))
        start = 0
    if not rgb or len(rgb) != len(extra) or len(rgb) != len(boxes):
        raise ValueError(f"Unaligned frame/annotation counts in {directory}")
    return rgb[start:], extra[start:], boxes[start:], start


def run_sequence(directory, dataset, config, num_gpus, max_frames=0, results_dir=None):
    import numpy as np
    import torch
    from lib.test.evaluation.environment import env_settings
    from lib.test.parameter.tgtrack import parameters
    from lib.test.tracker.tgtrack import TGTrack
    from lib.train.dataset.depth_utils import get_x_frame
    worker = mp.current_process()._identity
    torch.cuda.set_device(((worker[0] - 1) if worker else 0) % num_gpus)
    torch.set_num_threads(1)
    workspace = "RGBE_workspace" if dataset == "VisEvent" else "RGBT_workspace"
    output_root = Path(results_dir) if results_dir else Path(env_settings().prj_dir) / workspace / "results"
    output_dir = output_root / dataset / f"tgtrack_{config}"
    if max_frames:
        output_dir = output_dir / f"smoke_{max_frames}frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / (directory.name + ".txt")
    if destination.exists():
        print(f"Existing results: {destination}", flush=True)
        return
    rgb, extra, boxes, start = sequence_inputs(directory, dataset)
    limit = min(len(rgb), max_frames) if max_frames else len(rgb)
    tracker = TGTrack(parameters(config), dataset.lower())
    result = [boxes[0].tolist()]
    scores, times = [1.0], []
    for index in range(limit):
        image = get_x_frame(str(rgb[index]), str(extra[index]), dtype="rgbrgb")
        begin = time.perf_counter()
        if index == 0:
            tracker.initialize(image, {"init_bbox": result[0]})
        else:
            output = tracker.track(image)
            result.append(output["target_bbox"])
            scores.append(float(output["best_score"].max()))
        torch.cuda.synchronize()
        times.append(time.perf_counter() - begin)
    result = np.asarray(result)
    assert np.isfinite(result).all() and (result[:, 2:] > 0).all()
    np.savetxt(destination, result, delimiter=",", fmt="%.6f")
    np.savetxt(destination.with_name(directory.name + "_time.txt"), times, fmt="%.6f")
    np.savetxt(destination.with_name(directory.name + "_confidence.txt"), scores, fmt="%.6f")
    print(f"{dataset}/{directory.name}: {limit} frames, start={start}, {destination}", flush=True)


def main(datasets=None):
    import torch
    from lib.test.evaluation.environment import env_settings
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script_name", default="tgtrack", choices=["tgtrack"])
    parser.add_argument("--yaml_name", default="tgtrack_b256")
    parser.add_argument("--dataset_name", required=True, choices=datasets or ["LasHeR", "RGBT234", "VisEvent"])
    parser.add_argument("--dataset_dir", type=Path)
    parser.add_argument("--sequence")
    parser.add_argument("--results_dir", type=Path, help="Override the workspace results directory")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0, help="Smoke test only; 0 runs the full sequence")
    args = parser.parse_args()
    if args.num_gpus < 1 or args.num_gpus > torch.cuda.device_count():
        parser.error("--num_gpus must fit the visible CUDA devices")
    env = env_settings()
    roots = {"LasHeR": Path(env.lasher_path) / "testingset",
             "RGBT234": Path(env.rgbt234_path), "VisEvent": Path(env.visevent_path) / "test"}
    root = args.dataset_dir or roots[args.dataset_name]
    sequences = [root / args.sequence] if args.sequence else sorted(p for p in root.iterdir() if p.is_dir())
    tasks = [(path, args.dataset_name, args.yaml_name, args.num_gpus, args.max_frames, args.results_dir) for path in sequences]
    if args.threads:
        with mp.get_context("spawn").Pool(args.threads) as pool:
            pool.starmap(run_sequence, tasks)
    else:
        for task in tasks:
            run_sequence(*task)


if __name__ == "__main__":
    main()
