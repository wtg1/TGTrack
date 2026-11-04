"""TraX entry point; launch this command through an initialized VOT workspace."""
import argparse
import _init_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="tgtrack_b256")
    parser.add_argument("--dataset", choices=["depthtrack", "votrgbd22"], default="depthtrack")
    args = parser.parse_args()
    from lib.test.vot.tgtrack import run_vot_exp
    run_vot_exp("tgtrack", args.config, out_conf=True, channel_type="rgbd", dataset=args.dataset)


if __name__ == "__main__":
    main()
