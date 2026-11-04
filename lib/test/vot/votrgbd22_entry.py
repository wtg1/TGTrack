"""VOT-RGBD22 entry point for the VOT tracker registry."""
import os
from lib.test.vot.tgtrack import run_vot_exp


if __name__ == "__main__":
    run_vot_exp("tgtrack", os.environ.get("TGTRACK_CONFIG", "tgtrack_b256"),
                out_conf=True, channel_type="rgbd", dataset="votrgbd22")
