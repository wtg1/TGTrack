"""SUTrack-style entry point for TGTrack RGB-T evaluation."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tracking"))
from test_multimodal import main

if __name__ == "__main__":
    main(datasets=["LasHeR", "RGBT234"])
