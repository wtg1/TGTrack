# DepthTrack evaluation

This workspace uses the VOT long-term RGB-D experiment configuration `votrgbd2021`. Place the 50 official DepthTrack test sequences and `list.txt` under `sequences/`. The local prepared workspace links the existing dataset; dataset files are excluded from the source release.

Activate the tracking environment and install `requirements-eval.txt`, then run from this directory:

```bash
vot evaluate tgtrack_b256
vot analysis tgtrack_b256 --nocache
```

Or, from the repository root:

```bash
bash Depthtrack_workspace/run.sh tgtrack_b256
```

`trackers.ini` registers T224, S224, B256, B384, L256 and L384. Change the tracker identifier to select a model. The registry uses relative project paths and the Python interpreter running `vot`; activate the intended environment before evaluation.

The entry module is `lib.test.vot.depthtrack_entry`. Its update settings come from `DEPTHTRACK` in the selected model YAML. Results, analysis and logs stay inside this workspace.
