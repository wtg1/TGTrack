# VOT-RGBD22 evaluation

This workspace uses the official `vot2022/rgbd` stack. Place the 127 official VOT-RGBD22 sequences and `list.txt` under `sequences/`. The local prepared workspace links the existing dataset; images and historical results are excluded from the source release.

Activate the tracking environment and install `requirements-eval.txt`, then run from this directory:

```bash
vot evaluate tgtrack_b256
vot analysis tgtrack_b256 --nocache
```

Or, from the repository root:

```bash
bash VOT22RGBD_workspace/run.sh tgtrack_b256
```

`trackers.ini` registers T224, S224, B256, B384, L256 and L384. Change the tracker identifier to select a model. Relative project paths make the registry portable; it uses the Python interpreter running `vot`.

The entry module is `lib.test.vot.votrgbd22_entry`. Settings come from `VOTRGBD22` in the selected model YAML. Results, analysis and logs stay inside this workspace.
