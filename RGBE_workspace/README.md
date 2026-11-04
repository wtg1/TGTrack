# RGB-Event evaluation

From the repository root:

```bash
python RGBE_workspace/test_rgbe_mgpus.py --script_name tgtrack \
    --dataset_name VisEvent --yaml_name tgtrack_b256
```

The VisEvent test split is read from `data/visevent/test`. Results are saved in `RGBE_workspace/results/VisEvent/tgtrack_tgtrack_b256/`. Use the official VisEvent evaluation toolkit to calculate the full benchmark scores.

