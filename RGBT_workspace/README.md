# RGB-Thermal evaluation

From the repository root:

```bash
python RGBT_workspace/test_rgbt_mgpus.py --script_name tgtrack \
    --dataset_name LasHeR --yaml_name tgtrack_b256
python RGBT_workspace/test_rgbt_mgpus.py --script_name tgtrack \
    --dataset_name RGBT234 --yaml_name tgtrack_b256
```

The default test roots are `data/lasher/testingset` and `data/RGB-T234`. Results are saved under `RGBT_workspace/results/<dataset>/tgtrack_tgtrack_b256/`. Use the official benchmark toolkits to calculate full benchmark scores.
