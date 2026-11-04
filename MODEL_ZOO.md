# TGTrack Model Zoo

Trained models, [training logs](https://huggingface.co/wwwttt1127/TGTrack/tree/main/logs) and [pretrained weights](https://huggingface.co/wwwttt1127/TGTrack/tree/main/pretrained) are available on [Hugging Face](https://huggingface.co/wwwttt1127/TGTrack). Model results below are the values published in the CVPR 2026 paper.

## Models

| Model | Config | Checkpoint directory | Size (GiB) | Weights | Training log |
| --- | --- | --- | ---: | --- | --- |
| TGTrack-T224 | [YAML](experiments/tgtrack/tgtrack_t224.yaml) | `train/tgtrack/tgtrack_t224/` | 1.407 | [Download](https://huggingface.co/wwwttt1127/TGTrack/resolve/main/checkpoints/train/tgtrack/tgtrack_t224/TGTrack_ep0180.pth.tar?download=true) | — |
| TGTrack-S224 | [YAML](experiments/tgtrack/tgtrack_s224.yaml) | `train/tgtrack/tgtrack_s224/` | 1.582 | [Download](https://huggingface.co/wwwttt1127/TGTrack/resolve/main/checkpoints/train/tgtrack/tgtrack_s224/TGTrack_ep0180.pth.tar?download=true) | — |
| TGTrack-B256 | [YAML](experiments/tgtrack/tgtrack_b256.yaml) | `train/tgtrack/tgtrack_b256/` | 2.266 | [Download](https://huggingface.co/wwwttt1127/TGTrack/resolve/main/checkpoints/train/tgtrack/tgtrack_b256/TGTrack_ep0180.pth.tar?download=true) | [Log](https://huggingface.co/wwwttt1127/TGTrack/blob/main/logs/gentem_s1-gentem_s1_b256.log) |
| TGTrack-B384 | [YAML](experiments/tgtrack/tgtrack_b384.yaml) | `train/tgtrack/tgtrack_b384/` | 2.269 | [Download](https://huggingface.co/wwwttt1127/TGTrack/resolve/main/checkpoints/train/tgtrack/tgtrack_b384/TGTrack_ep0180.pth.tar?download=true) | [Log](https://huggingface.co/wwwttt1127/TGTrack/blob/main/logs/gentem_s1-gentem_s1_b384.log) |
| TGTrack-L256 | [YAML](experiments/tgtrack/tgtrack_l256.yaml) | `train/tgtrack/tgtrack_l256/` | 5.314 | [Download](https://huggingface.co/wwwttt1127/TGTrack/resolve/main/checkpoints/train/tgtrack/tgtrack_l256/TGTrack_ep0180.pth.tar?download=true) | [Log](https://huggingface.co/wwwttt1127/TGTrack/blob/main/logs/gentem_s1-gentem_s1_l256.log) |
| TGTrack-L384 | [YAML](experiments/tgtrack/tgtrack_l384.yaml) | `train/tgtrack/tgtrack_l384/` | 5.318 | [Download](https://huggingface.co/wwwttt1127/TGTrack/resolve/main/checkpoints/train/tgtrack/tgtrack_l384/TGTrack_ep0180.pth.tar?download=true) | [Log](https://huggingface.co/wwwttt1127/TGTrack/blob/main/logs/gentem_s1-gentem_s1_l384.log) |

Directories in the table are relative to `checkpoints/`. Each directory contains `TGTrack_ep0180.pth.tar`, a full training snapshot with all network tensors, optimizer state and training settings.

Models are loaded with `strict=True`.

## Published RGB and language results

See Tables 2 and 4 of the [CVPR 2026 paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Geng_TGTrack_Temporal_Generative_Learning_for_Unified_Single_Object_Tracking_CVPR_2026_paper.pdf).

| Model | LaSOT AUC | LaSOText AUC | GOT-10K AO | TrackingNet AUC | TNL2K AUC | OTB99 AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TGTrack-T224 | 71.3 | 51.2 | 76.1 | 84.3 | 63.0 | 69.8 |
| TGTrack-S224 | 72.9 | 52.4 | 77.2 | 85.3 | 64.7 | 69.9 |
| TGTrack-B256 | 74.6 | 53.2 | 80.8 | 86.2 | 65.4 | 72.5 |
| TGTrack-B384 | 75.3 | 54.8 | 79.8 | 87.5 | 66.7 | 70.9 |
| TGTrack-L256 | 75.8 | 54.8 | 81.4 | 87.5 | 67.7 | 72.5 |
| TGTrack-L384 | 76.4 | 55.9 | 81.8 | 88.0 | 68.6 | 72.9 |

## Published multi-modal results

Paper Table 3, percentages. DepthTrack uses F-score; RGBT234 uses MSR; the remaining columns use AUC.

| Model | DepthTrack | LasHeR | RGBT234 | VisEvent |
| --- | ---: | ---: | ---: | ---: |
| TGTrack-T224 | 63.3 | 58.2 | 67.0 | 62.2 |
| TGTrack-S224 | 63.4 | 59.4 | 68.5 | 63.1 |
| TGTrack-B256 | 65.5 | 61.7 | 69.9 | 63.9 |
| TGTrack-B384 | 66.0 | 62.8 | 69.5 | 64.7 |
| TGTrack-L256 | 66.8 | 62.7 | 71.1 | 65.1 |
| TGTrack-L384 | 67.5 | 63.3 | 71.3 | 65.7 |

