# Anatomy of the Modality Gap in Self-Supervised Visual Sound Source Localization

Code for the master's thesis by **Alejandro Donaire Salvador** (September 2026).

## SSL-TIE baseline and modifications

- [main.py](main.py): training and evaluation, contrastive losses, ATP/CU losses, temperature scheduling, checkpoint selection, and embedding export.
- [models/model.py](models/model.py) and [networks/](networks/): audio-visual model with ResNet, DINOv3, Whisper, and BEATs backbones, optional Transformer adapters, and configurable embedding dimensions.
- [models/lora.py](models/lora.py) and [models/modality_adversary.py](models/modality_adversary.py): DINO LoRA adaptation and modality classification with gradient reversal.
- [datasets/dataloader.py](datasets/dataloader.py), [utils/tf_equivariance_loss.py](utils/tf_equivariance_loss.py), and [opts.py](opts.py): dataset loading, augmentations, transformation equivariance, and experiment options.
- [analysis/](analysis/): modality-gap metrics (including NNO@k), retrieval and clustering analysis, and embedding visualizations.

The baseline configuration is [bash_scripts/resnet_resnet/](bash_scripts/resnet_resnet/), using two ResNet-18 backbones and 10k training samples.

## Dataset metadata

The files in [metadata/](metadata/) preserve the sample lists used in the experiments. The 1k–40k subsets are **VGGSound training sets**; validation also uses VGGSound.

| Split | Metadata file | Selection option |
| --- | --- | --- |
| Train: 1,000 | [train_vggs_1k.txt](metadata/train_vggs_1k.txt) | `--train_set_scale subset_1k` |
| Train: 5,000 | [train_vggs_5k.txt](metadata/train_vggs_5k.txt) | `--train_set_scale subset_5k` |
| Train: 10,000 | [train_vggs_10k.txt](metadata/train_vggs_10k.txt) | `--train_set_scale subset_10k` |
| Train: 20,000 | [train_vggs_20k.txt](metadata/train_vggs_20k.txt) | `--train_set_scale subset_20k` |
| Train: 40,000 | [train_vggs_40k.txt](metadata/train_vggs_40k.txt) | `--train_set_scale subset_40k` |
| Validation: 1,000 | [val_vggs_1k.txt](metadata/val_vggs_1k.txt) | `--val_set VGGS --val_set_scale subset_1k` |

Download and prepare the media separately. Under `--trainset_path`, training and validation expect `total_video_3s_audio/<id>.wav` and `total_video_frames/<id>/125.jpg`. Under `--vggss_test_path`, testing expects `audio/<id>.wav`, `frame/<id>.jpg`, and `anno/<id>.xml`. The loader skips samples with missing files, so the available dataset size may be smaller than the metadata list.

## Synthetic scenarios

[synthetic_scenarios/](synthetic_scenarios/) contains the six controlled embedding scenarios for comparing modality-gap metrics:

- [converging_clouds.py](synthetic_scenarios/converging_clouds.py): move two modality clouds toward overlap.
- [distributional_spread.py](synthetic_scenarios/distributional_spread.py): increase distribution spread around fixed poles.
- [pair_alignment.py](synthetic_scenarios/pair_alignment.py): vary pair alignment while keeping modality distributions fixed.
- [converging_rings.py](synthetic_scenarios/converging_rings.py): move modality rings toward one another.
- [interleaved_spirals.py](synthetic_scenarios/interleaved_spirals.py): vary the spread of interleaved spirals.
- [hybrid_bands.py](synthetic_scenarios/hybrid_bands.py): progressively overlap modality bands.

Each scenario generates plots and a metrics CSV, with `--seed`, `--output-dir`, and `--dpi` options. [plot_synthetic_scenarios_grid.py](synthetic_scenarios/plot_synthetic_scenarios_grid.py) assembles the six scenarios into one figure.

## Running the 37 experiments

[bash_scripts/](bash_scripts/) contains **37 experiment directories**, each with:

- `train.sh`: train the configuration.
- `test_VGGSS.sh`: evaluate on VGG-SS.
- `test_IS3plus.sh`: evaluate on IS3plus.
- `test_AVSBench.sh`: evaluate on AVSBench.

These are Slurm scripts. Before submitting, update the environment, repository, dataset, output, and checkpoint paths, plus any cluster-specific settings. DINO/Whisper scripts use an offline Hugging Face cache; BEATs requires its checkpoint and the `third_party/unilm` submodule. Python dependencies are listed in [requirements.txt](requirements.txt).

From the repository root, submit the baseline training job:

```bash
sbatch bash_scripts/resnet_resnet/train.sh
```

To submit training jobs for all 37 configurations:

```bash
for experiment in bash_scripts/*/; do
  sbatch "${experiment}train.sh"
done
```
