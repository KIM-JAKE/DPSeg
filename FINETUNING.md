# Fine-tuning

We provide fine-tuning scripts for classification, semantic segmentation, depth estimation and more.
Please check [SETUP.md](SETUP.md) for set-up instructions first.

- [General information](#general-information)
- [Semantic segmentation](#semantic-segmentation)

## General information

### Loading pre-trained models

All our fine-tuning scripts support models in the MultiMAE / MultiViT format. Pre-trained models using the timm / ViT format can be converted to this format using the [`vit2multimae_converter.py`](tools/vit2multimae_converter.py)
 script. More information can be found [here](README.md#model-formats).

### Modifying configs
The training scripts support both YAML config files and command-line arguments. See [here](cfgs/finetune) for all fine-tuning config files.

To modify fine-training settings, either edit / add config files or provide additional command-line arguments.

:information_source: Config files arguments override default arguments, and command-line arguments override both default arguments and config arguments.

:warning: When changing settings (e.g., using a different pre-trained model), make sure to modify the `output_dir` and `wandb_run_name` (if logging is activated) to reflect the changes.


### Experiment logging
To activate logging to [Weights & Biases](https://docs.wandb.ai/), either edit the config files or use the `--log_wandb` flag along with any other extra logging arguments.

## Semantic segmentation

### ADE20K
To fine-tune MultiMAE on ADE20K semantic segmentation with default settings and **RGB** as the input modality, run:
```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 run_finetuning_semseg.py \
--config cfgs/finetune/semseg/ade/ft_ade_64e_multimae-b_rgb.yaml \
--finetune /path/to/multimae_weights \
--data_path /path/to/ade20k/train \
--eval_data_path /path/to/ade20k/val
```

- For a list of possible arguments, see [`run_finetuning_semseg.py`](run_finetuning_semseg.py).


### NYUv2
To fine-tune MultiMAE on NYUv2 semantic segmentation with default settings and **RGB** as the input modality, run:
```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 run_finetuning_semseg.py \
--config /cfgs/finetune/DPSeg/nyu/DPSeg.yaml \
--finetune /path/to/multimae_weights \
--data_path /path/to/nyu/train \
--eval_data_path /path/to/nyu/test_or_val
```

- To fine-tune using **depth-only** and **RGB + depth** as the input modalities, simply swap the config file to the appropriate one.
- For a list of possible arguments, see [`run_finetuning_semseg.py`](run_finetuning_semseg.py).
