# Set-up

## Dependencies

This codebase has been tested with the packages specified in `requirements.txt`.

```bash
torch
torchvision
timm
einops
pandas
albumentations
wandb
```

## Dataset Preparation


### Dataset structure

For simplicity and uniformity, all our datasets are structured in the following way:
```
/path/to/data/
├── train/
│   ├── modality1/
│   │   └── subfolder1/
│   │       ├── img1.ext1
│   │       └── img2.ext1
│   └── modality2/
│       └── subfolder1/
│           ├── img1.ext2
│           └── img2.ext2
└── val/
    ├── modality1/
    │   └── subfolder2/
    │       ├── img3.ext1
    │       └── img4.ext1
    └── modality2/
        └── subfolder2/
            ├── img3.ext2
            └── img4.ext2
```
The folder structure and filenames should match across modalities.
If a dataset does not have specific subfolders, a generic subfolder name can be used instead (e.g., `all/`). 

For most experiments, we use RGB  (`rgb`), depth (`depth`), and semantic segmentation (`semseg`) as our modalities.

RGB images are stored as either PNG or JPEG images. 
Depth maps are stored as either single-channel JPX or single-channel PNG images. 
Semantic segmentation maps are stored as single-channel PNG images.

### Datasets

We use the following datasets in our experiments:
- [**ImageNet-1K**](https://www.image-net.org/)
- [**ADE20K**](http://sceneparsing.csail.mit.edu/)
- [**Hypersim**](https://github.com/apple/ml-hypersim)
- [**NYUv2**](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
- [**Taskonomy**](https://github.com/StanfordVL/taskonomy/tree/master/data)

To download these datasets, please follow the instructions on their respective pages. 
To prepare the NYUv2 dataset, we recommend using the provided [`prepare_nyuv2.py`](tools/prepare_nyuv2.py) script.

