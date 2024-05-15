# E-MTL: Efficient Multi-task Learning Architecture using Hybrid Transformer and ConvNet blocks

## Introduction

This repository provides a Python-based implementation of an efficient Multi-task learning architecture for dense prediction tasks. The repository is based upon [Swin-Transformer](https://github.com/microsoft/Swin-Transformer) and uses some modules from [Multi-Task-Learning-PyTorch](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch).

## Download Datasets
We use the same data (PASCAL-Context and NYUD-v2) as ATRC. You can download the data from: [PASCALContext.tar.gz](https://hkustconnect-my.sharepoint.com/personal/hyeae_connect_ust_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fhyeae%5Fconnect%5Fust%5Fhk%2FDocuments%2Fdataset%2FPASCALContext%2Etar%2Egz&parent=%2Fpersonal%2Fhyeae%5Fconnect%5Fust%5Fhk%2FDocuments%2Fdataset&ga=1), [NYUDv2.tar.gz](https://hkustconnect-my.sharepoint.com/personal/hyeae_connect_ust_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fhyeae%5Fconnect%5Fust%5Fhk%2FDocuments%2Fdataset%2FNYUDv2%2Etar%2Egz&parent=%2Fpersonal%2Fhyeae%5Fconnect%5Fust%5Fhk%2FDocuments%2Fdataset&ga=1)

And then extract the datasets by:
    ```
    tar xfvz NYUDv2.tar.gz
    tar xfvz PASCALContext.tar.gz
    ```
## Requirements

1. Clone the repo `git clone https://github.com/scale-lab/E-MTL.git ; cd E-MTL`

2. Create a virtual environment with Python 3.9 or later `python -m venv env ; source env/bin/activate`

3. Install the requirements using `pip install -r requirements.txt`

### Training
```
python -m torch.distributed.launch 
        --nproc_per_node 1 
        --master_port 12345 
        main.py --cfg {CONFIG.yaml}
                --pascal {PASCAL_DATA_DIR}
                --tasks {TASK_NAMES} 
                --batch-size 64 
                --ckpt-freq 10
                --epoch 200 
                [--resume-backbone {SWIN_PRETRAINED.pth}]
```
- `CONFIG.yaml` is the path of the desired model configuration, check [model.args](https://github.com/scale-lab/E-MTL/blob/master/configs/swin/swin_tiny_patch4_window7_224.yaml) for an example.
- `PASCAL_DATA_DIR` is the path of the downloaded pascal dataset.
- `TASK_NAMES` is the name of the desired tasks, available tasks for Pascal dataset are `semseg`, `normals`, `sal`, and `human_parts`. For example, to create a model that performs semantic segmentation and saliency distillation, `TASK_NAMES` should be set to `semseg,sal`
- `SWIN_PRETRAINED.pth` is the path to the pretrained SWIN transformer backbone. Pretrained SWIN transformer backbones can be downloaded from their [REPO](https://github.com/microsoft/Swin-Transformer). For example, to download pretrained SWIN Tiny, use `wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth`

## Inference 
```
python -m torch.distributed.launch 
        --nproc_per_node 1 
        --master_port 12345 
        main.py --cfg {CONFIG.yaml}
                --pascal {PASCAL_DATA_DIR}
                --tasks {TASK_NAMES} 
                --batch-size 64 
                --resume {PRETRAINED.pth}
                --eval
```
- `CONFIG.yaml` is the path of the desired model configuration, check [model.args](https://github.com/scale-lab/E-MTL/blob/master/configs/swin/swin_tiny_patch4_window7_224.yaml) for an example.
- `PASCAL_DATA_DIR` is the path of the downloaded pascal dataset.
- `TASK_NAMES` is the name of the desired tasks, available tasks for Pascal dataset are `semseg`, `normals`, `sal`, and `human_parts`. For example, to create a model that performs semantic segmentation and saliency distillation, `TASK_NAMES` should be set to `semseg,sal`
- `PRETRAINED.pth` is the path to the pretrained E-MTL model.

## Authorship
Since the release commit is squashed, the GitHub contributors tab doesn't reflect the authors' contributions. The following authors contributed equally to this codebase:
- [Ahmed Agiza](https://github.com/ahmed-agiza)
- [Marina Neseem](https://github.com/marina-neseem)

## License
MIT License. See [LICENSE](LICENSE) file
