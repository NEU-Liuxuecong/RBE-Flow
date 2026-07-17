# ECCV (ECCV 2026)
### [Project Page](https://github.com/NEU-Liuxuecong/RBE-Flow) 
Official implementation of **RBE-Flow: Recurrent Bayesian Estimation on Feature Manifolds for Cross-Modal Registration**.

## Links
- **Code**: [GitHub Repository](https://github.com/NEU-Liuxuecong/RBE-Flow)
- **Paper**: [arXiv]()
- **Pretrained Weights**: [Release](https://github.com/NEU-Liuxuecong/RBE-Flow/releases)

## Authors
**Mengzhu Ding, Xin Song, Xiaoke Ding, Hongwei Ding, Xuecong Liu**


## Overview
<img src="assets/over.pdf">


## Data Preparation
To evaluate/train CRFT, you will need to download the required datasets. 
* [RoadScene]()
* [OSdataset]()
* [WHU-OPT-SAR]()


You can create symbolic links to wherever the datasets were downloaded in the `datasets` folder

```Shell
├── datasets
    ├── os_dataset
        ├── train
           ├── image_pair
           ├── truth_flow
           ├── datum
        ├── test
           ├── image_pair
           ├── truth_flow
           ├── datum
        ├── val
           ├── image_pair
           ├── truth_flow
           ├── datum
     ├── RoadScene
        ├── train
           ├── image_pair
           ├── truth_flow
           ├── datum
        ├── test
           ├── image_pair
           ├── truth_flow
           ├── datum
        ├── val
           ├── image_pair
           ├── truth_flow
           ├── datum
```

## Requirements
```shell
conda create --name RBE-Flow python=3.9.7
conda activate crft
conda install pytorch=2.3.1 torchvision=0.18.1 pytorch-cuda=12.1 matplotlib tensorboard scipy opencv -c pytorch -c nvidia
pip install opencv-python==4.8.0.76
pip install numpy==1.26.4
pip install pytorch-lightning loguru joblib tqdm h5py einops
```

## Training
```shell
python train.py
```

## Models
We provide models trained on OSdataset and RoadScene respectively. The default path of the models for evaluation is:
```Shell
├── checkpoints
    ├── OS_15trans0.1scale30rot.ckpt
    ├── RS_15trans0.1scale30rot.ckpt 
    ├── WHU_15trans0.1scale30rot.ckpt 
```


## Test
```Shell
python test.py 
```


## Citation
```bibtex

```