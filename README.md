# SWSNet
## Environment and Datasets
- Environment

Create environmemt. 

Make sure the **torch version matches cuda**.
We use and recommend python 3.8,  torch 2.1.1, torchvision 0.16.1, torchaudio 0.9.0 and cuda 11.8.

```
conda create --name swsnet python=3.8
conda activate swsnet
cd .../.../SWSNet-main
sh setup.sh
```

-Datasets
Download the [Teeth3DS+](https://github.com/abenhamadou/3DTeethSeg22_challenge) dataset
Position the dataset as follows:
```
data
    |3dteethseg
        | raw
            | lower
            | upper
            | private-testing-set.txt
            | public-training-set-1.txt
            | public-training-set-2.txt
            | testing_lower.txt
            | testing_upper.txt
            | training_lower.txt
            | training_upper.txt

```
The model will preprocess the dataset automatically, when first training or evaluating.
