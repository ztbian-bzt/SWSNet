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

- Datasets
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

## Train
'''
python train_network.py --epochs 100 --tb_save_dir logs --experiment_name swsnet --experiment_version 1 --train_batch_size 2 --n_bit_precision 16 --train_test_split 1 --devices 0
python train_network.py --epochs 100 --tb_save_dir logs --experiment_name swsnet --experiment_version 1 --train_batch_size 2 --n_bit_precision 16 --train_test_split 2 --devices 0
'''

## Evaluate
```
python test_network.py --tb_save_dir logs --experiment_name swsnet --experiment_version test --devices 0 --n_bit_precision 16 --train_test_split 1 --ckpt <checkpoint path>
python test_network.py --tb_save_dir logs --experiment_name swsnet --experiment_version test --devices 0 --n_bit_precision 16 --train_test_split 2 --ckpt <checkpoint path>
```

## Inference
- inference1.py
Infer a single raw data.(need to modify the checkpoint path and the path of that single data)
- infernece2.py
Infer batch raw data.(need to modify the checkpoint path and the path of raw data file)


## Acknowledgements
<details><summary>models</summary>
    
*[DilatedToothSegNet](https://github.com/LucasKre/dilated_tooth_seg_net)
    
*[MeshSegNet](https://github.com/Tai-Hsien/MeshSegNet)

*[DBGANet](https://github.com/zhijieL513/DBGANet)

*[DGCNN](https://github.com/WangYueFt/dgcnn)

*[PointNet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

*[THISNet](https://github.com/li-pengcheng/THISNet)

*[TSGCNet](https://github.com/ZhangLingMing1/TSGCNet)
</details>

