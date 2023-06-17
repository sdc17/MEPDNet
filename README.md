# Multi-Encoder Parse-Decoder Network for Sequential Medical Image Segmentation (MEPDNet)

Official implementation of [Multi-Encoder Parse-Decoder Network for Sequential Medical Image Segmentation](https://ieeexplore.ieee.org/abstract/document/9506463). 

### Supported Models

* [MEPDNet(Ours)](./model/MEPDNet.py)
* [SegNet](./model/SegNet.py)
* [DeepLabv3+](./model/DeepLab_v3plus.py)
* [U-Net](./model/UNet.py)
* [Attention U-Net](./model/AttUNet.py)
* [R2U-Net](./model/AR2UNet.py)
* [Attention R2U-Net](./model/AR2UNet.py)
* [ScSE U-Net](./model/SCSEUNet.py)
* [CE-Net](./model/CENet.py)
* [UNet++](./model/NestedUNet.py)


### Training Script

```bash
python run.py --model $MODEL_NAME --mode train -l $LR -b $BATCH_SIZE -e $EPOCHS --gpu-id $GPU_ID

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL
  --mode {train,test,use}
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size
  -l LR, --learning-rate LR
                        Learning rate
```

For example, to train MEPDNet:

```bash
python run.py --model mepdnet --mode train -l 0.00008 -b 2 -e 100 --gpu-id 0 1
```

Training scripts for other models are in [train.sh](train.sh).


### Evaluation Script

```bash
python run.py --model $MODEL_NAME --mode test --state $MODEL_ID -b $BATCH_SIZE --gpu-ids $GPU_ID

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL
  --mode {train,test,use}
  --gpu-ids GPU_IDS [GPU_IDS ...]
  --state STATE
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size
```

For example, to evaluate MEPDNet:
```bash
python run.py --model mepdnet --mode test --state 70 -b 4 --gpu-ids 0 1
```
Evaluation scripts for other models are in [test.sh](test.sh).


### Cite
If you find this work useful, please consider citing the corresponding paper:
```bibtex
@inproceedings{shi2021multi,
  title={Multi-encoder parse-decoder network for sequential medical image segmentation},
  author={Shi, Dachuan and Liu, Ruiyang and Tao, Linmi and He, Zuoxiang and Huo, Li},
  booktitle={2021 IEEE international conference on image processing (ICIP)},
  pages={31--35},
  year={2021},
  organization={IEEE}
}
```