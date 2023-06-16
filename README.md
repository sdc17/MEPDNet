# Multi-Encoder Parse-Decoder Network for Sequential Medical Image Segmentation (MEPDNet)

Official implementation of [Multi-Encoder Parse-Decoder Network for Sequential Medical Image Segmentation](https://ieeexplore.ieee.org/abstract/document/9506463). 

### Supported Models

* MEPDNet (Ours)
* SegNet
* DeepLabv3+
* U-Net
* Attention U-Net
* R2U-Net
* Attention R2U-Net
* ScSE U-Net
* CE-Net
* UNet++


### Train

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
See examples in [train.sh](train.sh).


### Evaluate

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
See examples in [test.sh](test.sh).


### Cite

<pre/>
@inproceedings{shi2021multi,
  title={Multi-encoder parse-decoder network for sequential medical image segmentation},
  author={Shi, Dachuan and Liu, Ruiyang and Tao, Linmi and He, Zuoxiang and Huo, Li},
  booktitle={2021 IEEE international conference on image processing (ICIP)},
  pages={31--35},
  year={2021},
  organization={IEEE}
}
</pre>