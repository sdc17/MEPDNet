python run.py --model mepdnet --mode train -l 0.00008 -b 2 -e 100 --gpu-id 0 1

python run.py --model unet --mode train -l 0.0002 -b 5 -e 100 --gpu-id 0

python run.py --model deeplab_v3+ --mode train -l 0.001 -b 5 -e 100 --gpu-id 0

python run.py --model segnet --mode train -l 0.002 -b 5 -e 100 --gpu-id 1

python run.py --model r2unet --mode train -l 0.0001 -b 2 -e 100 --gpu-id 0 1

python run.py --model att_unet --mode train -l 0.0002 -b 6 -e 100 --gpu-id 0 1

python run.py --model r2att_unet --mode train -l 0.0001 -b 2 -e 100 --gpu-id 0 1

python run.py --model scse_unet --mode train -l 0.0005 -b 16 -e 100 --gpu-id 0 1

python run.py --model cenet --mode train -l 0.0001 -b 16 -e 100 --gpu-id 0 1

python run.py --model nested_unet --mode train -l 0.0005 -b 4 -e 100 --gpu-id 0 1