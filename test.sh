python run.py --model mepdnet --mode test --state 70 -b 4 --gpu-ids 0 1

python run.py --model deeplab_v3+ --mode test --state 97 -b 4 --gpu-ids 0

python run.py --model unet --mode test --state 99 -b 4 --gpu-ids 0

python run.py --model segnet --mode test --state 94 -b 4 --gpu-ids 0

python run.py --model r2unet --mode test --state 90 -b 4 --gpu-ids 0 1

python run.py --model att_unet --mode test --state 99 -b 4 --gpu-ids 0 1

python run.py --model r2att_unet --mode test --state 90 -b 4 --gpu-ids 0 1

python run.py --model scse_unet --mode test --state 100 -b 4 --gpu-ids 0 1

python run.py --model cenet --mode test --state 89 -b 4 --gpu-ids 0 1

python run.py --model nested_unet --mode test --state 82 -b 4 --gpu-ids 0 1