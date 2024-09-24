#!/bin/bash

FILE=$1

# celeba-hq
# CUDA_VISIBLE_DEVICES=3 
#pkill -u pradipta -f main.py
# find . -maxdepth 1 -type f -exec mv -t weights/ {} +
# conda activate DLPCV
# source ~/.bashrc 
# cd /project/g/pradipta/madhu/stargan-v2
# rsync -av --exclude='expr/' /project/g/pradipta/madhu/stargan-v2 ./
# cp /project/g/pradipta/madhu/stargan-v2/expr/checkpoints/celeba_lm_mean.npz /tmp2/pradipta/madhu/stargan-v2/expr/checkpoints/
# cp /project/g/pradipta/madhu/stargan-v2/expr/checkpoints/wing.ckpt /tmp2/pradipta/madhu/stargan-v2/expr/checkpoints/


if [ "$FILE" == "train" ]; then
    echo "v100 2 faces " > output_faces.log

    CUDA_VISIBLE_DEVICES=2 nohup python main.py --mode train --num_domains 1 --w_hpf 1 \
               --lambda_reg 1 --lambda_sty 1 --lambda_ds 1 --lambda_cyc 1 --batch_size 6 \
               --generator_backbone "ResNet" \
               --train_img_dir /project/g/pradipta/star-data/data/faces/train \
               --val_img_dir /project/g/pradipta/star-data/data/faces/val >> output_faces.log 2>&1 &
elif [ "$FILE" == "trainSwinStyle" ]; then
    echo "12 CUDA=3 swin style 當style encoder" >> output_SwinStyle_and_ResNet.log

    CUDA_VISIBLE_DEVICES=3 nohup python main.py --mode train --num_domains 2 --w_hpf 1 \
                --style_dim 256 \
               --lambda_reg 1 --lambda_sty 1 --lambda_ds 1 --lambda_cyc 1 --batch_size 8 \
               --generator_backbone "SwinStyle" \
               --train_img_dir /project/g/pradipta/star-data/data/celeba_hq/train \
               --val_img_dir /project/g/pradipta/star-data/data/celeba_hq/val >> output_SwinStyle_and_ResNet.log 2>&1 &
elif [ "$FILE" == "trainSwin" ]; then
    echo "cml9 CUDA_VISIBLE_DEVICES=2 調整styleEncoder為swin transformer 加入noise modelName noise_unet.ckpt" > test.log

    CUDA_VISIBLE_DEVICES=2 nohup python main.py --mode train --num_domains 2 --w_hpf 1 --style_dim 128\
               --lambda_reg 1 --lambda_sty 1 --lambda_ds 1 --lambda_cyc 1 --batch_size 8 \
               --generator_backbone "SwinUnet" \
               --train_img_dir /project/g/pradipta/star-data/data/celeba_hq/train \
               --val_img_dir /project/g/pradipta/star-data/data/celeba_hq/val >> test.log 2>&1 &

elif [ "$FILE" == "trainSwinT" ]; then
    echo "cml9 CUDA_VISIBLE_DEVICES=1 改為只當swin encoder pytorch T" > output-swinT.log

    CUDA_VISIBLE_DEVICES=1 nohup python main.py --mode train --num_domains 2 --w_hpf 1 --style_dim 64\
               --lambda_reg 1 --lambda_sty 2 --lambda_ds 1 --lambda_cyc 1 --batch_size 8 \
               --generator_backbone "SwinT" \
               --train_img_dir /project/g/pradipta/star-data/data/celeba_hq/train \
               --val_img_dir /project/g/pradipta/star-data/data/celeba_hq/val >> output-swinT.log 2>&1 &


elif [ "$FILE" == "trainSwinEncoder" ]; then
    echo "cml12 CUDA_VISIBLE_DEVICES=4 改為只當swin encoder" > output-only_encoder.log

    CUDA_VISIBLE_DEVICES=4 nohup python main.py --mode train --num_domains 2 --w_hpf 1 --style_dim 64\
               --lambda_reg 1 --lambda_sty 1 --lambda_ds 1 --lambda_cyc 1 --batch_size 8 --resume_iter 020000 \
               --generator_backbone "SwinEncoder" \
               --train_img_dir /project/g/pradipta/star-data/data/celeba_hq/train \
               --val_img_dir /project/g/pradipta/star-data/data/celeba_hq/val >> output-only_encoder.log 2>&1 &

elif [ "$FILE" == "trainSwinAfhq" ]; then
    echo "python main.py --mode train --num_domains 3 --w_hpf 1 \\
               --lambda_reg 1 --lambda_sty 1 --lambda_ds 1 --lambda_cyc 1 --generator_backbone "swin" --batch_size 8
               --train_img_dir /project/g/pradipta/star-data/data/afhq/train 
               --val_img_dir /project/g/pradipta/star-data/data/afhq/val" > outputSwin-3090-2.log

    CUDA_VISIBLE_DEVICES=3 nohup python main.py --mode train --num_domains 3 --w_hpf 1 \
               --lambda_reg 1 --lambda_sty 1 --lambda_ds 1 --lambda_cyc 1 --batch_size 8 \
               --generator_backbone "Swin" \
               --train_img_dir /project/g/pradipta/star-data/data/afhq/train \
               --val_img_dir /project/g/pradipta/star-data/data/afhq/val >> outputSwin-3090-2.log 2>&1 &

elif [ "$FILE" == "eval" ]; then
    echo "nohup python main.py --mode eval --num_domains 2 --w_hpf 1 \\
                --resume_iter 100000 \\
                --train_img_dir /project/g/pradipta/star-data/data/celeba_hq/train \\
                --val_img_dir /project/g/pradipta/star-data/data/celeba_hq/val \\
                --checkpoint_dir expr/checkpoints/celeba_hq \\
                --eval_dir expr/eval/celeba_hq > output.log 2>&1 &" > output.log

    nohup python main.py --mode eval --num_domains 2 --w_hpf 1 \
                --resume_iter 100000 \
                --train_img_dir /project/g/pradipta/star-data/data/celeba_hq/train \
                --val_img_dir /project/g/pradipta/star-data/data/celeba_hq/val \
                --checkpoint_dir expr/checkpoints/celeba_hq \
                --eval_dir expr/eval/celeba_hq >> output.log 2>&1 &

elif [ "$FILE" == "align" ]; then
    python main.py --mode align \
                --inp_dir assets/selfData \
                --out_dir assets/selfDataSrc

elif [ "$FILE" == "sample" ]; then
    CUDA_VISIBLE_DEVICES=0 nohup python main.py --mode sample --num_domains 3 --resume_iter 100000 --w_hpf 0 \
               --checkpoint_dir expr/checkpoints/run/afhq \
               --result_dir expr/results/self \
               --src_dir assets/representative/afhq/src \
               --ref_dir assets/representative/afhq/ref > nohup.out 2>&1 &

elif [ "$FILE" == "sampleResNet" ]; then
    CUDA_VISIBLE_DEVICES=0 nohup python main.py --mode sample --num_domains 2 --resume_iter 100000 --w_hpf 1 \
               --checkpoint_dir expr/checkpoints/celeba_hq \
               --result_dir expr/results/celeba_hq \
               --src_dir assets/representative/celeba_hq/src \
               --ref_dir assets/representative/celeba_hq/ref > nohup.out 2>&1 &
elif [ "$FILE" == "sampleSwinUnet" ]; then
    CUDA_VISIBLE_DEVICES=2 nohup python main.py --mode sample --num_domains 2 --resume_iter 100000 --w_hpf 1 \
                --generator_backbone "SwinUnet" --style_dim 512 \
               --checkpoint_dir expr/checkpoints/run/celeba_hq \
               --result_dir expr/results/celeba_hq \
               --src_dir assets/representative/celeba_hq/src \
               --ref_dir assets/representative/celeba_hq/ref > nohup.out 2>&1 &

elif [ "$FILE" == "sampleSwinEncoder" ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python main.py --mode sample --num_domains 2 --resume_iter 100000 --w_hpf 1 \
                --generator_backbone "SwinEncoder" --style_dim 64\
               --checkpoint_dir expr/checkpoints/run/celeba_hq \
               --result_dir expr/results/celeba_hq \
               --src_dir assets/representative/celeba_hq/src \
               --ref_dir assets/representative/celeba_hq/ref > nohup.out 2>&1 &

elif [ "$FILE" == "sampleSwinStyle" ]; then
    CUDA_VISIBLE_DEVICES=1 nohup python main.py --mode sample --num_domains 2 --resume_iter 030000 --w_hpf 1 \
                --generator_backbone "SwinStyle" --style_dim 256\
               --checkpoint_dir expr/checkpoints/run/celeba_hq \
               --result_dir expr/results/celeba_hq \
               --src_dir assets/representative/celeba_hq/src \
               --ref_dir assets/representative/celeba_hq/ref > nohup.out 2>&1 &

elif [ "$FILE" == "sampleSwinUnetAfhq" ]; then
    CUDA_VISIBLE_DEVICES=0 nohup python main.py --mode sample --num_domains 3 --resume_iter 020000 --w_hpf 1 --generator_backbone "Swin"\
               --checkpoint_dir expr/checkpoints/run/afhq \
               --result_dir expr/results/afhq \
               --src_dir assets/representative/afhq/src \
               --ref_dir assets/representative/afhq/ref > nohup.out 2>&1 &
elif [ "$FILE" == "sampleFaces" ]; then
    CUDA_VISIBLE_DEVICES=0 nohup python main.py --mode sample --num_domains 1 --resume_iter 040000 --w_hpf 1 \
               --checkpoint_dir expr/checkpoints/run/celeba_hq \
               --result_dir expr/results/celeba_hq \
               --src_dir assets/representative/faces/src \
               --ref_dir assets/representative/faces/ref > nohup.out 2>&1 &

fi

#CUDA_VISIBLE_DEVICES = 0
