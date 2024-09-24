FILE=$1

if [ $FILE == "pretrained-network-celeba-hq" ]; then
    URL=https://www.dropbox.com/s/96fmei6c93o8b8t/100000_nets_ema.ckpt?dl=0
    mkdir -p ./expr/checkpoints/celeba_hq
    OUT_FILE=./expr/checkpoints/celeba_hq/100000_nets_ema.ckpt
    wget -N $URL -O $OUT_FILE

elif  [ $FILE == "pretrained-network-afhq" ]; then
    URL=https://www.dropbox.com/s/etwm810v25h42sn/100000_nets_ema.ckpt?dl=0
    mkdir -p ./expr/checkpoints/afhq
    OUT_FILE=./expr/checkpoints/afhq/100000_nets_ema.ckpt
    wget -N $URL -O $OUT_FILE
    
elif  [ $FILE == "wing" ]; then
    URL=https://www.dropbox.com/s/tjxpypwpt38926e/wing.ckpt?dl=0
    mkdir -p ./expr/checkpoints/
    OUT_FILE=./expr/checkpoints/wing.ckpt
    wget -N $URL -O $OUT_FILE
    URL=https://www.dropbox.com/s/91fth49gyb7xksk/celeba_lm_mean.npz?dl=0
    OUT_FILE=./expr/checkpoints/celeba_lm_mean.npz
    wget -N $URL -O $OUT_FILE

elif  [ $FILE == "celeba-hq-dataset" ]; then
    URL=https://www.dropbox.com/s/f7pvjij2xlpff59/celeba_hq.zip?dl=0
    ZIP_FILE=./data/celeba_hq.zip
    mkdir -p ./data
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./data
    rm $ZIP_FILE

elif  [ $FILE == "afhq-dataset" ]; then
    URL=https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=0
    ZIP_FILE=./data/afhq.zip
    mkdir -p ./data
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./data
    rm $ZIP_FILE

elif  [ $FILE == "afhq-v2-dataset" ]; then
    #URL=https://www.dropbox.com/s/scckftx13grwmiv/afhq_v2.zip?dl=0
    URL=https://www.dropbox.com/s/vkzjokiwof5h8w6/afhq_v2.zip?dl=0
    ZIP_FILE=./data/afhq_v2.zip
    mkdir -p ./data
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./data
    rm $ZIP_FILE

elif [ $FILE == "faces-dataset" ]; then
    file_url="https://drive.usercontent.google.com/download?id=1IGrTr308mGAaCKotpkkm8wTKlWs9Jq-p&authuser=0"
    output_file="./data/crypko_data.zip"
    wget --load-cookies /tmp/cookies.txt "${file_url}&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt \
        --keep-session-cookies --no-check-certificate "${file_url}" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')" \
        -O ${output_file} && rm -rf /tmp/cookies.txt

else
    echo "Available arguments are pretrained-network-celeba-hq, pretrained-network-afhq, celeba-hq-dataset, and afhq-dataset."
    exit 1

fi
