#!/bin/sh
python utilities/cythonize_invert_flow.py build_ext --inplace
#python3 utilities/downsample_images.py --indir "/misc/home6/s0101/geo_dataset/" --outdir "/misc/home6/s0101/geo_dataset_resized" --rgb-suffix "tif"

# DATA_DIR="/misc/home4/s0105/geo_dataset"
DATA_DIR="/misc/home6/s0101/geo_dataset"
# WDATA_DIR="/misc/home4/s0105/geo_dataset"
WDATA_DIR="/misc/home6/s0101/geo_dataset"
GPUS="1"

echo 'Start training'
# train fold 1
./dist_train.sh $DATA_DIR $WDATA_DIR  1 $GPUS > logs/l1

echo 'done dist_train'

./dist_train_tune.sh $DATA_DIR $WDATA_DIR  1 $GPUS  /home/s0101/_scratch2/geopose_inna/weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_1_r2 >> logs/l1

rm -r preds

# train fold 6
# ./dist_train.sh $DATA_DIR $WDATA_DIR  6 $GPUS > logs/l6
# ./dist_train_tune.sh $DATA_DIR $WDATA_DIR  6 $GPUS  /home/s0105/_scratch2/project/weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_6_r2 >> logs/l6

echo 'Done training'