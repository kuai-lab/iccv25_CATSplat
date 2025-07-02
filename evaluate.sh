#!/bin/sh

# re10k testing
CUDA_VISIBLE_DEVICES=3 \
python evaluate.py \
    hydra.run.dir=./exp/evaluate/CATSplat \
    hydra.job.chdir=true \
    +experiment=layered_re10k \
    +dataset.crop_border=true \
    dataset.test_split_path=./splits/re10k_mine_filtered/test_files.txt \
    model.depth.version=v1 \
    ++eval.save_vis=false \
    run.checkpoint=ckpts/CATSplat.pth


