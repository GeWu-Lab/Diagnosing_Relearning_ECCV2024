#!/bin/bash
python ours.py \
--dataset CREMAD \
--model ours \
--gpu_ids 3 \
--n_classes 6 \
--move_lambda 3 \
--reinit_epoch 20 \
--reinit_num 3 \
--epochs 90 \
--train \
| tee log_print/ours.log