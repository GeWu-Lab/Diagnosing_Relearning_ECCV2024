#!/bin/bash
python baseline.py \
--dataset CREMAD \
--model baseline \
--gpu_ids 2 \
--n_classes 6 \
--epochs 90 \
--train \
| tee log_print/baseline.log