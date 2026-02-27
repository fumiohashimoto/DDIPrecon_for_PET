#!/bin/bash

save_root=$1
test_root=$2
num_steps=$3
lora_rank=$4
T_sampling=$5
start_t=$6
em_itr=$7
em_beta=$8

Nview=128
eta=0.85
gamma=5.0
lr='5e-4'
gamma=5.0
end_t=0
CG_iter_adapt=1
CG_iter=0
num_test_slice=8

python main_simu.py \
    --config brainweb_2d_mri.yml \
    --adaptation \
    --Nview ${Nview} \
    --eta ${eta} \
    --T_sampling ${T_sampling} \
    --deg "PET" \
    --save_root ${save_root} \
    --test_root ${test_root} \
    --num_test_slice ${num_test_slice} \
    --lora_rank ${lora_rank} \
    --lr ${lr} \
    --num_steps ${num_steps} \
    --gamma ${gamma}\
    --CG_iter_adapt ${CG_iter_adapt}\
    --CG_iter ${CG_iter}\
    --start_t ${start_t}\
    --end_t ${end_t}\
    --em_itr ${em_itr}\
    --em_beta ${em_beta}\