#!/bin/bash
export CUDA_VISIBLE_DEVICES="1"
data=gowalla
root_dic="result/log_${data}/baselines"


# ============ Baseline Comparison =========================
( 
epoch=100;
s=10;
b=2048;
for s in 10 20 30 40 50
do
    # Sampled Softmax with no debias (SSL)
    logs="${root_dic}/ssl/bs_2048/lr_0.005/diff_seed";
    lr=0.005;
    w=1e-5;
    python run.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 1 --batch_size $b --data_name ${data} --epoch ${epoch} --seed ${s};

    # Sampled Softmax with pop debias (SSL-Pop)
    logs="${root_dic}/ssl-pop/bs_2048/lr_0.01/diff_seed";
    lr=0.01;
    w=1e-5;
    python run.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 2 --batch_size $b --data_name ${data} --epoch ${epoch} --seed ${s};

    # sampling bias correction (correct-sfx)
    logs="${root_dic}/correct-sfx/bs_2048/lr_0.01/diff_seed";
    lr=0.01;
    w=1e-5;
    python run.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 3 --batch_size $b --data_name ${data} --epoch ${epoch} --seed ${s} --alpha 1e-4;

    # Mix Negatives Sampling (MNS)
    logs="${root_dic}/mns/bs_2048/lr_0.01/diff_seed";
    lr=0.01;
    w=1e-5;
    python run.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 4 --batch_size $b --data_name ${data} --epoch ${epoch} --seed ${s} --sample_size ${b};

    # Batch importance Resampling (BIR)
    logs="${root_dic}/bir/bs_2048/lr_0.01/diff_seed";
    lr=0.01;
    w=1e-5;
    python run.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 6 --batch_size $b --data_name ${data} --epoch ${epoch} --seed ${s}
done )
