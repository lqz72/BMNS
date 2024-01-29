#!/bin/bash
export CUDA_VISIBLE_DEVICES="2"
data=gowalla
root_dic="result/log_${data}/batch_mix"


# ========================= Run BMNS =========================
( logs="${root_dic}/bs_2048/lr_0.01/diff_seed";
epoch=100;
lr=0.01;
w=1e-6;
s=10;
alpha=1;
t=1;
b=2048; 
sd=uniform;
gamma=0.2;
for s in 10 20 30 40 50
do
    python run.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 5 --batch_size $b --data_name ${data} --epoch ${epoch} --seed ${s}  --beta_alpha ${alpha} --beta_beta ${alpha} --mix_neg_num ${b} --sample_dist ${sd} --temp ${t} --loss_gamma ${gamma};
done )






