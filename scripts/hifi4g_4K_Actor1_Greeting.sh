export CUDA_HOME=/usr/local/cuda 
export CUDA_VISIBLE_DEVICES=1

id_name=4K_Actor1_Greeting

dataset=../dataset/HiFi4G/${id_name}/
iterations=30000
smpl_type=smpl

exp_name=hifi4g_mono/${id_name}

python train.py -s $dataset --eval \
    --exp_name $exp_name \
    --motion_offset_flag \
    --smpl_type ${smpl_type} \
    --actor_gender neutral \
    --iterations ${iterations} \
    --white_background \
    --port 6005 \
    # --is_continue \
#    --wandb_disable
