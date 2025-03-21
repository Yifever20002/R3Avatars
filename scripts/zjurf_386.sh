export CUDA_HOME=/usr/local/cuda 
export CUDA_VISIBLE_DEVICES=3

id_name=my_386

dataset=../dataset/zju_mocap/${id_name}/
iterations=10000
smpl_type=smpl

exp_name=zjurf_mono/${id_name}

python train.py -s $dataset --eval \
    --exp_name $exp_name \
    --motion_offset_flag \
    --smpl_type ${smpl_type} \
    --actor_gender neutral \
    --iterations ${iterations} \
    --port 6005 \
    --wandb_disable
