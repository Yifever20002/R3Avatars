export CUDA_HOME=/usr/local/cuda 
export CUDA_VISIBLE_DEVICES=3

id_name=100009

dataset=../dataset/MVHumanNet/${id_name}/
iterations=60000
smpl_type=smplx

exp_name=mvhuman_stage1/${id_name}

python train.py -s $dataset --eval \
    --exp_name $exp_name \
    --motion_offset_flag \
    --smpl_type ${smpl_type} \
    --actor_gender neutral \
    --iterations ${iterations} \
    --port 6005 \
    # --wandb_disable
