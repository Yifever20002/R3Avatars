export CUDA_HOME=/usr/local/cuda 
export CUDA_VISIBLE_DEVICES=3

id_name=0010_03

dataset=../dataset/DNA-Rendering/${id_name}/
iterations=60000
smpl_type=smplx

exp_name=dna_ab_sh/${id_name}

python train.py -s $dataset --eval \
    --exp_name $exp_name \
    --motion_offset_flag \
    --smpl_type ${smpl_type} \
    --actor_gender neutral \
    --iterations ${iterations} \
    --port 6005 \
#    --wandb_disable
