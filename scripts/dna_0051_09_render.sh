export CUDA_HOME=/usr/local/cuda 
export CUDA_VISIBLE_DEVICES=3

id_name=0051_09

dataset=../dataset/DNA-Rendering/${id_name}/
iterations=60000
smpl_type=smplx
sim_type=euclidean   # [axis-angle, quater, matrix, euclidean]
pose_step=2

exp_name=dna_stage1/${id_name}

outdir=output/${exp_name}

python render.py -m $outdir \
   --motion_offset_flag \
   --smpl_type ${smpl_type} \
   --sim_type ${sim_type} \
   --pose_step ${pose_step} \
   --actor_gender neutral \
   --iteration ${iterations} \
   --skip_train \
   --mono_test \
