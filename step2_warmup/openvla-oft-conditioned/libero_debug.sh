export NCCL_DEBUG=WARN 
# export WANDB_API_KEY=''
#
# /data/users/kongyilun/code/RLinf/dataset/
# export WANDB_MODE='offline'
unset CUDA_VISIBLE_DEVICES
# export CUDA_VISIBLE_DEVICES=1
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path /data/users/kongyilun/models/openvla-7b \
  --data_root_dir /data/users/kongyilun/code/RLinf/dataset/ \
  --dataset_name libero_object_no_noops60 \
  --run_root_dir runs \
  --use_l1_regression False \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 1 \
  --use_proprio False \
  --batch_size 2 \
  --learning_rate 5e-4 \
  --max_steps 100000 \
  --use_val_set False \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity kongyilun333 \
  --wandb_project openvla-oft-z \
  --run_id_note libero_condition_z_debug