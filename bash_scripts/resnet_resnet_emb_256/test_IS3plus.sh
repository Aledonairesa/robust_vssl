#!/bin/bash
#SBATCH --job-name=vssl
#SBATCH --output=slurm_outputs/gpu_test_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00-05:00

# Load required modules
module load conda

# Activate environment
source activate /data/upftfm03/adonaire/myenv

# Change directory where code lives
cd /home/adonaire/code/robust_vssl

# Execute script
srun -n 1 python -u main.py \
  --exp_name resnet_resnet_emb_256 \
  --output_dir /data/upftfm03/adonaire/outputs \
  --save_val_embeddings \
  --save_test_embeddings \
  --batch_size 64 \
  --n_threads 16 \
  --epochs 80 \
  --early_stop_patience 5 \
  --eval_freq 1 \
  --eval_start 1 \
  --img_backbone_type resnet18 \
  --aud_backbone_type resnet18 \
  --img_aug moco_v2 \
  --aud_aug SpecAug \
  --equi_loss_type mse \
  --rescale_factor 0.5 1.0 \
  --max_rotation_angle 30 \
  --trainset_path /data/upftfm03/adonaire/vggsound/train \
  --train_set_scale subset_10k \
  --val_set VGGS \
  --val_set_scale subset_1k \
  --cl_loss ce \
  --embedding_dim 256 \
  --test_set IS3plus \
  --is3plus_test_path /data/upftfm03/adonaire/IS3plus \
  --test_set_scale fullset \
  --test /data/upftfm03/adonaire/outputs/resnet_resnet_emb_256/checkpoints/model_best_epoch19.pth.tar \

# --- Unused/Commented Parameters ---
# --sigmoid_t_init 0.5
# --sigmoid_b_init -5
# --use_vision_blocks
# --vggss_test_path /content/data/VGGSound_test/
# --testing_set_scale subset_250
# --test ckpts/SSL-TIE_VGGSound_1k_test/model/model_best_epoch7.pth.tar