#!/bin/bash
#SBATCH --job-name=encoder_salf_rl
#SBATCH --time=00:59:59

### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --mem=64gb
#SBATCH -p a100

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12340
export WORLD_SIZE=4

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

### init virtual environment if needed
conda activate ssl_rl_game

### the command to run

srun python3 encoder2/main.py --batch_size 1024 --epoch 2 \
         --patch_size 16 --checkpoint_interval 2 --sequence_length 1 --image_size 128 \
         --data_type Random --transformer_layer 2 --hidden_dim 128 --n_heads 8 --distributed
