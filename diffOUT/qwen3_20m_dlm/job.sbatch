#!/bin/bash
#SBATCH --account=PHD_dallari
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:10:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=1
#SBATCH --output=/leonardo_work/PHD_dallari/scaling/oellm-autoexp/diffOUT/qwen3_20m_dlm/logs/slurm-%j.log
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --job-name=qwen3_20m_dlm_0

export HF_HOME=/leonardo_work/PHD_dallari/hf_cache
export WANDB_MODE=offline
export MACHINE_NAME=LEONARDO
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MASTER_ADDR=$(nslookup $(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1) | grep "Address: " | sed "s/Address: //" | tail) 
export MASTER_PORT=20074
export NUM_NODES=$SLURM_JOB_NUM_NODES
export GPUS_PER_NODE=4
export NUM_GPUS_PER_NODE=4
export NUM_GPUS=$((NUM_GPUS_PER_NODE*SLURM_NNODES))
export ARCH=$(uname -m)
export SLURM_CPUS_PER_TASK=32
export SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID
export NCCL_IB_TIMEOUT=120
export TRITON_LIBCUDA_PATH=/usr/local/cuda/lib64/stubs
export OMP_NUM_THREADS=16
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCHDYNAMO_DISABLE=1
export UB_SKIPMC=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

echo "Launching job qwen3_20m_dlm_0"
srun bash -c 'export HF_HOME='$HF_HOME' ; \
 export WANDB_MODE='$WANDB_MODE' ; \
 export MACHINE_NAME='$MACHINE_NAME' ; \
 export HF_ALLOW_CODE_EVAL='$HF_ALLOW_CODE_EVAL' ; \
 export HF_DATASETS_OFFLINE='$HF_DATASETS_OFFLINE' ; \
 export TRANSFORMERS_OFFLINE='$TRANSFORMERS_OFFLINE' ; \
 export MASTER_ADDR='$MASTER_ADDR' ; \
 export MASTER_PORT='$MASTER_PORT' ; \
 export NUM_NODES='$NUM_NODES' ; \
 export GPUS_PER_NODE='$GPUS_PER_NODE' ; \
 export NUM_GPUS_PER_NODE='$NUM_GPUS_PER_NODE' ; \
 export NUM_GPUS='$NUM_GPUS' ; \
 export ARCH='$ARCH' ; \
 export SLURM_CPUS_PER_TASK='$SLURM_CPUS_PER_TASK' ; \
 export SLURM_ARRAY_TASK_ID='$SLURM_ARRAY_TASK_ID' ; \
 export NCCL_IB_TIMEOUT='$NCCL_IB_TIMEOUT' ; \
 export TRITON_LIBCUDA_PATH='$TRITON_LIBCUDA_PATH' ; \
 export OMP_NUM_THREADS='$OMP_NUM_THREADS' ; \
 export CUDA_DEVICE_MAX_CONNECTIONS='$CUDA_DEVICE_MAX_CONNECTIONS' ; \
 export PYTORCH_CUDA_ALLOC_CONF='$PYTORCH_CUDA_ALLOC_CONF' ; \
 export CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' ; \
 export TORCHDYNAMO_DISABLE='$TORCHDYNAMO_DISABLE' ; \
 export UB_SKIPMC='$UB_SKIPMC' ; \
 export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='$TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD' ; \
 export LOCAL_ADDR=$(nslookup $(hostname) | grep "Address: " | sed "s/Address: //" | tail) ; \
singularity exec --env HF_HOME=$HF_HOME \
 --env WANDB_MODE=$WANDB_MODE \
 --env MACHINE_NAME=$MACHINE_NAME \
 --env HF_ALLOW_CODE_EVAL=$HF_ALLOW_CODE_EVAL \
 --env HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE \
 --env TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE \
 --env MASTER_ADDR=$MASTER_ADDR \
 --env MASTER_PORT=$MASTER_PORT \
 --env NUM_NODES=$NUM_NODES \
 --env GPUS_PER_NODE=$GPUS_PER_NODE \
 --env NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE \
 --env NUM_GPUS=$NUM_GPUS \
 --env ARCH=$ARCH \
 --env SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK \
 --env SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID \
 --env NCCL_IB_TIMEOUT=$NCCL_IB_TIMEOUT \
 --env TRITON_LIBCUDA_PATH=$TRITON_LIBCUDA_PATH \
 --env OMP_NUM_THREADS=$OMP_NUM_THREADS \
 --env CUDA_DEVICE_MAX_CONNECTIONS=$CUDA_DEVICE_MAX_CONNECTIONS \
 --env PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF \
 --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
 --env TORCHDYNAMO_DISABLE=$TORCHDYNAMO_DISABLE \
 --env UB_SKIPMC=$UB_SKIPMC \
 --env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=$TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD \
 --env PYTHONPATH=.:submodules/MegaDLMs \
 --env RUN_DIR=./submodules/MegaDLMs \
 --env PROJECT_DIR=. \
 --env HF_HOME=/leonardo_work/PHD_dallari/hf_cache \
 --nv --bind /leonardo_scratch \
 --bind /leonardo \
 --bind /leonardo_work/ \
 /leonardo_work/PHD_dallari/containers/megatronDLM.sif torchrun \
  --nproc-per-node 1\
  --nnodes 1\
  --max-restarts 0\
  --tee 3\
  --rdzv-endpoint $MASTER_ADDR:$MASTER_PORT\
  --rdzv-backend static\
  --node-rank $SLURM_NODEID\
  --master-addr $MASTER_ADDR\
  --master-port $MASTER_PORT\
   ./submodules/MegaDLMs/pretrain_difflm.py \
   --seq-length 513\
   --tokenizer-type HuggingFaceTokenizer\
   --tokenizer-model /leonardo_work/PHD_dallari/tokenizers/tokenizer-256k\
   --dist-ckpt-strictness log_unexpected\
   --ckpt-format torch\
   --bf16\
   --group-query-attention\
   --swiglu\
   --disable-bias-linear\
   --rotary-base 1000000\
   --normalization RMSNorm\
   --norm-epsilon 1e-06\
   --position-embedding-type rope\
   --use-rotary-position-embeddings\
   --attention-dropout 0.0\
   --hidden-dropout 0.0\
   --attention-softmax-in-fp32\
   --qk-layernorm\
   --num-layers 12\
   --hidden-size 384\
   --ffn-hidden-size 1152\
   --num-attention-heads 6\
   --kv-channels 64\
   --max-position-embeddings 513\
   --num-query-groups 2\
   --train-iters 8000\
   --micro-batch-size 64\
   --global-batch-size 128\
   --lr 0.002\
   --min-lr 0.0002\
   --lr-warmup-iters 400\
   --lr-decay-style cosine\
   --weight-decay 0.1\
   --adam-beta2 0.95\
   --no-bias-gelu-fusion\
   --no-bias-dropout-fusion\
   --no-bias-swiglu-fusion\
   --no-rope-fusion\
   --data-path /leonardo_work/PHD_dallari/fineweb-10BT_text_document\
   --split 998,2,0\
   --data-cache-path /leonardo_work/PHD_dallari/cache\
   --save /leonardo_work/PHD_dallari/scaling/oellm-autoexp/diffOUT/qwen3_20m_dlm/checkpoints\
   --wandb-save-dir /leonardo_work/PHD_dallari/scaling/oellm-autoexp/diffOUT/qwen3_20m_dlm/wandb\
   --tensorboard-dir /leonardo_work/PHD_dallari/scaling/oellm-autoexp/diffOUT/qwen3_20m_dlm/tensorboard\
   --save-interval 500\
   --eval-interval 500\
   --eval-iters 10\
   --log-interval 20\
   --log-throughput\
   --use-flash-attn\
   --wandb-project scaling_laws --attention-mask-type no_mask --mask-token 50257 --base-model vanilla --model-running-mode difflm-noshift --difflm-varilen-prob 0.01 --wandb-exp-name qwen3_20m_dlm --wandb-mode offline'
