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
#SBATCH --output=/leonardo_work/PHD_dallari/scaling/oellm-autoexp/diffOUT/megadlm-1.7b/megadlm-1.7b_n1_tp1_pp1_cp1_mbs4_gbs128_seq2049/slurm-%j.log
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --job-name=megadlm-1.7b_n1_tp1_pp1_cp1_mbs4_gbs128_seq2049_0

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

echo "Launching job megadlm-1.7b_n1_tp1_pp1_cp1_mbs4_gbs128_seq2049_0"
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
 --env PYTHONPATH=.:submodules/MegaDLMs \
 --env RUN_DIR=./submodules/MegaDLMs \
 --env PROJECT_DIR=. \
 --env HF_HOME=/leonardo_work/PHD_dallari/hf_cache \
 --env TOKENIZERS_PARALLELISM=false \
 --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
 --env PYTHONWARNINGS=ignore::UserWarning \
 --env NCCL_NET_GDR_LEVEL=0 \
 --env CUDA_DEVICE_MAX_CONNECTIONS=1 \
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
   --adam-beta2 0.95\
   --disable-bias-linear\
   --attention-dropout 0.0\
   --attention-softmax-in-fp32\
   --auto-detect-ckpt-format\
   --bf16\
   --no-bias-dropout-fusion\
   --no-bias-gelu-fusion\
   --no-bias-swiglu-fusion\
   --ckpt-fully-parallel-load\
   --data-cache-path /leonardo_work/PHD_dallari/cache\
   --data-path /leonardo_work/PHD_dallari/fineweb-10BT_text_document\
   --dataloader-type cyclic\
   --distributed-timeout-minutes 60\
   --eval-interval 1000\
   --eval-iters 0\
   --ffn-hidden-size 6144\
   --global-batch-size 128\
   --hidden-dropout 0.0\
   --hidden-size 2048\
   --norm-epsilon 1e-06\
   --log-interval 10\
   --log-memory-to-tensorboard\
   --log-num-zeros-in-grad\
   --log-params-norm\
   --log-progress\
   --log-throughput\
   --log-timers-to-tensorboard\
   --log-validation-ppl-to-tensorboard\
   --lr 0.0002\
   --lr-decay-iters 10000\
   --lr-decay-style WSD\
   --lr-warmup-iters 2000\
   --lr-wsd-decay-iters 0\
   --max-position-embeddings 2049\
   --micro-batch-size 4\
   --min-lr 2e-05\
   --moe-aux-loss-coeff 0.0\
   --moe-router-load-balancing-type aux_loss\
   --normalization RMSNorm\
   --num-attention-heads 16\
   --num-layers 28\
   --num-workers 8\
   --overlap-grad-reduce\
   --overlap-param-gather\
   --position-embedding-type rope\
   --qk-layernorm\
   --save-interval 1000\
   --seq-length 2049\
   --swiglu\
   --tensorboard-dir /leonardo_work/PHD_dallari/scaling/oellm-autoexp/diffOUT/megadlm-1.7b/megadlm-1.7b_n1_tp1_pp1_cp1_mbs4_gbs128_seq2049/tensorboard\
   --tokenizer-model /leonardo_work/PHD_dallari/tokenizers/tokenizer-256k\
   --tokenizer-type HuggingFaceTokenizer\
   --train-iters 10000\
   --untie-embeddings-and-output-weights\
   --use-distributed-optimizer\
   --use-flash-attn\
   --vocab-size 50289\
   --wandb-exp-name megadlm-1.7b_n1_tp1_pp1_cp1_mbs4_gbs128_seq2049\
   --wandb-project scaling\
   --wandb-save-dir /leonardo_work/PHD_dallari/scaling/oellm-autoexp/diffOUT/megadlm-1.7b/megadlm-1.7b_n1_tp1_pp1_cp1_mbs4_gbs128_seq2049/wandb\
   --weight-decay 0.1 '
