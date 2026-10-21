#!/bin/bash
#SBATCH --account=PHD_dallari
#SBATCH --error=%j.err
#SBATCH --output=%j.out
#SBATCH --partition=lrd_all_serial
#SBATCH --job-name=creating_container
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --mem=30800

export TMPDIR=/scratch_local/
export APPTAINER_CACHEDIR=$CINECA_SCRATCH
export APPTAINER_TMPDIR=$CINECA_SCRATCH
export SINGULARITY_TMPDIR=$CINECA_SCRATCH
export SINGULARITY_CACHEDIR=$CINECA_SCRATCH
cd $WORK
mkdir -p containers
cd containers
apptainer build megatronDLM.sif /leonardo_work/PHD_dallari/scaling/oellm-autoexp/container/megatron/MegatronTraining-leonardo.def.in