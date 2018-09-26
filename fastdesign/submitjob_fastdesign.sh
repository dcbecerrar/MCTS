#!/bin/bash
#SBATCH -t 0-02:00
#SBATCH --cpus-per-task=1
#SBATCH --mem 2G
#SBATCH --account=rrg-pmkim
#SBATCH --output="backonly_fastdesign-%A_%a.out"
#SBATCH --array=0-999

DIR="/home/wuzhen3/scratch/MCTS_THESIS/5ggs_relaxed/frag_full_constructs_phe/run1_disres_rand1000"
FILES=($(ls ${DIR}))

#rosetta_scripts.linuxgccrelease -s $DIR/${FILES[$SLURM_ARRAY_TASK_ID]} -nstruct 100 -parser:protocol fastdesign.xml -packing:resfile tigit_design_99-111.resfile 

relax.default.linuxgccrelease -s $DIR/${FILES[$SLURM_ARRAY_TASK_ID]} -relax:constrain_relax_to_start_coords -relax:respect_resfile -resfile 5ggs_relaxed_phe_design.resfile -packing:ex1:level 4 -ex2 -nstruct 2



