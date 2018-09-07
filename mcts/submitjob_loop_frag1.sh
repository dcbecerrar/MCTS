#!/bin/bash
#SBATCH -t 0-02:00
#SBATCH --cpus-per-task=16
#SBATCH --mem 4000
#SBATCH --account=rrg-pmkim
#SBATCH --output="5ggs_relaxed_phe_frag1-%A_%a.out"


#parallel < commands.txt
python mcts_frag.py --levels 15 --num_sims 16 --num_rand 4


