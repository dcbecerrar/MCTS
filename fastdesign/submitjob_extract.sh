#!/bin/bash
#SBATCH -t 0-04:00
#SBATCH --cpus-per-task=1
#SBATCH --mem 2G
#SBATCH --account=rrg-pmkim

python extract_seq.py
