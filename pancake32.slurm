#!/bin/bash
#SBATCH --error=err-%j.err
#SBATCH --output=prod-%j.out
#SBATCH --partition=snowy
#SBATCH --time=0:10:00
#SBATCH --nodes=4
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=8G
#SBATCH --job-name=ilovk-kvoli
module load gcc/8.3.0
module load openmpi/3.1.4

echo pancake 32
mpicxx -o multi-astar-pancake multi-astar-pancake.cpp -O3
sh tester.sh 32
