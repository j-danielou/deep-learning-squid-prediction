#!/bin/bash

#SBATCH --job-name=opt_squid
#SBATCH --output=/marbec-data/Osmose-Montpellier/Lou-Jules/logs/job_%j.out
#SBATCH --error=/marbec-data/Osmose-Montpellier/Lou-Jules/logs/job_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 10

mkdir -p /marbec-data/Osmose-Montpellier/Lou-Jules/logs

python optimize_cluster.py

