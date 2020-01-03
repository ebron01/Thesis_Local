#!/bin/bash
#SBATCH --nodes=1
#SBATCH --workdir=/home/eboran/Thesis_Local/adv-inf-master/  		# working dir
#SBATCH --gres=gpu:3  					# kac gpu reserve edilecek
#SBATCH --output=/home/eboran/Thesis_Local/adv-inf-master/slurm-%j.out	# ciktilarin yazilacagi dosya
#SBATCH --error=/home/eboran/Thesis_Local/adv-inf-master/slurm-%j.err	# hatalarin yazilacagi dosya
#SBATCH --time=1-00:00:00				# isin max calisma zamani. han icin 1 gun, hanabi icin 2 gun.
#SBATCH --mail-type=ALL        # send mail when process begins
#SBATCH --mail-user=eboran01@gmail.com

########### Cuda path leri, bunlari silebilirsiniz #############
####export PATH=/home/samet/cuda-9.0/bin:$PATH
####export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/samet/cuda-9.0/lib64
#########################################################

CUDA_LAUNCH_BLOCKING=1 python train.py --g_pre_nepoch 22 --g_start_from 'save_e15_lange1_b_32_0101' --g_start_epoch 'latest'
