#!/bin/bash
#SBATCH --partition=caslake
#SBATCH --account=pi-chihway
#SBATCH --job-name=gaia_psf_match
#SBATCH --partition=caslake
#SBATCH --account=pi-chihway
#SBATCH --ntasks-per-node=12
#SBATCH --time=3:00:00
#SBATCH --output=results_r.out
#SBATCH --error=error_r.out
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=shrihan@uchicago.edu

source ~/.bash_profile
cd /home/shrihan/weaklens/gaia_match_psf/gaia_match_psf_code/src
conda activate weaklens2
python gaiamatch_des_r.py