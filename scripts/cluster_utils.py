import os

genes = ['ORF1a','ORF3a', 'ORF9b','ORF1b',
           'N', 'S', 'E', 'ORF8', 'M',
           'ORF7a', 'ORF7b', 'ORF6']

def submit(cmdfile):
    cmd = "/opt/pbs/bin/qsub " + cmdfile
    result = os.popen(cmd).read()
    return result.split(".")[0]

def create_sh(dataset, noised=False, queue='gpu'):
    gene = os.path.basename(dataset).split('.txt')[0]
    save_prefix = f'/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/models/{gene}/lm_{gene}'
    save_path = f'/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/gpu_logs/{gene}/'
    alias = f'mlm_64_{gene}'
    tags = f'mlm,{gene},gpu'
    add_noise = ''
    if noised:
        tags += ',noised'
        alias += '_noised'
        add_noise += '--noise'
        save_prefix = f'/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/models/{gene}/noised/lm_{gene}'
        save_path = f'/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/gpu_logs/{gene}/noised/'
    os.makedirs(save_prefix, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)


    cmds = f'''#!/bin/bash
#PBS -S /bin/bash
#PBS -j oe
#PBS -r y
#PBS -q {queue}
#PBS -v PBS_O_SHELL=bash,PBS_ENVIRONMENT=PBS_BATCH
#PBS -N {alias}
#PBS -o /sternadi/home/volume1/daniellem1/gpu_tests
#PBS -e /sternadi/home/volume1/daniellem1/gpu_tests
#PBS -l select=1:ncpus=1:ngpus=1
id
date
hostname

source ~/.bashrc
conda activate mlm
export PATH=$CONDA_PREFIX/bin:$PATH

python /sternadi/home/volume1/daniellem1/chronic_covid/mt_lm/trainer.py --dataset {dataset} --batch-size 64 --tags {tags} --save-prefix {save_prefix} {add_noise}
date'''

    with open(os.path.join(save_path, 'run.sh'), 'w') as o:
        o.write(cmds)
