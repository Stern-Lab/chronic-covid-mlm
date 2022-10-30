import argparse
import os
import sys
import json
import glob

sys.path.append('/sternadi/home/volume3/chronic-corona-pred/repos/chronic-covid-pred/')
sys.path.append("/sternadi/home/volume1/daniellem1/SternLab/")
from pbs_runners import script_runner


parser = argparse.ArgumentParser(
    description="Grep fasta sequences from GISAID by IDs file",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('-i', '--ids', type=str, required=True, help="the file containing the ids to search")
parser.add_argument('-o', '--output-dir', type=str, defualt='./', help="the file containing the ids to search")
args = parser.parse_args()

ids_txt = args.ids
ids_name = os.path.basename(ids_txt).replace('.txt', '')
with open(ids_txt, 'r') as o:
    sequences = o.readlines()

JSON_PATH = '/sternadi/home/volume3/chronic-corona-pred/data/GISAID/sequence_mapper.json'
with open(JSON_PATH) as o:
    mapper = json.load(o)

tmp_dir = os.path.join(args.output_dir, f'tmp_{ids_name}')
os.makedirs(tmp_dir, exist_ok=True)

files = [os.path.join('/sternadi/home/volume3/chronic-corona-pred/data/GISAID/fasta_parts/',mapper[seq.strip()]) for
         seq in sequences]

cmds_seqkit = '\n'.join([f"/sternadi/home/volume3/chronic-corona-pred/software/seqkit grep -f "
                  f"{ids_txt} {f} > {os.path.join(tmp_dir, os.path.basename(f).replace('.txt', ''))}" for f in files])

cmds_wrap = f"cat {tmp_dir}/*.fasta > {os.path.join(args.output_dir, f'{ids_name}')}.fasta\nrm -r {tmp_dir}/"

job_id = script_runner(cmds_seqkit, queue='adistzachi', alias='seqkit')
_ = script_runner(cmds_wrap, queue='adistzachi', alias='seqkit_warp', run_after_job=job_id)


