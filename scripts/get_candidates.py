import argparse
import sys

sys.path.append('/sternadi/home/volume3/chronic-corona-pred/repos/chronic-covid-pred/')

from src.Clade import Clade
from src.Candidates import KnownCandidates

parser = argparse.ArgumentParser(
    description="Get candidate data by clade",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--clade', type=str, required=True, help="clade name")
args = parser.parse_args()

clade = args.clade

c = Clade(clade_path=f'/sternadi/home/volume3/chronic-corona-pred/nextclade/subsets/{clade}/', ref_path='/sternadi/home/volume3/chronic-corona-pred/data/clades_gts.json', clade=clade,out=f'/sternadi/home/volume3/chronic-corona-pred/nextclade/subsets/{clade}/candidates/')

c.calc_mutations_from_ref()
c.fit_OLS(by=['num_aa_from_ref', 'num_insertions', 'num_deletions'])
c.filter_extremes()
c.save_extremes()
c.set_known_candidates()

known = KnownCandidates(clade=c,mut_path='/sternadi/home/volume3/chronic-corona-pred/data/GISAID/sample_path_mapper.pkl', mutation_to_consider=['num_aa_from_ref', 'num_insertions', 'num_deletions'])
known.get_stats()
known.extract_potential_chronic()
known.summarize()
known.save_candidate_data()
known.plot_manhattan()
