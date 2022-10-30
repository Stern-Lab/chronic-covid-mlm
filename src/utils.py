import itertools
import numpy as np
import pandas as pd
import networkx as nx


def create_mutational_graph(candidate, max_reversions=3, max_days=30, add_weights=False):
    # create node mapping
    n = candidate.shape[0]
    seq2aa = candidate.sort_values(by='date').reset_index(drop=True)[
        ['seqid', 'date', 'datenum', 'aa_from_ref', 'num_aa_from_ref']].to_dict(orient='index')
    all_pairs = list(itertools.combinations(range(n), 2))
    nodes = [(i, {'label': (seq2aa[i]['date'], seq2aa[i]['seqid']), 'pos': (seq2aa[i]['datenum'], seq2aa[i]['num_aa_from_ref'])}) for i in seq2aa]

    # build the DAG
    G = nx.DiGraph()
    reversions = np.zeros((n, n))
    counts = np.zeros((n, n))
    deltas = np.zeros((n, n))
    for i, j in all_pairs:
        inter = seq2aa[i]['aa_from_ref'].intersection(seq2aa[j]['aa_from_ref'])
        reversions[i, j] = len(seq2aa[i]['aa_from_ref'] - seq2aa[j]['aa_from_ref'])
        counts[i, j] = len(seq2aa[j]['aa_from_ref'] - inter)
        deltas[i, j] = (seq2aa[j]['date'] - seq2aa[i]['date']).days


    G.add_nodes_from(nodes)

    for i, j in all_pairs:
        not_same_date = G.nodes[i]['label'][0] != G.nodes[j]['label'][0]
        if reversions[i, j] <= max_reversions and deltas[i, j] <= max_days and not_same_date:
            if add_weights:
                G.add_edge(i, j, label=seq2aa[j]['date'] - seq2aa[i]['date'], weight=counts[i,j])
            else:
                G.add_edge(i, j, label=seq2aa[j]['date'] - seq2aa[i]['date'])
    return G

def get_time_interval_from_component(G, component_nodes):
    node_mapper = nx.get_node_attributes(G, 'label')
    time_interval = node_mapper[max(component_nodes)][0] - node_mapper[min(component_nodes)][0]
    return time_interval

def get_longest_path_per_component(G, time_interval='20 days'):
    # get all connected components
    components = list(nx.connected_components(G.to_undirected()))
    df = pd.DataFrame([tuple(components)]).T
    df = df.rename(columns={0: 'component'})

    df['subgraph'] = df['component'].apply(lambda x: G.subgraph(nodes=x))
    df['longest_path'] = df['subgraph'].apply(lambda g: nx.algorithms.dag_longest_path(g))
    df['len_longest_path'] = df['longest_path'].map(len)
    df['component_time_interval'] = df.apply(lambda row: get_time_interval_from_component(
        row['subgraph'], row['component']), axis=1)
    df['component_seqids'] = df['subgraph'].apply(
        lambda g: ';'.join([x[1] for x in nx.get_node_attributes(g, 'label').values()]))
    df['longest_path_seqids'] = df['longest_path'].apply(lambda p: ';'.join([G.nodes[node]['label'][1] for node in p]))

    filtered = df[df['component_time_interval'] >= time_interval][['component_time_interval',
                                                               'component_seqids', 'longest_path_seqids', 'len_longest_path']]
    return filtered



