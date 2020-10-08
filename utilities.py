from libraries import *

def priority_edges(edge_index,user):
    aa = np.where(np.array(edge_index[:, edge_mask.argsort()[:]][0])==user)
    bb = edge_index[:, edge_mask.argsort()[:]][:,aa]
    
    return bb[:,0]

def sorted_indexes(nz_indexes):
    
    indices = [node_feat_mask.argsort().tolist().index(i) for i in tqdm(nz_indexes)]
    prio_nz_indices = [i[1] for i in sorted(list(zip(indices,nz_indexes)))]
    
    return prio_nz_indices

