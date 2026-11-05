"""Shared small-model features, computed from the complete prefix, never labels."""
import numpy as np
from rc3_nav_value.r1.policy import State

NAMES = ['gain_frac','gain_count','scarcity','distance','home','ratio','left_gain',
 'right_gain','overlap_max','overlap_mean','complement','menu_D_min','menu_D_mean',
 'menu_D_max','menu_euclidean_mean','detour_mean','ref_flag','ref_gain_diff',
 'ref_distance_diff','ref_ratio_diff','ref_home_diff','covered_frac','prefix_count',
 'prefix_length','prefix_return','prefix_edge_mean','prefix_edge_max',
 'prefix_weighted_edges','prefix_turn','prefix_x_weighted','prefix_y_weighted',
 'remaining_count','legal_fraction','x_relative','y_relative','theta_cos','theta_sin',
 'gain_of_uncommon_tasks']


def build(item, B, prefix, menu):
    s = State(item, prefix)
    C = item.C[s.remaining]
    legal = np.flatnonzero(s.legal_mask())
    degree = C[:, legal].sum(1)
    gain = C.sum(0)
    scarcity = (C/degree[:, None]).sum(0)
    ref = next(a for a in B if s.legal_mask()[a-1])
    D = item.D
    ids = np.asarray(menu)-1
    Cm = C[:, ids]
    scene = item.base.scene
    xy = np.asarray([[c['x'],c['y']] for c in scene['candidates']])
    home = np.asarray(scene['start'][:2])
    prefix_nodes = [0, *prefix]
    edges = np.asarray([D[a,b] for a,b in zip(prefix_nodes,prefix_nodes[1:])])
    weights = np.arange(1,len(prefix)+1,dtype=float)
    if len(prefix):
        wxy = (xy[np.asarray(prefix)-1]*weights[:,None]).sum(0)/weights.sum()-home
        points = np.vstack([home,xy[np.asarray(prefix)-1]])
        diff = np.diff(points,axis=0)
        turn = float(np.abs(np.diff(np.arctan2(diff[:,1],diff[:,0]))).sum())
    else:
        wxy = [0.,0.]; turn = 0.
    n = len(item.C)
    ratio = gain/(D[s.current,1:]+.75)
    out = []
    for a in menu:
        j = a-1
        links = D[a,np.asarray(menu)]
        eu = np.linalg.norm(xy[ids]-xy[j],axis=1)
        overlap = (Cm & C[:,j,None]).sum(0)
        arm = item.base.F[s.remaining,j,:].sum(0)
        uncommon = int((C[:,j] & (degree <= 3)).sum())
        c = scene['candidates'][j]
        row = [gain[j]/max(n,1),gain[j]/10,scarcity[j],D[s.current,a]/10,D[a,0]/10,
          ratio[j],arm[0]/10,arm[1]/10,overlap.max()/max(n,1),overlap.mean()/max(n,1),
          (len(C)-gain[j])/max(n,1),links.min()/10,links.mean()/10,links.max()/10,
          eu.mean()/10,(links-eu).mean()/10,float(a==ref),(gain[j]-gain[ref-1])/10,
          (D[s.current,a]-D[s.current,ref])/10,ratio[j]-ratio[ref-1],(D[a,0]-D[ref,0])/10,
          1-len(C)/max(n,1),len(prefix)/100,s.rawL/100,D[s.current,0]/10,
          edges.mean()/10 if len(edges) else 0.,edges.max()/10 if len(edges) else 0.,
          float(edges@weights/weights.sum())/10 if len(edges) else 0.,turn/100,
          wxy[0]/10,wxy[1]/10,len(C)/100,len(legal)/max(item.C.shape[1],1),
          (xy[j,0]-home[0])/10,(xy[j,1]-home[1])/10,np.cos(c['theta']),np.sin(c['theta']),uncommon/10]
        out.append(row)
    values = np.asarray(out,dtype=np.float32)
    if values.shape != (len(menu),len(NAMES)) or not np.isfinite(values).all():
        raise ValueError('Invalid feature matrix')
    return values
