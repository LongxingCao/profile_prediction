import sys
import os
import numpy as np

import scipy.spatial
from sklearn.neighbors import KDTree

ss2idx = {'L':0, 'H':1, 'E':2}

def psudo_CB(N, CA=None, C=None):

    # assume the first
    if CA is None:
        N, CA, C = N[...,0,:], N[...,1,:], N[...,2,:]
    #
    # calc the position of the CB atom
    b = CA - N
    c = C - CA
    a = np.cross(b, c)
    CB = -0.58273431*a + 0.56802827*b - 0.54067466*c + CA

    return CB
def get_dihedrals(a, b, c, d, degrees=False):
    b0 = -1.0*(b-a)
    b1 = c - b
    b2 = d - c

    b1 /= np.linalg.norm(b1, axis=-1)[:, None]

    v = b0 - np.sum(b0*b1, axis=-1)[:,None]*b1
    w = b2 - np.sum(b2*b1, axis=-1)[:,None]*b1

    x = np.sum(v*w, axis=-1)
    y = np.sum(np.cross(b1, v)*w, axis=-1)

    if degrees:
        return np.degrees( np.arctan2(y,x) )
    else:
        return np.arctan2(y, x)

def get_chain_break( conformation, amide_bond_distance_cutoff = 2.5 ):
    #
    C = conformation[:-1,2,:]
    N = conformation[1:,0,:]
    chain_break = np.linalg.norm(N-C, axis=1) > amide_bond_distance_cutoff

    return chain_break

def get_torsions(conformation,
                 output_degree = False,
                 rosetta_convention = False,
                 nan_for_break = False):
    
    # length*[N,CA,C,O]*[x,y,z]
    assert(conformation.ndim == 3)

    chain_break = get_chain_break(conformation)
    fill_val = np.nan if nan_for_break else 0.0

    phi = get_dihedrals(conformation[:-1,2,:], conformation[1:,0,:], conformation[1:,1,:], conformation[1:,2,:])
    psi = get_dihedrals(conformation[:-1,0,:], conformation[:-1,1,:], conformation[:-1,2,:], conformation[1:,0,:])
    omega = get_dihedrals(conformation[:-1,1,:], conformation[:-1,2,:], conformation[1:,0,:], conformation[1:,1,:])

    phi[chain_break] = fill_val
    psi[chain_break] = fill_val
    omega[chain_break] = fill_val

    #
    phi = np.hstack([[fill_val],phi])
    psi = np.hstack([psi,[fill_val]])
    if rosetta_convention:
        omega = np.hstack([omega, [fill_val]])
    else:
        omega = np.hstack([[fill_val], omega])

    if output_degree:
        phi = np.degrees(phi)
        psi = np.degrees(psi)
        omega = np.degrees(omega)

    return phi, psi, omega

def sidechain_neighbors(conformation, 
                        mask = None,
                        dist_midpoint = 9.0,
                        dist_exponent = 1.0,
                        angle_shift_factor = 0.5,
                        angle_exponent = 2.0):

    # check the dimension
    assert(conformation.ndim == 3 and (mask is None or mask.ndim == 1) )

    if mask is None:
        mask = np.ones(conformation.shape[0], dtype=np.bool)
    elif mask.dtype != np.bool:
        mask = mask.astype(np.bool)
    else:
        pass

    CA = conformation[:,1,:]
    CB = psudo_CB(conformation)

    sc_vector = CB[mask] - CA[mask]
    sc_vector /= np.linalg.norm(sc_vector, axis=-1)[:, None]

    # calc the weighted sidechain neighbors
    vect = CB[None,:,:] - CB[mask][:,None,:]


    dist_term = 1.0/(1.0+np.exp(dist_exponent*(np.linalg.norm(vect, axis=-1)-dist_midpoint)))
    vect /= (np.linalg.norm(vect, axis=-1)[:,:,None] + 1e-6)
    angle_term = (np.sum(sc_vector[:,None,:] * vect, axis=-1) + angle_shift_factor)/(1+angle_shift_factor)
    angle_term = np.clip(angle_term, 0.0,None)
    id0 = np.arange(np.count_nonzero(mask))
    id1 = np.hstack(np.argwhere(mask==True))
    angle_term[id0, id1] = 0.0
    return np.sum( dist_term * np.power(angle_term, angle_exponent), axis=-1 )


def neighbors(conformation, atom = 'CA', cutoff=10.0):
    if type(atom) == int:
        pass
    elif type(atom) == str:
        allowed_atoms = ['N', 'CA', 'C', 'O', 'CB']
        assert(atom in allowed_atoms)
        atom = allowed_atoms.index(atom)
    else:
        print("Unknown type of atom: ", type(atom))
        exit(0)

    coords = conformation[:, atom, :]
    tree = KDTree(coords)

    counts = tree.query_radius(coords,r=cutoff,count_only=True)

    return counts - 1

class DistOriMap:
    def __init__(self, dmax=20.0): 
        self.dmax = dmax
        return
    
    #
    def get_dihedrals(self, a, b, c, d):
        b0 = -1.0*(b-a)
        b1 = c - b
        b2 = d - c

        b1 /= np.linalg.norm(b1, axis=-1)[:, None]

        v = b0 - np.sum(b0*b1, axis=-1)[:,None]*b1
        w = b2 - np.sum(b2*b1, axis=-1)[:,None]*b1

        x = np.sum(v*w, axis=-1)
        y = np.sum(np.cross(b1, v)*w, axis=-1)

        return np.arctan2(y, x)
    #
    def get_angles(self, a, b, c):
        v = a - b
        v /= np.linalg.norm(v, axis=-1)[:, None]

        w = c - b
        w /= np.linalg.norm(w, axis=-1)[:, None]

        x = np.sum(v*w, axis=1)

        return np.arccos(x)

    # given a pose, compute the dist&tor map, not considering the gaps
    def get_dist_ori_maps( self, conformation ): # [N, CA, C, O]

        n_res = conformation.shape[0]
        N  = conformation[...,0,:]
        CA = conformation[...,1,:]
        C  = conformation[...,2,:]
        O  = conformation[...,3,:]
        #

        # use the virtual CB atom, since sequence design is sequence agnostic
        CB = psudo_CB(N, CA, C)

        #
        dist_CB_CB = scipy.spatial.distance.cdist(CB, CB, metric='euclidean')
        dist_N_O   = scipy.spatial.distance.cdist(N, O, metric='euclidean')
        dist_O_N = dist_N_O.T

        #
        dist_maps = np.stack((dist_CB_CB, dist_N_O, dist_O_N), axis=-1)

        #
        # neighbor search. I would still do masking, so this is just to save computational time.
        kdCB = scipy.spatial.cKDTree(CB)
        indices = kdCB.query_ball_tree(kdCB, self.dmax)
        assert( len(indices) == n_res )
        #
        # indices of contacting residues
        idx = np.array([[i,j] for i in range(len(indices)) for j in indices[i] if i != j]).T
        idx0 = idx[0]
        idx1 = idx[1]
        #
        # matrix of Ca-CB-CB-Ca dihedrals
        omega6d_c = np.zeros((n_res, n_res))
        omega6d_s = np.zeros((n_res, n_res))
        ang = self.get_dihedrals(CA[idx0], CB[idx0], CB[idx1], CA[idx1])
        omega6d_c[idx0, idx1] = (np.cos(ang)+1.0) / 2.0 # range 0~1
        omega6d_s[idx0, idx1] = (np.sin(ang)+1.0) / 2.0 # range 0~1
        #
        # matrix of polar coord theta
        theta6d_c = np.zeros((n_res, n_res))
        theta6d_s = np.zeros((n_res, n_res))
        ang = self.get_dihedrals(N[idx0], CA[idx0], CB[idx0], CB[idx1])
        theta6d_c[idx0, idx1] = (np.cos(ang)+1.0) / 2.0 # range 0~1
        theta6d_s[idx0, idx1] = (np.sin(ang)+1.0) / 2.0 # range 0~1
        #
        # matrix of polar coord phi
        phi6d_c = np.zeros((n_res, n_res))
        phi6d_s = np.zeros((n_res, n_res))
        ang = self.get_angles(CA[idx0], CB[idx0], CB[idx1])
        phi6d_c[idx0, idx1] = (np.cos(ang)+1.0) / 2.0 # range 0~1
        phi6d_s[idx0, idx1] = (np.sin(ang)+1.0) / 2.0 # range 0~1
        #
        ori_maps = np.stack([omega6d_s, theta6d_s, phi6d_s, omega6d_c, theta6d_c, phi6d_c], axis=-1)
        
        return dist_maps, ori_maps

# distance scaling
def dist_SP_scaling( X, d0=4.0):
    return 2/(1+np.maximum(X,d0)/d0)
def dist_arcsinh_scaling(X, cutoff=6.0, scaling=3.0):
        X_prime = np.maximum(X, np.zeros_like(X)+cutoff) - cutoff
        return np.arcsinh(X_prime)/scaling
# Warning!!!!!!!!!!!!! 
# The pixels corresponding to the missing regions are zero.
# Fix me!!!!
def dist_clipping_scaling(X, lower_bound=0.0, upper_bound=15.0, reverse=True):
    X_prime = ( np.clip(X, lower_bound, upper_bound) - lower_bound ) / ( upper_bound - lower_bound )
    if reverse:
        return 1 - X_prime
    else:
        return X_prime

def extract_features( coords, dssp_str ):

    #
    # 1-d features
    sc_neighbors = sidechain_neighbors(coords)
    ca_neighbors = neighbors(coords, atom="CA", cutoff=10.0)

    #
    # dist and orientation map
    o = DistOriMap(dmax=20.0)
    dist_maps, ori_maps = o.get_dist_ori_maps( coords )
    dist_maps = dist_SP_scaling( dist_maps, d0=4.0)
    
    #
    # dssp
    dssp_np = np.array( [ ss2idx[ii] for ii in dssp_str ] )
    dssp_np = np.eye(3)[dssp_np]

    #
    # tors
    phis, psis, omegas = get_torsions(coords, False, False, True)
    tors = np.vstack([
            (np.sin(phis).T+1.0)/2.0, # range 0-1
            (np.cos(phis).T+1.0)/2.0, # range 0-1
            (np.sin(psis).T+1.0)/2.0, # range 0-1
            (np.cos(psis).T+1.0)/2.0, # range 0-1
            (np.sin(omegas).T+1.0)/2.0,
            (np.cos(omegas).T+1.0)/2.0
        ]).T
    #
    # This is best way to do this .....................................................
    # If I assign all None to 0.0 before sin/cos, the final values would be bad..................
    np.nan_to_num(tors, copy=False, nan=0.0)

    #
    feat1d = np.vstack([dssp_np.T, tors.T, sc_neighbors.T, ca_neighbors.T]).T
    #
    feat2d = np.concatenate([dist_maps, ori_maps], axis=-1)
    
    return feat1d, feat2d
