import argparse

import sys
import json, time, os, sys, glob

import shutil
import warnings
import numpy as np
import random

import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

print("Built with GPU:", tf.test.is_built_with_cuda())
print("GPU device:", tf.config.list_physical_devices('GPU'))

from AttenBiRNN import AttenBiRNN
import feature_util

from pyrosetta import *
from pyrosetta.rosetta import *
init()


def parse_args( argv ):
    argv_tmp = sys.argv
    sys.argv = argv
    description = 'do protein sequence design using the MPNN model ...'
    parser = argparse.ArgumentParser( description = description )
    parser.add_argument('-pdbs', type=str, nargs='*', help='name of the input pdb file')
    parser.add_argument('-pdb_list', type=str, help='a list file of all pdb files')
    parser.add_argument('-pssm_outname', type=str, help='the output name for the pssm file')
    parser.add_argument('-output_path', type=str, default='./', help="the path for the outputs")
    args = parser.parse_args()
    sys.argv = argv_tmp

    return args

AA = [ 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V' ]
#bias = np.array([ 0.551,  0.993,  5.621,  2.135,  0.427, 14.592,  0.548,  0.571,
#        3.749,  1.721,  0.85 ,  1.21 ,  2.51 ,  2.336,  3.126,  1.462,
#        1.084,  4.346,  1.866,  0.729])

args = parse_args( sys.argv )

if args.pdbs == None:
    assert (args.pdb_list != None)
    with open(args.pdb_list) as f:
        all_pdbs = [line.strip() for line in f]
else:
    all_pdbs = args.pdbs

def init_seq_optimize_model():

    # model configs
    model_weights_path = './weights/profile_prediction_weights'

    ML_model = AttenBiRNN(False, # is_train
                1, # batch_size
                20, # n_1d_layer
                12, # n_2d_layer
                [1,2,4,8], #dilation
                64, # n_feat_1d
                32, # n_bottle_1d
                128, # n_feat_2d
                64, # n_bottle_2d
                64, #n_hidden_rnn
                50, # FLAGS.attention
                3, # FLAGS.kernel
                0.2, # FLAGS.p_dropout
                0.0005, # FLAGS.l2_coeff
                False, # FLAGS.use_fragment_profile
                False, # FLAGS.train_pssm
                )

    if ML_model.load(model_weights_path):
        return ML_model
    else:
        print("Model params loading error!")
        exit(0)

# the deep learning model
model = init_seq_optimize_model()

results = []

for pdb in all_pdbs:

    if pdb.endswith('.pdb'):
        tag = pdb.split('/')[-1][:-4]
    if pdb.endswith('.pdb.gz'):
        tag = pdb.split('/')[-1][:-7]

    pose = pose_from_file(pdb)
    dssp = core.scoring.dssp.Dssp(pose)
    for ires in range(1, pose.size()+1):
        res = pose.residue(ires)
        for at in ['N','CA','C','O']:
            xyz = res.xyz(at)
            try:
                coords = np.vstack([coords, np.asarray([xyz.x, xyz.y, xyz.z])])
            except NameError:
                coords = np.array([[xyz.x, xyz.y, xyz.z]])
    
    coords = coords.reshape(-1, 4, 3)
    feat1d, feat2d = feature_util.extract_features(coords, dssp.get_dssp_reduced_IG_as_L_secstruct())
        # batch size 1, so take the first one
    probs = model(feat1d[None,...], feat2d[None,...]).numpy()[0]

    # Length of chain A
    L = len(pose.split_by_chain()[1])

    # Get the probability matrix just for chain A
    prob_cA = probs[:L]

    # Data from BLOSUM62 ncbi-blast-2.6.0+-src/c++/src/algo/blast/composition_adjustment/matrix_frequency_data.c
    bg_freqs = np.array([7.4216205067993410e-02, 5.1614486141284638e-02, 4.4645808512757915e-02,
                         5.3626000838554413e-02, 2.4687457167944848e-02, 3.4259650591416023e-02,
                         5.4311925684587502e-02, 7.4146941452644999e-02, 2.6212984805266227e-02,
                         6.7917367618953756e-02, 9.8907868497150955e-02, 5.8155682303079680e-02,
                         2.4990197579643110e-02, 4.7418459742284751e-02, 3.8538003320306206e-02,
                         5.7229029476494421e-02, 5.0891364550287033e-02, 1.3029956129972148e-02,
                         3.2281512313758580e-02, 7.2919098205619245e-02])
    bg_freqs = bg_freqs / bg_freqs.sum()

    # Compute log-odds
    pssm = np.log(prob_cA/bg_freqs)
    
    with open(args.pssm_outname, 'w') as f_out:
        f_out.write('\n')
        f_out.write('Last position-specific scoring matrix computed, weighted observed percentages rounded down, information per position, and relative weight of gapless real matches to pseudocounts\n')
        f_out.write('            A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V   A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V\n')

        for i, odds in enumerate(pssm):
            aa = seq[i]
            pos = str(i+1)
            odds_str = ' '.join([str(x) for x in pssm[i]])
            occ_str = ' '.join([str(x) for x in prob_cA[i]])
            f_out.write(pos+' '+aa+' '+odds_str+' '+occ_str+' 0.00 0.00'+'\n')
        f_out.write('\n\n\n\n')

