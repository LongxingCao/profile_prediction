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
    parser.add_argument("-omit_AAs", type=list, help="Specify which amino acids should be omitted in the generated sequence, e.g. 'AC' would omit alanine and cystine.")
    parser.add_argument('-output_path', type=str, default='./', help="the path for the outputs")
    parser.add_argument('-cum_prob', type=float, default=0.6, help='the cutoff value of the accumulative probability')
    parser.add_argument('-max_aas_per_pos', type=int, default=20, help='the maximum allowed amino acids at each residue position')
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

    # np.savez_compressed(f"{tag}.npz", pssm=probs)


    if args.omit_AAs != None:
        for aa in  args.omit_AAs:
            probs[:,AA.index(aa)] = 0
    prob_sums = np.sum( probs, axis=1 ) + 1e-9
    probs = probs / prob_sums[:,None]

    # sorting
    inx_sort = np.argsort(probs, axis=1)[:,::-1]
    probs_sorted = np.sort(probs, axis=1)[:,::-1]
    cumsum = np.cumsum(probs_sorted, axis=1)

    resfile = "NATAA\nSTART\n"
    for idx,v in enumerate(np.argmax(cumsum>=args.cum_prob, axis=1)):
        good = []
        v = min(v, args.max_aas_per_pos-1) # use the smaller value
        for aa_idx in inx_sort[idx,:v+1]:
            good.append(AA[aa_idx])
        resfile += "{} A PIKAA {}\n".format(idx+1, "".join(good))

    output_res_file = os.path.join(args.output_path,'{}.resfile'.format(tag))
    print("Save the resfile to ", output_res_file)

    with open(output_res_file, 'w') as f:
        f.write(resfile)


