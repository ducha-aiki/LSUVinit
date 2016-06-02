#!/usr/bin/env python
from __future__ import print_function 
import os
import sys

class bcolors:
    LINE = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


for arg in sys.argv:
    if  arg == 'help' or arg == 'HELP' or arg == '-help' or arg == '--help' or arg == '-h' or arg == '--h' or arg == '/h' or arg == '/help' or arg == '/H' or arg == '/HELP':
        print (bcolors.LINE + """____________________________________________

""" + bcolors.ENDC + """By $CAFFE_ROOT we mean Caffe's installation folder.
Place this script into """+ bcolors.OKGREEN +"""$CAFFE_ROOT/tools/extra/"""+ bcolors.ENDC +"""

Use it as:
    """+ bcolors.OKGREEN +"""python $CAFFE_ROOT/tools/extra/lsuv-init.py /path/to/solver.prototxt /path/to/initialised.caffemodel LSUV
"""+ bcolors.ENDC +"""or
    """+ bcolors.OKGREEN +"""python $CAFFE_ROOT/tools/extra/lsuv-init.py /path/to/solver.prototxt /path/to/initialised.caffemodel Orthonormal noFetch gpu
"""+ bcolors.ENDC +"""
"""+ bcolors.OKBLUE +"""initialised.caffemodel"""+ bcolors.ENDC +""" is where the initialised model will be saved to. If such file already exists, it will be loaded and the initialisation distortion will be applied to it instead.
"""+ bcolors.OKBLUE +"""noFetch"""+ bcolors.ENDC +""" is an optional parameter for not loading existing "initialised.caffemodel" file.

It's highly recommended to """+ bcolors.BOLD + bcolors.UNDERLINE +"""USE LARGE BATCHES"""+ bcolors.ENDC +""" - set them in appropriate *.prototxt - when running LSUV. Obviously, the more different your data is, the bigger the need for larger batches. For 99% of us large batches are easier to get on the CPU using RAM and swapping, which is why CPU is the default platform for computing LSUV.
"""+ bcolors.OKBLUE +"""gpu"""+ bcolors.ENDC +""" is an optional parameter for computing on GPU (the first one of them - "device #0" - if you have several) instead of CPU. You will be limited by your GPU's ram size then, but the LSUV init computation is likely to finish much faster.

"""+ bcolors.OKBLUE +"""LSUV"""+ bcolors.ENDC +""" scientific paper can be found at http://arxiv.org/abs/1511.06422
"""+ bcolors.OKBLUE +"""Orthonormal"""+ bcolors.ENDC +""" is a different initialisation type, which is pretty cool too. http://arxiv.org/abs/1312.6120

"""+ bcolors.BOLD + bcolors.FAIL +"""NOTE!"""+ bcolors.ENDC +"""
* stands for anything
Name your """+ bcolors.WARNING +"""activation layers"""+ bcolors.ENDC +""" as """+ bcolors.OKBLUE +"""*_act*"""+ bcolors.ENDC +""", or  """+ bcolors.OKBLUE +"""*_ACT*"""+ bcolors.ENDC +"""
Name your """+ bcolors.WARNING +"""batch normalization layers"""+ bcolors.ENDC +""" as """+ bcolors.OKBLUE +"""*BN*"""+ bcolors.ENDC +""", or """+ bcolors.OKBLUE +"""*bn*"""+ bcolors.ENDC +"""
- so that the script wouldn't try to process stuff like """+ bcolors.WARNING +"""PReLU activation layers"""+ bcolors.ENDC +""" and get """+ bcolors.FAIL +"""stuck"""+ bcolors.ENDC +""". This algorithm can only process fully-connected and convolutional layers. Not their activations.
(That doesn't mean that you can't use PReLU. Just name them as *_act*)

"""+ bcolors.LINE +"""____________________________________________
""" + bcolors.ENDC)
        sys.exit()


from pylab import *
import random
import numpy as np
caffe_root_dir=os.path.dirname(os.path.realpath(__file__))
caffe_root_dir+='/../../python'
sys.path.insert(0, caffe_root_dir)
import caffe
# Orthonorm init code is taked from Lasagne
# https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py

def svd_orthonormal(shape):
    if len(shape) < 2:
        raise RuntimeError("Only shapes of length 2 or more are supported.")
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = standard_normal(flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return q

noFetch = False
for arg in sys.argv:
    if  arg == 'noFetch':
        noFetch = True

    
if __name__ == '__main__':
    if len (sys.argv) < 4:
        raise RuntimeError('Usage: python ' + sys.argv[0] + ' path_to_solver path_to_save_model mode')
    solver_path = str(sys.argv[1])
    init_path = str(sys.argv[2])
    init_mode =  str(sys.argv[3])
    margin = 0.02;
    max_iter = 20;
    needed_variance = 1.0
    var_before_relu_if_inplace=True
    mode_check=False;  
    if init_mode == 'Orthonormal':
        mode_check=True
    elif init_mode == 'LSUV':
        mode_check=True
    elif init_mode == 'OrthonormalLSUV':
        mode_check=True
    else:
        raise RuntimeError('Unknown mode. Try Orthonormal or LSUV or  OrthonormalLSUV')

    caffe.set_mode_cpu()
    for arg in sys.argv:
        if  arg == 'gpu':
            caffe.set_mode_gpu()

    solver = caffe.SGDSolver(solver_path)
    if os.path.isfile(init_path) and not noFetch:
        print("Loading")
        try:
            solver.net.copy_from(init_path)
        except:
            print('Failed to load weights from ', init_path) 

    for k,v in solver.net.params.iteritems():
        if ('BN' in k) or ('bn' in k):
            print('Skipping BatchNorm (*BN* name) layer')
            continue;
        if ('_act' in k) or ('_ACT' in k):
            print('Skipping activation (*_act* name) layer')
            continue;
        try:
            print(k, v[0].data.shape)
        except:
            print('Skipping layer ', k, ' as it has no parameters to initialize')
            continue
        if 'Orthonormal' in init_mode:
            weights=svd_orthonormal(v[0].data[:].shape)
            solver.net.params[k][0].data[:]=weights#* sqrt(2.0/(1.0+neg_slope*neg_slope));
        else:
            weights=solver.net.params[k][0].data[:]
            
        if 'LSUV' in init_mode:
            if var_before_relu_if_inplace:
                solver.net.forward(end=k)
            else:
                solver.net.forward()
            
            v = solver.net.blobs[k]
            var1  = np.var(v.data[:])
            mean1 = np.mean(v.data[:]);
            print(k,'var = ', var1,'mean = ', mean1)
            sys.stdout.flush()
            iter_num = 0;
            while (abs(needed_variance - var1) > margin):
                weights = solver.net.params[k][0].data[:]
                solver.net.params[k][0].data[:] = weights / sqrt(var1);
                if var_before_relu_if_inplace:
                    solver.net.forward(end=k)
                else:
                    solver.net.forward()
                v = solver.net.blobs[k];
                var1  = np.var(v.data[:]);
                mean1= np.mean(v.data[:]);
                print(k,'var = ', var1,'mean = ', mean1)
                sys.stdout.flush()
                iter_num+=1;
                if iter_num > max_iter:
                    print('Could not converge in ', iter_num, ' iterations, go to next layer')
                    break; 
    print("Initialization finished!")
    solver.net.forward()
    for k,v in solver.net.blobs.iteritems():
        try:
            print(k,v.data[:].shape, ' var = ', np.var(v.data[:]), ' mean = ', np.mean(v.data[:]))
        except:
            print('Skiping layer', k)
            
    print("Saving model...")
    solver.net.save(init_path)
    print("Finished. Model saved to", init_path)
