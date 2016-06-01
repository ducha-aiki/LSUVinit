#!/usr/bin/env python
from __future__ import print_function 
import os
import sys

for arg in sys.argv:
    if  arg == 'help' or arg == 'HELP' or arg == '-help' or arg == '--help' or arg == '-h' or arg == '--h' or arg == '/h' or arg == '/help' or arg == '/H' or arg == '/HELP':
        print ("""____________________________________________
		
By $CAFFE_ROOT we mean the installation folder for Caffe. Place this script into $CAFFE_ROOT/tools/extra/

Use it as:
    python $CAFFE_ROOT/tools/extra/lsuv.py /path/to/solver.prototxt /path/to/initialised.caffemodel LSUV
or
    python $CAFFE_ROOT/tools/extra/lsuv.py /path/to/solver.prototxt /path/to/initialised.caffemodel Orthonormal noFetch
	
initialised.caffemodel is where the initialised model be saved to. If such file already exists, it will be loaded and the initialisation distortion will be applied to it instead.
noFetch is an optional parameter for not loading existing "initialised.caffemodel" file.

NOTE!
Name your activation layers as *_act*, or  *_ACT*
Name your batch normalization layers as *BN*, or *bn*
(* - stands for anything),
so that the script wouldn't try to process the activation layers like PReLU and get stuck. This algorithm can only process linear and convolutional layers.

____________________________________________
""")
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

    caffe.set_mode_gpu()
    solver = caffe.SGDSolver(solver_path)
    if os.path.isfile(init_path) and not noFetch:
        print("Loading")
        try:
            solver.net.copy_from(init_path)
        except:
            print('Failed to load weights from ', init_path) 

    for k,v in solver.net.params.iteritems():
        try:
            print(k, v[0].data.shape)
        except:
            print('Error with layer',k, 'skipping it')
            continue
            
        if ('BN' in k) or ('bn' in k):
            print('Skipping BatchNorm (*BN* name) layer')
            continue;
            
        if ('_act' in k) or ('_ACT' in k):
            print('Skipping activation (*_act* name) layer')
            continue;
            
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
            print('Cannot proceed layer',k,'skiping')
            
    print("Saving model...")
    solver.net.save(init_path)
    print("Finished. Model saved to", init_path)
