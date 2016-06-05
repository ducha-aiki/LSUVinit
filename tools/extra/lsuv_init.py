#!/usr/bin/env python
from __future__ import print_function 
import os
import gc
import sys
import time

class bcolors:
    BLUE = '\033[94m'
    GREEN     = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


for arg in sys.argv:
    argl = arg.lower()
    if  argl == 'help' or argl == '-help' or argl == '--help' or argl == '-h' or argl == '--h' or argl == '/h' or argl == '/help':
        print (bcolors.RED + """___________________________________________________

""" + bcolors.END + """By $CAFFE_ROOT we mean Caffe's installation folder.
Place this script into """+ bcolors.GREEN     +"""$CAFFE_ROOT/tools/extra/"""+ bcolors.END +"""

Some common usage examples:
  """+ bcolors.GREEN     +"""python $CAFFE_ROOT/tools/extra/lsuv-init.py /path/to/solver.prototxt /path/to/initialised.caffemodel ONN mean=0.1 300
"""+ bcolors.END +""",
  """+ bcolors.GREEN     +"""python $CAFFE_ROOT/tools/extra/lsuv-init.py /path/to/solver.prototxt /path/to/initialised.caffemodel OrthonormalLSUV 100 skips=3 std=1.1
"""+ bcolors.END +""", or
  """+ bcolors.GREEN     +"""python $CAFFE_ROOT/tools/extra/lsuv-init.py /path/to/solver.prototxt /path/to/initialised.caffemodel LSUV noFetch cpu
"""+ bcolors.END +"""
Parameters:
"""+ bcolors.BLUE +"""initialised.caffemodel"""+ bcolors.END +""" here is where the initialised model will be saved to. If such file already exists, it will be loaded and the initialisation distortion will be applied to it instead.
"""+ bcolors.BLUE +"""noFetch"""+ bcolors.END +""" is an optional parameter for not loading existing "initialised.caffemodel" file and initialising via solver. It only matters for LSUV. Orthonormal/OrthonormalLSUV automatically uses noFetch, as Orthonormal inits values from scratch to it's own data, not just adjust existing values, like LSUV does.

"""+ bcolors.BLUE +"""LSUV"""+ bcolors.END +"""            tries to adjust variance to make it equal to 1 on your data.
"""+ bcolors.BLUE +"""Orthonormal"""+ bcolors.END +"""     paper: """+ bcolors.BLUE +"""http://arxiv.org/abs/1312.6120"""+ bcolors.END +"""
"""+ bcolors.BLUE +"""OrthonormalLSUV"""+ bcolors.END +""" paper: """+ bcolors.BLUE +"""http://arxiv.org/abs/1511.06422"""+ bcolors.END +""" is the """+ bcolors.GREEN     +"""classical LSUV from the paper"""+ bcolors.END +""". It first generates values from scratch with Orthonormal initialization and then adjusts them scaling each layer and adjusting its variance.
"""+ bcolors.BLUE +"""ONN"""+ bcolors.END +"""  """+ bcolors.YELLOW +"""experimental per-neuron"""+ bcolors.END +""" normalization. Uses Orthonormal init first. Then adjusts neural activations on tested set to have certain variance. If used with mean=[float] parameter, will also adjust all biases accordingly.

"""+ bcolors.BLUE +"""mean=0.2"""+ bcolors.END +"""  set """+ bcolors.YELLOW +"""mean"""+ bcolors.END +""" activation you want neurons to have. It will adjust their biases accordingly. Works for both per-neuron, or per-layer levels depending on which you choose - LSUV, or ONN.
"""+ bcolors.BLUE +"""std=1."""+ bcolors.END +"""  defaults to 1. Set required """+ bcolors.YELLOW +"""standard deviation"""+ bcolors.END +""" for neuron activations here. It will scale weights accordingly. Works for both per-neuron, or per-layer levels depending on which you choose - LSUV, or ONN.

"""+ bcolors.BLUE +"""cpu"""+ bcolors.END +"""  is an optional parameter for computing on CPU   instead of (default) GPU (the first one of them - "device #0" - if you have several). On GPU, LSUV init computation is likely to happen faster.
"""+ bcolors.BLUE +"""gpu1"""+ bcolors.END +""" is an optional parameter for computing on GPU#1 instead of (default) GPU#0
If you're using gpu, you may gain some speed by setting your NVidia GPU drivers into persistance mode
"""+ bcolors.GREEN     +"""sudo nvidia-smi -pm ENABLED -i 0"""+ bcolors.END     +"""
and then something like
"""+ bcolors.GREEN  +"""sudo nvidia-smi -ac """+ bcolors.YELLOW +"""3505,1392"""+ bcolors.GREEN     +""" -i 0"""+ bcolors.END     +"""
"""+ bcolors.YELLOW +"""3505,1392"""+ bcolors.END +""" - are values for my Titan X. Check out """+ bcolors.BLUE +"""https://devblogs.nvidia.com/parallelforall/increase-performance-gpu-boost-k80-autoboost/"""+ bcolors.END +""" to find out what values you need and other details.

"""+ bcolors.BLUE +"""number"""+ bcolors.END +"""  - defaults to 20, optional - """+ bcolors.YELLOW +"""this number"""+ bcolors.END +""" * """+ bcolors.YELLOW +"""batch size you've set up"""+ bcolors.END +""" influences how much data will be used for output data variance normalization calculation. Obviously, unless you're doing Orthonormal-only init, the more different your """+ bcolors.YELLOW +"""data"""+ bcolors.END +""" gets, the more you benefit from having this number """+ bcolors.GREEN +"""higher"""+ bcolors.END +""". Also, if your data isn't shuffled too well, you need this number to be higher, along with considering to use skips, so that you don't feed too few really different examples to the init. Having your data shuffled for maximum variance is highly recommended. The bigger it is, though, the more calculations will be performed which """+ bcolors.RED +"""takes longer"""+ bcolors.END +""".
"""+ bcolors.BLUE +"""skips=5"""+ bcolors.END +""" - defaults to 0,  optional - if your data is not finely shuffled, you can use this to skip """+ bcolors.YELLOW +"""skips"""+ bcolors.END +""" amount of batches in between taking every batch for evaluation. If data is not shuffled in your database for maximum variance per """+ bcolors.YELLOW +"""number"""+ bcolors.END +""" * """+ bcolors.YELLOW +"""batch size you've set up"""+ bcolors.END +""", before running this script you should probably set batch size to a very low value and skips to non-zero in order to gain more data variance.


"""+ bcolors.BOLD + bcolors.RED +"""IMPORTANT!"""+ bcolors.END +"""

First three parameters - solver, caffemodel and [OrthonormalLSUV|Orthonormal|LSUV|..] are a must and you should place them on these very places - in this arrangement.

* stands for anything
Name your """+ bcolors.YELLOW +"""activation layers"""+ bcolors.END +""" as """+ bcolors.BLUE +"""*_act*"""+ bcolors.END +""", or  """+ bcolors.BLUE +"""*_ACT*"""+ bcolors.END +"""
Name your """+ bcolors.YELLOW +"""batch normalization layers"""+ bcolors.END +""" as """+ bcolors.BLUE +"""*BN*"""+ bcolors.END +""", or """+ bcolors.BLUE +"""*bn*"""+ bcolors.END +"""
- so that the script wouldn't try to process stuff like """+ bcolors.YELLOW +"""PReLU activation layers"""+ bcolors.END +""" and get """+ bcolors.RED +"""stuck"""+ bcolors.END +""". This algorithm can only process fully-connected and convolutional layers. Not their activations. Working with PReLUs is not possible without naming them as *_act*

"""+ bcolors.RED +"""___________________________________________________
""" + bcolors.END)
        sys.exit()


print (bcolors.RED + """___________________________________________________
""" + bcolors.END)

        
from pylab import *
import random
import numpy as np
caffe_root_dir=os.path.dirname(os.path.realpath(__file__))
caffe_root_dir+='/../../python'
sys.path.insert(0, caffe_root_dir)
import caffe

caffe.set_mode_gpu()
for arg in sys.argv:
    if  arg == 'cpu':
        caffe.set_mode_cpu()

for arg in sys.argv:
    if  arg == 'gpu1':
        caffe.set_mode_gpu()
        caffe.set_device(1)



# Orthonorm init code was borrowed from Lasagne
# https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py


def svd_orthonormal(shape):
    if len(shape) < 2:
        raise RuntimeError("Only shapes of length 2 or more are supported.")
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = standard_normal(flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices = False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return q


noFetch = False
for arg in sys.argv:
    if  arg.lower() == 'nofetch':
        noFetch = True

nTimes  = 20
for arg in sys.argv:
    try:
       nTimes = int(arg)
    except ValueError:
       pass

skips  = 0
for arg in sys.argv:
    try:
       skips = int(arg)
    except ValueError:
       pass
       
       
skips   = False
for arg in sys.argv:
    if	arg.find('skips=')>-1:
        skips = int(arg[6:])
        print('Using skips of', skips, 'batches')
       
       
set_mean  = -999
for arg in sys.argv:
    if	arg.find('mean=')>-1:
        set_mean = float32(arg[5:])
        print('Setting mean to', set_mean)
       
       
required_stdeviation = 1.
for arg in sys.argv:
    if	arg.find('std=')>-1:
        required_stdeviation = float32(arg[4:])
        print('Setting std to', required_stdeviation)


       
if __name__ == '__main__':
    if len (sys.argv) < 4:
        raise RuntimeError('Usage: python ' + sys.argv[0] + ' path_to_solver path_to_save_model mode')

    solver_path = str(sys.argv[1])
    init_path   = str(sys.argv[2])
    init_mode   = str(sys.argv[3]).lower()
    max_iter            = 15
    margin              = 0.1
    var_before_relu_if_inplace = True

    ortho      = False  # start by performing orthonormal init
    post_init  = ''


    if   init_mode  == 'orthonormal':
        ortho       = True

    elif init_mode  == 'lsuv':
        post_init    = 'lsuv'

    elif init_mode  == 'orthonormallsuv':
        ortho       = True
        post_init    = 'lsuv'

    elif init_mode  == 'orthonormalnorm':
        ortho       = True

    elif init_mode  == 'onn':
        ortho       = True
        post_init    = 'neurons'

    else:
        raise RuntimeError('Unknown mode. Read -help. Try Orthonormal or LSUV or  OrthonormalLSUV, etc.')

    if ortho:
        noFetch     = True
    

    solver = caffe.SGDSolver(solver_path)
    if os.path.isfile(init_path) and (not noFetch):
        print("Loading")
        try:
            solver.net.copy_from(init_path)
        except:
            print('Failed to load weights from ', init_path) 
            
            
    first_layer = ''

    for k,v in solver.net.params.iteritems():
        if first_layer == '':
            first_layer = k
            
        if ('BN' in k) or ('bn' in k):
            print(bcolors.BLUE,'Skipping BatchNorm (*BN* name) layer', bcolors.END)
            continue
            
        if ('_act' in k) or ('_ACT' in k):
            print(bcolors.BLUE,'Skipping activation (*_act* name) layer', bcolors.END)
            continue
            
            
        if ortho:
            weights = svd_orthonormal(v[0].data[:].shape)
            solver.net.params[k][0].data[:] = weights   #* sqrt(2.0/(1.0+neg_slope*neg_slope))

        if post_init != '':
            solver.net.forward(end=k)
            print( solver.net.blobs[k].data[:].shape )
            activations = np.empty( shape = ((nTimes,) + solver.net.blobs[k].data[:].shape), dtype=float32 )
            gc.collect()

            for i in xrange(0, nTimes):
                if var_before_relu_if_inplace:
                    solver.net.forward(end=k)
                else:
                    solver.net.forward()
                    
                for skip in xrange(0, skips):
                    solver.net.forward(end=first_layer)
                
                activations[i] = solver.net.blobs[k].data[:].astype(float32)
                
                if i%25 == 0 and i!=0:
                    print(time.strftime("%d %H:%M:%S"), i, 'of', nTimes, 'computed')
                    gc.collect()
                    sys.stdout.flush()                    
            
            # Calculate the standart deviation of resulting data taking into account fact that the sample is not complete

            if post_init == 'neurons':
                #   per neuron analysis
                a = activations
                b = solver.net.params[k][1].data[:] # - bias
                w = solver.net.params[k][0].data[:] # - weights
                print(a.shape, b.shape, w.shape, 'a.shape, b.shape, w.shape')

                #   Ignoring type of neuronal layer we have - what its dimensions are - lets flatten it into 2D, or 3D
                while True:
                    ash = a.shape
                    try:
                        a = a.reshape( (ash[0]*ash[1], ash[2], ash[3]*ash[4]*ash[5]*ash[6]) )
                        break
                    except:
                        pass

                    try:
                        a = a.reshape( (ash[0]*ash[1], ash[2], ash[3]*ash[4]*ash[5]) )
                        break
                    except:
                        pass

                    try:
                        a = a.reshape( (ash[0]*ash[1], ash[2], ash[3]*ash[4]) )
                        break
                    except:
                        pass

                    try:
                        a = a.reshape( (ash[0]*ash[1], ash[2], ash[3]) )
                        break
                    except:
                        pass

                    try:
                        a = a.reshape( (ash[0]*ash[1], ash[2]) )
                        break
                    except:
                        pass

                    print(bcolors.RED ,"Oops! We can't process this layer, because it has some unconvetional dimentions. Check out python code to fix that.", bcolors.END)
                    break
                #   [ training case, neuron, spatial activations ]


                while True:
                    wsh = w.shape
                    try:
                        w = w.reshape( (wsh[0], wsh[1]*wsh[2]*wsh[3]*wsh[4]*wsh[5]*wsh[6]) )
                        break
                    except:
                        pass

                    try:
                        w = w.reshape( (wsh[0], wsh[1]*wsh[2]*wsh[3]*wsh[4]*wsh[5]) )
                        break
                    except:
                        pass

                    try:
                        w = w.reshape( (wsh[0], wsh[1]*wsh[2]*wsh[3]*wsh[4]) )
                        break
                    except:
                        pass

                    try:
                        w = w.reshape( (wsh[0], wsh[1]*wsh[2]*wsh[3]) )
                        break
                    except:
                        pass

                    try:
                        w = w.reshape( (wsh[0], wsh[1]*wsh[2]) )
                        break
                    except:
                        pass

                    try:
                        w = w.reshape( (wsh[0], wsh[1]) )
                        break
                    except:
                        pass

                    print(bcolors.RED ,"Oops! We can't process this layer, because it has some unconvetional dimentions. Check out python code to fix that.", bcolors.END)
                    break

                #   a[ training case, neuron, spatial activations ]  ->  [neuron, training case * spatial activations]
                #   b[neuron                ]
                #   w[neuron,        weights]
                # print(a.shape, 'a.shape')

                while True:
                    try:
                        a = np.transpose( a, axes=(1, 0, 2) )
                        # print(a.shape, 'a.shape transposed')
                        a = a.reshape( (a.shape[0], a.shape[1]*a.shape[2]) )
                        # print(a.shape, 'a.shape reshaped')
                        break
                    except:
                        # e = sys.exc_info()[0]
                        # print('Exception was: ', e)
                        pass

                    try:
                        a = a.transpose()
                        break
                    except:
                        pass

                    print(bcolors.RED ,"Oops! We can't process this layer, because it has some unconvetional dimentions. Check out python code to fix that.", bcolors.END)
                    break

                # print(a.shape, 'a.shape')
                mean = a.mean( 1, dtype=np.float64 )
                #   Centering neuronal activations using biases
                a1   = a.transpose()
                if set_mean != -999:
                    a1  -= mean
                    a1  += set_mean
                    b   -= mean
                    b   += set_mean
                    print('\n', time.strftime("%d %H:%M:%S"), bcolors.BLUE, k, bcolors.END,'Adjusted mean to become ', set_mean)
                stdD = a.std(  axis=1, ddof=1, dtype=np.float64  )
                w    = w.transpose()
                w   /= stdD
                if required_stdeviation != 1.:
                    w *= required_stdeviation

                print(bcolors.GREEN, "Neurons' weights rescaled to produce activations with", required_stdeviation,"standard deviation.", bcolors.END, 'Mean of standard deviation of neural activations (per neuron) was',    stdD.mean(dtype=np.float64), ', std of standard deviations themselves was',    stdD.std(ddof=1, dtype=np.float64), '. Layer-wise mean of neurons\' mean was', mean.mean(dtype=np.float64))

            else:
                #   per layer analysis
                mean  = activations.mean( dtype=np.float64 )
                if set_mean != -999:
                    #   Center neuronal activations at zero by manipulating bias
                    solver.net.params[k][1].data[:] -= mean
                    activations                     -= mean
                    solver.net.params[k][1].data[:] += set_mean
                    activations                     += set_mean
                    print('\n', time.strftime("%d %H:%M:%S"), bcolors.BLUE, k, bcolors.END,'Adjusted mean to become ', set_mean)

                stdD  = np.std(  activations, ddof=1, dtype=np.float64  )

                # print(activations.shape, 'activations.shape')
                print(time.strftime("%d %H:%M:%S"), bcolors.BLUE, k, bcolors.END,'Standard deviation was ~', stdD,', mean was ~', mean)
                sys.stdout.flush()
                solver.net.params[k][0].data[:] /= stdD
                if required_stdeviation != 1.:
                    solver.net.params[k][0].data[:] *= required_stdeviation


                print(bcolors.GREEN, "Layer's weights rescaled to produce activations with", required_stdeviation,"standard deviation.", bcolors.END)#, 'in',    iter_num, 'iterations')
                    
    print(bcolors.GREEN + """Initialization finished!""" + bcolors.YELLOW + """

Testing on a single batch. That's why numbers will not be as accurate. Also, don't be afraid of stats for data and labels - they don't matter.""" + bcolors.END)
    solver.net.forward()
    for k,l in solver.net.blobs.iteritems():
        try:
            v     = l.data[:]
            stdD  = np.std(  v, ddof=1, dtype=np.float64  )
            mean  = np.mean( v, dtype=np.float64 )
        
            print(bcolors.BLUE, k, bcolors.RED if abs(stdD - required_stdeviation)>margin*2 or (set_mean != -999 and abs(mean)>margin*2) else bcolors.END,'standard deviation ~', stdD,', mean ~', mean, bcolors.END)
        except:
            print('Cannot proceed layer',k,'skiping')
            print('Skiping layer', k)
            
    print("Saving model...")
    solver.net.save(init_path)
    print("Finished. Model saved to", init_path)
print (bcolors.RED + """___________________________________________________
""" + bcolors.END)
