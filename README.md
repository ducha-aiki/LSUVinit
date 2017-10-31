# Layer-sequential unit-variance (LSUV) initialization 

This is sample code for LSUV and Orthonormal initializations, implemented in python script within Caffe framework. 
The original Caffe README is reproduced below the line.
LSUV initialization is described in:

Mishkin, D. and Matas, J.,(2015). All you need is a good init. arXiv preprint [arXiv:1511.06422](http://arxiv.org/abs/1511.06422).

Orthonormal initialization is described in:

Andrew M. Saxe, James L. McClelland, Surya Ganguli (2013). Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. arXiv preprint [arXiv:1312.6120](http://arxiv.org/abs/1312.6120)

**upd.:** Torch re-implementation [https://github.com/yobibyte/torch-lsuv](https://github.com/yobibyte/torch-lsuv)

**upd.:** Keras re-implementation [https://github.com/ducha-aiki/LSUV-keras](https://github.com/ducha-aiki/LSUV-keras)

**New!** PyTorch re-implementation [https://github.com/ducha-aiki/LSUV-pytorch](https://github.com/ducha-aiki/LSUV-pytorch)

**New!** Thinc re-implementation [LSUV-thinc](https://github.com/explosion/thinc/blob/e653dd3dfe91f8572e2001c8943dbd9b9401768b/thinc/neural/_lsuv.py)

update: Why it is important to scale your input to var=1 before applying LSUV:

![scale-no-scale](https://github.com/ducha-aiki/caffenet-benchmark/blob/master/logs/contrib/img/0_dataset_init.png)


## Examples

See examples and sample log outputs in examples/lsuv. Run get_datasets.sh and training_*.sh for experiments reproduction. Note, than we haven`t freezed random seed and results may vary a little from run to run.
Initialization script itself is in tools/extra/lsuv_init.py

## Using with you current Caffe, or different framework installation

If you have Caffe installed already, you can simply copy script from   [tools/extra/lsuv_init.py](https://github.com/ducha-aiki/LSUVinit/blob/master/tools/extra/lsuv_init.py) and view its help (python lsuv_init.py -help) for further instructions. If you use a different framework, please, adapt that same script to your needs and notify us once you've got the code working, so we could let others know of existance of your solution if you're not against sharing your code. The script is self-explanatory, readable and adaptable.

## Citation

Please cite us if you use this code:

    @ARTICLE{LSUVInit2015,
    author = {{Mishkin}, D. and {Matas}, J.},
    title = "{All you need is a good init}",
    journal = {arXiv preprint arXiv:1511.06422},
    year = 2015,
    month = nov
    }

----


# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
