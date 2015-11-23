#!/bin/bash
caffe_tool=../../build/tools/caffe
DATETIME=`date +%Y-%m-%d-%R_%s%N`

##CIFAR10
protos=`find cifar10/ -maxdepth 1 -name '*lsuv*.prototxt' -type f -exec echo '{}' \;`
if  [[ ${protos} ]]; then
  for pr in ${protos[@]}; do
    echo ${pr}
    b_pr=$(basename "$pr" )
    python ../../tools/extra/lsuv_init.py ${pr} ${b_pr}.caffemodel OrthonormalLSUV
    ${caffe_tool} train --solver=${pr} --weights=${b_pr}.caffemodel 2>&1 | tee logs/cifar10_${b_pr}_${DATETIME}.log
  done
fi

protos=`find cifar10/ -maxdepth 1 -name '*ortho*.prototxt' -type f -exec echo '{}' \;`
if  [[ ${protos} ]]; then
  for pr in ${protos[@]}; do
    echo ${pr}
    b_pr=$(basename "$pr" )
    python ../../tools/extra/lsuv_init.py ${pr} ${b_pr}.caffemodel Orthonormal
    ${caffe_tool} train --solver=${pr} --weights=${b_pr}.caffemodel 2>&1 | tee logs/cifar10_${b_pr}_${DATETIME}.log
  done
fi

protos=`find cifar10/ -maxdepth 1 -name '*xavier*.prototxt' -type f -exec echo '{}' \;`
f  [[ ${protos} ]]; then
  for pr in ${protos[@]}; do
    echo ${pr}
    b_pr=$(basename "$pr" )
    ${caffe_tool} train --solver=${pr} 2>&1 | tee logs/cifar10_${b_pr}_${DATETIME}.log
  done
fi


protos=`find cifar10/ -maxdepth 1 -name '*msra*.prototxt' -type f -exec echo '{}' \;`
if  [[ ${protos} ]]; then
  for pr in ${protos[@]}; do
    echo ${pr}
    b_pr=$(basename "$pr" )
    ${caffe_tool} train --solver=${pr} 2>&1 | tee logs/cifar10_${b_pr}_${DATETIME}.log
  done
fi
