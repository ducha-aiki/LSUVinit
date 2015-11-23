#!/bin/bash

#CIFAR-10 from https://github.com/flukeskywalker/highway-networks 
wget -c https://www.dropbox.com/s/r9zuhhhii4uzi24/cifar10-gcn-leveldb-splits.tar.bz2?dl=0
mv cifar10-gcn-leveldb-splits.tar.bz2?dl=0 cifar10-gcn-leveldb-splits.tar.bz2
tar -xjf cifar10-gcn-leveldb-splits.tar.bz2

#CIFAR-100 from https://github.com/flukeskywalker/highway-networks 
wget -c https://www.dropbox.com/s/w2qywjihzr7avfa/cifar100-gcn-leveldb-splits.tar.bz2?dl=0
mv cifar100-gcn-leveldb-splits.tar.bz2?dl=0 cifar100-gcn-leveldb-splits.tar.bz2
tar -xjf cifar100-gcn-leveldb-splits.tar.bz2

#Caffe MNIST
../../data/mnist/get_mnist.sh
cd ../..
examples/mnist/create_mnist.sh

