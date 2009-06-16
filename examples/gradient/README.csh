#!/bin/csh -f

#
# test various Gradient Machines 
`uname -s`_`uname -m`/gradientmachines

#
# test the forward mecanism in the gradient machines composing a MLP
`uname -s`_`uname -m`/mlpforward1

#
# test the forward mecanism in a MLP
`uname -s`_`uname -m`/mlpforward2

#
# train the XOR problem with a MLP with GradientDescent
`uname -s`_`uname -m`/xor

#
# train and test a classification problem with a MLP with GradientDescent
`uname -s`_`uname -m`/mlpclassification train.list test.list -stochastic -nhu 10 -max_iter 50 -learning_rate 0.01


