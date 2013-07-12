/**
 * @file trainer/python/whitening.cc
 * @date Tue Apr 2 21:20:00 2013 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <boost/python.hpp>
#include <bob/python/ndarray.h>
#include <bob/trainer/WhiteningTrainer.h>
#include <bob/machine/LinearMachine.h>
#include <boost/shared_ptr.hpp>

using namespace boost::python;

static char CLASS_DOC[] = \
  "Trains a :py:class:`bob.machine.LinearMachine` to perform Cholesky Whitening.\n" \
  "\n" \
  "The whitening transformation is a decorrelation method that converts the covariance matrix of a set of samples into the identity matrix :math:`I`. This effectively linearly transforms random variables such that the resulting variables are uncorrelated and have the same variances as the original random variables. this transformation is invertible. The method is called the whitening transform because it transforms the input matrix X closer towards white noise (let's call it :math:`\\tilde{X}`): \n"\
  "\n"\
  ".. math::\n" \
  "   \n"\
  "   Cov(\\tilde{X}) = I\n"\
  "\n"\
  "where:\n"\
  "\n"\
  ".. math::\n" \
  "   \n"\
  "   \\tilde{X} = X W\n"\
  "\n"\
  "W is the projection matrix that allows us to linearly project the data matrix X to another (sub) space such that:\n"\
  "\n"\
  ".. math::\n" \
  "   \n" \
  "   Cov(X) = W W^T\n" \
  "\n"\
  "W is computed using Cholesky Decomposition:\n"\
  "\n"\
  ".. math::\n" \
  "   \n" \
  "   W = cholesky([Cov(X)]^{-1})\n" \
  "\n" \
    "References:\n" \
  "\n" \
  "1. https://rtmath.net/help/html/e9c12dc0-e813-4ca9-aaa3-82340f1c5d24.htm\n"\
  "2. http://en.wikipedia.org/wiki/Cholesky_decomposition\n"\
  "\n"\
  "\n"\
;


void py_train1(bob::trainer::WhiteningTrainer& t, 
  bob::machine::LinearMachine& m, bob::python::const_ndarray data)
{
  const blitz::Array<double,2> data_ = data.bz<double,2>();
  t.train(m, data_);
}

object py_train2(bob::trainer::WhiteningTrainer& t, 
  bob::python::const_ndarray data)
{
  const blitz::Array<double,2> data_ = data.bz<double,2>();
  const int n_features = data_.extent(1);
  bob::machine::LinearMachine m(n_features,n_features);
  t.train(m, data_);
  return object(m);
}


void bind_trainer_whitening() 
{
  class_<bob::trainer::WhiteningTrainer, boost::shared_ptr<bob::trainer::WhiteningTrainer> >("WhiteningTrainer", CLASS_DOC, init<>((arg("self")), "Initializes a new Whitening trainer."))
    .def(init<const bob::trainer::WhiteningTrainer&>((arg("self"), arg("other")), "Copy constructs a WhiteningTrainer"))
    .def(self == self)
    .def(self != self)
    .def("is_similar_to", &bob::trainer::WhiteningTrainer::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this WhiteningTrainer with the 'other' one to be approximately the same.")
    .def("train", &py_train1, (arg("self"), arg("machine"), arg("data")), "Trains the LinearMachine to perform the Whitening, given a training set.")
    .def("train", &py_train2, (arg("self"), arg("data")), "Allocates, trains and returns a LinearMachine to perform the Whitening, given a training set.")
  ;
}
