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
#include <bob/core/python/ndarray.h>
#include <bob/trainer/WhiteningTrainer.h>
#include <bob/machine/LinearMachine.h>
#include <boost/shared_ptr.hpp>

using namespace boost::python;

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
  class_<bob::trainer::WhiteningTrainer, boost::shared_ptr<bob::trainer::WhiteningTrainer> >("WhiteningTrainer", "Trains a linear machine to perform whitening.\nReference:\n'Independent component analysis: algorithms and applications', Aapo Hyvärinen, Erkki Oja, Neural Networks, 2000, vol. 13, p. 411--430\nGiven a training set X, this will compute the W matrix such that:\nW = cholesky(inv(cov(X_{n},X_{n}^{T}))), where X_{n} corresponds to the center data.", init<>("Initializes a new Whitening trainer."))
    .def(init<const bob::trainer::WhiteningTrainer&>(args("other"), "Copy constructs a WhiteningTrainer"))
    .def(self == self)
    .def(self != self)
    .def("is_similar_to", &bob::trainer::WhiteningTrainer::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this WhiteningTrainer with the 'other' one to be approximately the same.")
    .def("train", &py_train1, (arg("self"), arg("machine"), arg("data")), "Trains the LinearMachine to perform the Whitening, given a training set.")
    .def("train", &py_train2, (arg("self"), arg("data")), "Allocates, trains and returns a LinearMachine to perform the Whitening, given a training set.")
  ;
}
