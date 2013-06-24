/**
 * @file trainer/python/wccn.cc
 * @date Tue Apr 10 15:00:00 2013 +0200
 * @author Elie Khoury <Elie.Khoury@idiap.ch>
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
#include <boost/shared_ptr.hpp>
#include <boost/python/stl_iterator.hpp>
#include <bob/trainer/WCCNTrainer.h>
#include <bob/machine/LinearMachine.h>

using namespace boost::python;

void py_train1(bob::trainer::WCCNTrainer& t, bob::machine::LinearMachine& m, object data)
{
  stl_input_iterator<blitz::Array<double,2> > dbegin(data), dend;
  std::vector<blitz::Array<double,2> > vdata(dbegin, dend);
  blitz::Array<double,1> eig_val(vdata[0].extent(1)-1);
  t.train(m, vdata);
}

object py_train2(bob::trainer::WCCNTrainer& t, object data)
{
  stl_input_iterator<blitz::Array<double,2> > dbegin(data), dend;
  std::vector<blitz::Array<double,2> > vdata(dbegin, dend);
  bob::machine::LinearMachine m(vdata[0].extent(1),vdata[0].extent(1));
  t.train(m, vdata);
  return object(m);
}

void bind_trainer_wccn()
{
  class_<bob::trainer::WCCNTrainer, boost::shared_ptr<bob::trainer::WCCNTrainer> >("WCCNTrainer", "Trains a linear machine to perform WCCN.\nReference:\n'Independent component analysis: algorithms and applications', Aapo Hyvarinen, Erkki Oja, Neural Networks, 2000, vol. 13, p. 411--430\nGiven a training set X, this will compute the W matrix such that:\nW = cholesky(inv(cov(X_{n},X_{n}^{T}))), where X_{n} corresponds to the center data.", init<>("Initializes a new WCCN trainer."))
    .def(init<const bob::trainer::WCCNTrainer&>(args("other"), "Copy constructs a WCCNTrainer"))
    .def(self == self)
    .def(self != self)
    .def("is_similar_to", &bob::trainer::WCCNTrainer::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this WCCNTrainer with the 'other' one to be approximately the same.")
    .def("train", &py_train1, (arg("self"), arg("machine"), arg("data")), "Trains the LinearMachine to perform the WCCN, given a training set.")
    .def("train", &py_train2, (arg("self"), arg("data")), "Allocates, trains and returns a LinearMachine to perform the WCCN, given a training set.")
  ;
}
