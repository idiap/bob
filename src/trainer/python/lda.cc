/**
 * @file trainer/python/linear.cc
 * @date Fri Jun 10 16:43:41 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Python bindings to LinearMachine trainers
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

#include <boost/shared_ptr.hpp>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>

#include <bob/python/ndarray.h>
#include <bob/trainer/FisherLDATrainer.h>

using namespace boost::python;

tuple lda_train1(bob::trainer::FisherLDATrainer& t, object data)
{
  stl_input_iterator<blitz::Array<double,2> > dbegin(data), dend;
  std::vector<blitz::Array<double,2> > vdata(dbegin, dend);
  int output_dimension = t.lda_dimensions(vdata);
  blitz::Array<double,1> eig_val(output_dimension);
  bob::machine::LinearMachine m(vdata[0].extent(1), output_dimension);
  t.train(m, eig_val, vdata);
  return make_tuple(m, eig_val);
}

object lda_train2(bob::trainer::FisherLDATrainer& t,
  bob::machine::LinearMachine& m, object data)
{
  stl_input_iterator<blitz::Array<double,2> > dbegin(data), dend;
  std::vector<blitz::Array<double,2> > vdata(dbegin, dend);
  blitz::Array<double,1> eig_val(t.lda_dimensions(vdata));
  t.train(m, eig_val, vdata);
  return object(eig_val);
}

int lda_dimensions(bob::trainer::FisherLDATrainer& t, object data)
{
  stl_input_iterator<blitz::Array<double,2> > dbegin(data), dend;
  std::vector<blitz::Array<double,2> > vdata(dbegin, dend);
  return t.lda_dimensions(vdata);
}

void bind_trainer_lda()
{
  class_<bob::trainer::FisherLDATrainer, boost::shared_ptr<bob::trainer::FisherLDATrainer> >("FisherLDATrainer", "Implements a multi-class Fisher/LDA linear machine Training using Singular Value Decomposition (SVD). For more information on Linear Machines and associated methods, please consult Bishop, Machine Learning and Pattern Recognition chapter 4. The number of LDA dimensions can be: 0: The theoretical limit (#classes-1) is kept; -1: all dimensions are kept (also the ones with zero eigenvalue); >0: The given number of dimensions are kept (can be at most the input dimension)", init<int>(args("number_of_kept_lda_dimensions")=0))
    .def(init<const bob::trainer::FisherLDATrainer&>(args("other")))
    .def(self == self)
    .def(self != self)
    .def("is_similar_to", &bob::trainer::FisherLDATrainer::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this FisherLDATrainer with the 'other' one to be approximately the same.")
    .def("train", &lda_train1, (arg("self"), arg("data")), "Creates a LinearMachine that performs Fisher/LDA discrimination. The resulting machine will have the eigen-vectors of the Sigma-1 * Sigma_b product, arranged by decreasing 'energy'. Each input arrayset represents data from a given input class. This method returns a tuple containing the resulting linear machine and the eigen values in a 1D array. This way you can reset the machine as you see fit.\n\nNote we set only the N-1 eigen vectors in the linear machine since the last eigen value should be zero anyway. You can compress the machine output further using resize() if necessary.")
    .def("train", &lda_train2, (arg("self"), arg("machine"), arg("data")), "Trains a given LinearMachine to perform Fisher/LDA discrimination. After this method has been called, the input machine will have the eigen-vectors of the Sigma-1 * Sigma_b product, arranged by decreasing 'energy'. Each input arrayset represents data from a given input class. This method also returns the eigen values allowing you to implement your own compression scheme.\n\nNote we set only the N-1 eigen vectors in the linear machine since the last eigen value should be zero anyway. You can compress the machine output further using resize() if necessary.")
    .def("lda_dimensions", &lda_dimensions, (arg("self"), arg("data")), "Returns the output dimensionality of the LinearMachine for the given training data.")
  ;

}
