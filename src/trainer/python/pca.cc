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
#include <bob/trainer/CovMatrixPCATrainer.h>
#include <bob/trainer/SVDPCATrainer.h>

using namespace boost::python;

tuple covmat_pca_train1(bob::trainer::CovMatrixPCATrainer& t, bob::python::const_ndarray data)
{
  const blitz::Array<double,2> data_ = data.bz<double,2>();
  const int n_eigs = std::min(data_.extent(0), data_.extent(1)) - 1;
  bob::machine::LinearMachine m(data_.extent(1), n_eigs);
  blitz::Array<double,1> eig_val(n_eigs);
  t.train(m, eig_val, data_);
  return make_tuple(m, object(eig_val));
}

object covmat_pca_train2(bob::trainer::CovMatrixPCATrainer& t, bob::machine::LinearMachine& m, bob::python::const_ndarray data)
{
  const blitz::Array<double,2> data_ = data.bz<double,2>();
  const int n_eigs = std::min(data_.extent(0), data_.extent(1)) - 1;
  blitz::Array<double,1> eig_val(n_eigs);
  t.train(m, eig_val, data_);
  return object(eig_val);
}

tuple svd_pca_train1(bob::trainer::SVDPCATrainer& t, bob::python::const_ndarray data)
{
  const blitz::Array<double,2> data_ = data.bz<double,2>();
  const int n_eigs = std::min(data_.extent(0), data_.extent(1)) - 1;
  bob::machine::LinearMachine m(data_.extent(1), n_eigs);
  blitz::Array<double,1> eig_val(n_eigs);
  t.train(m, eig_val, data_);
  return make_tuple(m, object(eig_val));
}

object svd_pca_train2(bob::trainer::SVDPCATrainer& t, bob::machine::LinearMachine& m, bob::python::const_ndarray data)
{
  const blitz::Array<double,2> data_ = data.bz<double,2>();
  const int n_eigs = std::min(data_.extent(0), data_.extent(1)) - 1;
  blitz::Array<double,1> eig_val(n_eigs);
  t.train(m, eig_val, data_);
  return object(eig_val);
}

void bind_trainer_pca() {
  class_<bob::trainer::CovMatrixPCATrainer, boost::shared_ptr<bob::trainer::CovMatrixPCATrainer> >("CovMatrixPCATrainer", "Sets a linear machine to perform the Karhunen-Loeve Transform (KLT) on a given dataset using the Covariance Matrix method. References:\n\n 1. Eigenfaces for Recognition, Turk & Pentland, Journal of Cognitive Neuroscience (1991) Volume: 3, Issue: 1, Publisher: MIT Press, Pages: 71-86\n 2. http://en.wikipedia.org/wiki/Principal_component_analysis", no_init)
    
    .def(init<>(
          "Initializes a new Covariance-Method-based PCA trainer.\n" \
          "\n" \
          "The training stage will place the resulting principal components in the linear machine and set it up to extract the variable means automatically. As an option, you may preset the trainer so that the normalization performed by the resulting linear machine also divides the variables by the standard deviation of each variable ensemble."))
    .def(init<const bob::trainer::CovMatrixPCATrainer&>(args("other")))

    .def(self == self)
    .def(self != self)
    
    .def("is_similar_to", &bob::trainer::CovMatrixPCATrainer::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this CovMatrixPCATrainer with the 'other' one to be approximately the same.")

    .def("train", &covmat_pca_train1, 
        (arg("self"), arg("data")), 
        "Trains a LinearMachine to perform the KLT.\n" \
        "\n" \
        "The resulting machine will have the same number of inputs as columns in ``data`` and :math:`K-1` eigen-vectors, where :math:`K=\\min{(S,F)}`, with :math:`S` being the number of rows in ``data`` (samples) and :math:`F` the number of columns (or features). The vectors are arranged by decreasing eigen-value automatically. You don't need to sort the results.\n"
        "\n" \
        "This method returns a tuple containing the resulting linear machine and the eigen values in a 1D array."
        )
    
    .def("train", &covmat_pca_train2, 
        (arg("self"), arg("machine"), arg("data")),
        "Trains a LinearMachine to perform the KLT.\n" \
        "\n" \
        "The resulting machine will have the same number of inputs as columns in ``data`` and :math:`K-1` eigen-vectors, where :math:`K=\\min{(S,F)}`, with :math:`S` being the number of rows in ``data`` (samples) and :math:`F` the number of columns (or features). The vectors are arranged by decreasing eigen-value automatically. You don't need to sort the results.\n"
        "\n" \
        "This method returns a tuple containing the resulting linear machine and the eigen values in a 1D array."
        )
    ;

  class_<bob::trainer::SVDPCATrainer, boost::shared_ptr<bob::trainer::SVDPCATrainer> >("SVDPCATrainer", "Sets a linear machine to perform the Karhunen-Loeve Transform (KLT) on a given dataset using Singular Value Decomposition (SVD). References:\n\n 1. Eigenfaces for Recognition, Turk & Pentland, Journal of Cognitive Neuroscience (1991) Volume: 3, Issue: 1, Publisher: MIT Press, Pages: 71-86\n 2. http://en.wikipedia.org/wiki/Singular_value_decomposition\n 3. http://en.wikipedia.org/wiki/Principal_component_analysis", no_init)
    
    .def(init<>(
          "Initializes a new SVD/PCA trainer.\n" \
          "\n" \
          "The training stage will place the resulting principal components in the linear machine and set it up to extract the variable means automatically. As an option, you may preset the trainer so that the normalization performed by the resulting linear machine also divides the variables by the standard deviation of each variable ensemble."))

    .def(init<const bob::trainer::SVDPCATrainer&>(args("other")))

    .def(self == self)
    .def(self != self)
    
    .def("is_similar_to", &bob::trainer::SVDPCATrainer::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this SVDPCATrainer with the 'other' one to be approximately the same.")

    .def("train", &svd_pca_train1, (arg("self"), arg("data")), 
        "Trains a LinearMachine to perform the KLT.\n" \
        "\n" \
        "The resulting machine will have the same number of inputs as columns in ``data`` and :math:`K-1` eigen-vectors, where :math:`K=\\min{(S,F)}`, with :math:`S` being the number of rows in ``data`` (samples) and :math:`F` the number of columns (or features). The vectors are arranged by decreasing eigen-value automatically. You don't need to sort the results.\n"
        "\n" \
        "This method returns a tuple containing the resulting linear machine and the eigen values in a 1D array."
        )

    .def("train", &svd_pca_train2, (arg("self"), arg("machine"), arg("data")),
        "Trains a LinearMachine to perform the KLT.\n" \
        "\n" \
        "The resulting machine will have the same number of inputs as columns in ``data`` and :math:`K-1` eigen-vectors, where :math:`K=\\min{(S,F)}`, with :math:`S` being the number of rows in ``data`` (samples) and :math:`F` the number of columns (or features). The vectors are arranged by decreasing eigen-value automatically. You don't need to sort the results.\n"
        "\n" \
        "This method returns a tuple containing the resulting linear machine and the eigen values in a 1D array."
        )
    ;

}
