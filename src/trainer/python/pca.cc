/**
 * @file trainer/python/pca.cc
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

static const char SVD_CLASS_DOC[] = \
  "Sets a linear machine to perform the Principal Component Analysis (a.k.a. Karhunen-Loève Transform) on a given dataset using Singular Value Decomposition (SVD).\n" \
  "\n" \
  "SVD is factorization of a matrix X, with m rows and n columns, that\n" \
  "allows for the decomposition of a matrix :math:`X`, with size (m,n) into\n" \
  "3 other matrices in this way:\n" \
  "\n" \
  ".. math::\n" \
  "   \n" \
  "   X = U S V^*\n" \
  "\n" \
  "where:\n" \
  "\n" \
  ":math:`U`\n" \
  "  unitary matrix of size (m,m) - a.k.a., left singular vectors of X\n" \
  "\n" \
  ":math:`S`\n" \
  "  rectangular diagonal matrix with nonnegative real numbers, size (m,n)\n" \
  "\n" \
  ":math:`V^*`\n" \
  "  (the conjugate transpose of V) unitary matrix of size (n,n), a.k.a. right singular vectors of X\n" \
  "\n" \
  "We can use this property to avoid the computation of the covariance matrix of the data matrix :math:`X`, if we note the following:\n" \
  "\n" \
  ".. math::\n" \
  "   \n" \
  "   X &= U S V^* \\text{ , so} \\\\\n" \
  "   XX^T &= U S V^* V S U^*\\\\\n" \
  "   XX^T &= U S^2 U^*\n" \
  "\n" \
  "If X has zero mean, we can conclude by inspection that the U matrix obtained by SVD contains the eigen vectors of the covariance matrix of X (:math:`XX^T`) and :math:`S^2/(m-1)` corresponds to its eigen values.\n" \
  "\n" \
  "This trainer will apply these correspondances for you automatically, so you get a :py:class:`bob.machine.LinearMachine` which is correctly set to perform PCA and the eigen-values of :math:`XX^T`, all sorted in descending order (the first eigen-vector - or column - of the weight matrix in the :py:class:`~bob.machine.LinearMachine` corresponds to the highest eigen value obtained.\n" \
  "\n" \
  ".. note::\n" \
  "   \n" \
  "   Our implementation uses LAPACK's ``dgesdd`` to compute the solution to this linear equation.\n" \
  "\n" \
  ".. note::\n" \
  "   \n" \
  "   You should prefer this implementation over :py:class:`bob.trainer.CovMatrixPCATrainer` when the number of samples (rows of :math:`X`) is smaller than the number of features (columns of :math:`X`). It provides a faster execution path in that case.\n"
  "\n" \
  "References:\n" \
  "\n" \
  "1. Eigenfaces for Recognition, Turk & Pentland, Journal of Cognitive Neuroscience (1991) Volume: 3, Issue: 1, Publisher: MIT Press, Pages: 71-86\n" \
  "2. http://en.wikipedia.org/wiki/Singular_value_decomposition\n" \
  "3. http://en.wikipedia.org/wiki/Principal_component_analysis\n" \
  "4. http://en.wikipedia.org/wiki/Singular_value_decomposition\n" \
  "5. http://www.netlib.org/lapack/double/dgesdd.f\n" \
  ;

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

  class_<bob::trainer::SVDPCATrainer, boost::shared_ptr<bob::trainer::SVDPCATrainer> >("SVDPCATrainer", SVD_CLASS_DOC, no_init)
    
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
