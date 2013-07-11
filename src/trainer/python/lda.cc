/**
 * @file trainer/python/lda.cc
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

#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/shared_ptr.hpp>

#include <bob/python/ndarray.h>
#include <bob/trainer/FisherLDATrainer.h>

using namespace boost::python;

static tuple lda_train1(bob::trainer::FisherLDATrainer& t, object data)
{
  stl_input_iterator<bob::python::const_ndarray> dbegin(data), dend;
  std::vector<bob::python::const_ndarray> vdata_ref(dbegin, dend);
  std::vector<blitz::Array<double,2> > vdata;
  for(std::vector<bob::python::const_ndarray>::iterator it=vdata_ref.begin(); 
      it!=vdata_ref.end(); ++it)
    vdata.push_back(it->bz<double,2>());
  int osize = t.output_size(vdata);
  blitz::Array<double,1> eig_val(osize);
  bob::machine::LinearMachine m(vdata[0].extent(1), osize);
  t.train(m, eig_val, vdata);
  return make_tuple(m, eig_val);
}

static object lda_train2(bob::trainer::FisherLDATrainer& t,
  bob::machine::LinearMachine& m, object data)
{
  stl_input_iterator<bob::python::const_ndarray> dbegin(data), dend;
  std::vector<bob::python::const_ndarray> vdata_ref(dbegin, dend);
  std::vector<blitz::Array<double,2> > vdata;
  for(std::vector<bob::python::const_ndarray>::iterator it=vdata_ref.begin(); 
      it!=vdata_ref.end(); ++it)
    vdata.push_back(it->bz<double,2>());
  blitz::Array<double,1> eig_val(t.output_size(vdata));
  t.train(m, eig_val, vdata);
  return object(eig_val);
}

static size_t output_size(bob::trainer::FisherLDATrainer& t, object data) {
  stl_input_iterator<bob::python::const_ndarray> dbegin(data), dend;
  std::vector<bob::python::const_ndarray> vdata_ref(dbegin, dend);
  std::vector<blitz::Array<double,2> > vdata;
  for(std::vector<bob::python::const_ndarray>::iterator it=vdata_ref.begin(); 
      it!=vdata_ref.end(); ++it)
    vdata.push_back(it->bz<double,2>());
  return t.output_size(vdata);
}

static char CLASS_DOC[] = \
  "Trains a :py:class:`bob.machine.LinearMachine` to perform Fisher's Linear Discriminant Analysis (LDA).\n" \
  "\n" \
  "LDA finds the projection matrix W that allows us to linearly project the data matrix X to another (sub) space in which the between-class and within-class variances are jointly optimized: the between-class variance is maximized while the with-class is minimized. The (inverse) cost function for this criteria can be posed as the following:\n" \
  "\n" \
  ".. math::\n" \
  "   \n" \
  "   J(W) = \\frac{W^T S_b W}{W^T S_w W}\n" \
  "\n" \
  "where:\n" \
  "\n" \
  ":math:`W`\n" \
  "   \n" \
  "   the transformation matrix that converts X into the LD space\n" \
  "\n" \
  ":math:`S_b`\n" \
  "   \n" \
  "   the between-class scatter; it has dimensions (X.shape[0], X.shape[0]) and is defined as :math:`S_b = \\sum_{k=1}^K N_k (m_k-m)(m_k-m)^T`, with K equal to the number of classes.\n" \
  "\n" \
  ":math:`S_w`\n" \
  "  \n" \
  "   the within-class scatter; it also has dimensions (X.shape[0], X.shape[0]) and is defined as :math:`S_w = \\sum_{k=1}^K \\sum_{n \\in C_k} (x_n-m_k)(x_n-m_k)^T`, with K equal to the number of classes and :math:`C_k` a set representing all samples for class k.\n" \
  "\n" \
  ":math:`m_k`\n" \
  "  \n" \
  "   the class *k* empirical mean, defined as :math:`m_k = \\frac{1}{N_k}\\sum_{n \\in C_k} x_n`\n" \
  "\n" \
  ":math:`m`\n" \
  "  \n" \
  "   the overall set empirical mean, defined as :math:`m = \\frac{1}{N}\\sum_{n=1}^N x_n = \\frac{1}{N}\\sum_{k=1}^K N_k m_k`\n" \
  "\n" \
  ".. note::\n" \
  "   \n" \
  "   A scatter matrix equals the covariance matrix if we remove the division factor.\n" \
  "\n" \
  "Because this cost function is convex, you can just find its maximum by solving :math:`dJ/dW = 0`. This problem can be re-formulated as finding the eigen values (:math:`\\lambda_i`) that solve the following condition:\n" \
  "\n" \
  ".. math::\n" \
  "  \n" \
  "  S_b &= \\lambda_i Sw \\text{ or} \\\\\n" \
  "  (Sb - \\lambda_i Sw) &= 0\n" \
  "\n" \
  "The respective eigen vectors that correspond to the eigen values :math:`\\lambda_i` form W.\n" \
  "\n" \
  ;

void bind_trainer_lda()
{
  class_<bob::trainer::FisherLDATrainer, boost::shared_ptr<bob::trainer::FisherLDATrainer> >("FisherLDATrainer", CLASS_DOC, no_init)
    
    .def(init<optional<bool,bool> >((arg("self"), arg("use_pinv")=false, arg("strip_to_rank")=true),
          "Creates a new trainer to perform LDA\n" \
          "\n" \
          "Keyword parameters:\n" \
          "\n" \
          "use_pinv (bool) - defaults to ``False``\n" \
          "   \n" \
          "   If set to ``True``, use the pseudo-inverse to calculate :math:`S_w^{-1} S_b` and then perform eigen value decomposition (using LAPACK's ``dgeev``) instead of using (the more numerically stable) LAPACK's ``dsyvgd`` to solve the generalized symmetric-definite eigenproblem of the form :math:`S_b v=(\\lambda) S_w v`\n" \
          "   \n" \
          "   .. note::\n" \
          "     \n" \
          "     Using the pseudo-inverse for LDA is only recommended if you cannot make it work using the default method (via ``dsyvg``). It is slower and requires more machine memory to store partial values of the pseudo-inverse and the dot product :math:`S_w^{-1} S_b`.\n" \
          "\n" \
          "strip_to_rank (bool) - defaults to ``True``\n" \
          "   \n" \
          "   Specifies how to calculate the final size of the to-be-trained :py:class:`bob.machine.LinearMachine`. The default setting (``True``), makes the trainer return only the K-1 eigen-values/vectors limiting the output to the rank of :math:`S_w^{-1} S_b`. If you set this value to ``False``, the it returns all eigen-values/vectors of :math:`S_w^{-1} Sb`, including the ones that are supposed to be zero.\n" \
          "\n" \
          ))
    
    .def(init<const bob::trainer::FisherLDATrainer&>((arg("self"), arg("other")), "Copy constructor - use this to deepcopy another trainer"))

    .def(self == self)
    .def(self != self)

    .def("is_similar_to", &bob::trainer::FisherLDATrainer::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this FisherLDATrainer with the 'other' one to be approximately the same.")

    .def("train", &lda_train1, (arg("self"), arg("X")), 
        "Creates a LinearMachine that performs Fisher/LDA discrimination.\n" \
        "\n" \
        "The resulting machine will contain the eigen-vectors of the :math:`S_w^{-1} S_b` product, arranged by decreasing energy. Each input arrayset represents data from a given input class. This method returns a tuple containing the resulting linear machine and the eigen values in a 1D array. This way, you can reset the machine as you see fit.\n" \
        "\n" \
        ".. note::\n" \
        "   \n" \
        "   We set at most :py:meth:`bob.trainer.FisherLDATrainer.output_size` eigen-values and vectors on the passed machine. You can compress the machine output further using :py:meth:`bob.machine.LinearMachine.resize` if necessary.\n" \
        )

    .def("train", &lda_train2, (arg("self"), arg("machine"), arg("X")),
        "Trains a given LinearMachine to perform Fisher/LDA discrimination.\n" \
        "\n" \
        "After this method has been called, the input machine will have the eigen-vectors of the :math:`S_w^{-1} S_b` product, arranged by decreasing energy. Each input data set represents data from a given input class. This method also returns the eigen values allowing you to implement your own compression scheme.\n" \
        "\n" \
        ".. note::\n" \
        "   \n" \
        "   We set at most :py:meth:`bob.trainer.FisherLDATrainer.output_size` eigen-values and vectors on the passed machine. You can compress the machine output further using :py:meth:`bob.machine.LinearMachine.resize` if necessary.\n" \
        )

    .def("output_size", &output_size, (arg("self"), arg("X")),
       "Returns the expected size of the output (or the number of eigen-values returned) given the data.\n" \
       "\n" \
       "This number could be either K-1 (where K is number of classes) or the number of columns (features) in X, depending on the setting of ``strip_to_rank``.\n" \
       )

    .add_property("use_pinv", &bob::trainer::FisherLDATrainer::getUsePseudoInverse, &bob::trainer::FisherLDATrainer::setUsePseudoInverse,
        "If ``True``, use the pseudo-inverse to calculate :math:`S_w^{-1} S_b` and then perform the eigen value decomposition (using LAPACK's ``dgeev``) instead of using (the more numerically stable) LAPACK's ``dsyvgd`` to solve the generalized symmetric-definite eigenproblem of the form :math:`S_b v=(\\lambda) S_w v`")

    .add_property("strip_to_rank", &bob::trainer::FisherLDATrainer::getStripToRank, &bob::trainer::FisherLDATrainer::setStripToRank,
        "Specifies how to calculate the final size of the to-be-trained :py:class:`bob.machine.LinearMachine`. The default setting (``True``), makes the trainer return only the K-1 eigen-values/vectors limiting the output to the rank of :math:`S_w^{-1} S_b`. If you set this value to ``False``, the it returns all eigen-values/vectors of :math:`S_w^{-1} Sb`, including the ones that are supposed to be zero.")

  ;

}
