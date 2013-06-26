/**
 * @file trainer/cxx/BICTrainer.cc
 * @date Wed Jun  6 10:29:09 CEST 2012
 * @author Manuel Guenther <Manuel.Guenther@idiap.ch>
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

#include <bob/core/array_exception.h>
#include <bob/trainer/BICTrainer.h>
#include <bob/trainer/PCATrainer.h>

static double sqr(const double& x){
  return x*x;
}

/**
 * This function trains one of the classes of the given machine with the given data.
 * It computes either BIC projection matrices, or IEC mean and variance.
 *
 * @param  clazz    false for the intrapersonal class, true for the extrapersonal one.
 * @param  machine  The machine to be trained.
 * @param  differences  A set of (intra/extra)-personal difference vectors that should be trained.
 */
void bob::trainer::BICTrainer::train_single(bool clazz, bob::machine::BICMachine& machine, const blitz::Array<double,2>& differences) const {
  int subspace_dim = clazz ? m_M_E : m_M_I;
  int input_dim = differences.extent(1);
  int data_count = differences.extent(0);
  blitz::Range a = blitz::Range::all();

  if (subspace_dim){
    // train the class using BIC

    // Compute PCA on the given dataset
    bob::trainer::PCATrainer trainer;
    const int n_eigs = trainer.output_size(differences);
    bob::machine::LinearMachine pca(input_dim, n_eigs);
    blitz::Array<double,1> variances(n_eigs);
    trainer.train(pca, variances, differences);

    // compute rho
    double rho = 0.;
    int non_null_eigenvalues = std::min(input_dim, data_count) - 1;
    // assert that the number of kept eigenvalues is not chosen to big
    if (subspace_dim >= non_null_eigenvalues) throw bob::machine::ZeroEigenvalueException();
    // compute the average of the reminding eigenvalues
    for (int i = subspace_dim; i < non_null_eigenvalues; ++i){
      rho += variances(i);
    }
    rho /= non_null_eigenvalues - subspace_dim;

    // limit dimensionalities
    pca.resize(input_dim, subspace_dim);
    variances.resizeAndPreserve(subspace_dim);

    // check that all variances are meaningful
    for (int i = subspace_dim; i--;){
      if (variances(i) < 1e-12) throw bob::machine::ZeroEigenvalueException();
    }

    // initialize the machine
    blitz::Array<double, 2> projection = pca.getWeights();
    blitz::Array<double, 1> mean = pca.getInputSubtraction();
    machine.setBIC(clazz, mean, variances, projection, rho);
  } else {
    // train the class using IEC
    // => compute mean and variance only
    blitz::Array<double,1> mean(input_dim), variance(input_dim);

    // compute mean and variance
    mean = 0.;
    variance = 0.;
    for (int n = data_count; n--;){
      const blitz::Array<double,1>& diff = differences(n,a);
      assert(diff.shape()[0] == input_dim);
      for (int i = input_dim; i--;){
        mean(i) += diff(i);
        variance(i) += sqr(diff(i));
      }
    }
    // normalize mean and variances
    for (int i = input_dim; i--;){
      // intrapersonal
      variance(i) = (variance(i) - sqr(mean(i)) / data_count) / (data_count - 1.);
      mean(i) /= data_count;
      if (variance(i) < 1e-12) throw bob::machine::ZeroEigenvalueException();
    }

    // set the results to the machine
    machine.setIEC(clazz, mean, variance);
  }
}
