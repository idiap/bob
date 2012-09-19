/**
 * @file bob/machine/BICMachine.h
 * @date Tue Jun  5 16:54:27 CEST 2012
 * @author Manuel Guenther <Manuel.Guenther@idiap.ch>
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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


#ifndef BOB_MACHINE_BICMACHINE_H
#define BOB_MACHINE_BICMACHINE_H

#include <blitz/array.h>
#include "bob/io/HDF5File.h"

#include "bob/machine/Exception.h"


namespace bob { namespace machine {

  /**
   * Exceptions of this class are thrown when eigenvalues are too small.
   */
  class ZeroEigenvalueException : public Exception{
    public:
      ZeroEigenvalueException() throw(){}
      virtual ~ZeroEigenvalueException() throw(){}
      virtual const char* what() const throw(){return "One given/read/calculated eigenvalue is close to zero. Use a smaller number of kept eigenvalues!";}
  };

  /**
   * This class computes the Bayesian Intrapersonal/Extrapersonal Classifier (BIC),
   * (see "Beyond Eigenfaces: Probabilistic Matching for Face Recognition" from Moghaddam, Wahid and Pentland)
   * which estimates the posterior probability that the given <b>image difference vector</b>
   * is of the intrapersonal class, i.e., both images stem from the same person.
   *
   * There are two possible implementations of the BIC:
   * <ul>
   * <li> "The Bayesian Intrapersonal/Extrapersonal Classifier" from Teixeira.<br>
   *    A full projection of the data is performed. No prior for the classes has to be selected.</li>
   *
   * <li> "Face Detection and Recognition using Maximum Likelihood Classifiers on Gabor Graphs" from Günther and Würtz.<br>
   *    Only mean and variance of the difference vectors are calculated. There is no subspace truncation and no priors.</li>
   * </ul>
   * In any of the two implementations, the resulting score (using the forward() method) is a log-likelihood estimate,
   * using Mahalanobis(-like) distance measures.
   *
   */
  class BICMachine {
    public:
      //! generates an empty BIC machine
      BICMachine(bool use_DFFS = false);

      //! Copy constructor
      BICMachine(const BICMachine& other);

      //! Assignment Operator
      BICMachine& operator =(const BICMachine &other);

      //! Equality operator
      bool operator ==(const BICMachine& other) const;

      //! computes the BIC probability score for the given input difference vector
      void forward_(const blitz::Array<double,1>& input, blitz::Array<double,1>& output) const;

      //! performs some checks before calling the forward_ method
      void forward (const blitz::Array<double,1>& input, blitz::Array<double,1>& output) const;

      //! sets the IEC vectors of the given class
      void setIEC(bool clazz, const blitz::Array<double,1>& mean, const blitz::Array<double,1>& variances, bool copy_data = false);

      //! sets the BIC projection details of the given class
      void setBIC(bool clazz, const blitz::Array<double,1>& mean, const blitz::Array<double,1>& variances, const blitz::Array<double,2>& projection, const double rho, bool copy_data = false);

      //! loads this machine from the given hdf5 file.
      void load (bob::io::HDF5File& config);

      //! saves this machine to the given hdf5 file.
      void save (bob::io::HDF5File& config) const;

      //! Use the Distance From Feature Space
      void use_DFFS(bool use_DFFS = true);

      //! Use the Distance From Feature Space
      bool use_DFFS() const {return m_use_DFFS;}

    private:

      //! initializes internal data storages for the given class
      void initialize(bool clazz, int input_length, int projected_length);

      //! project data?
      bool m_project_data;

      //! mean vectors
      blitz::Array<double, 1> m_mu_I, m_mu_E;
      //! variances (eigenvalues)
      blitz::Array<double, 1> m_lambda_I, m_lambda_E;

      ///
      // only required when projection is enabled
      //! add the distance from feature space?
      bool m_use_DFFS;
      //! projection matrices (PCA)
      blitz::Array<double, 2> m_Phi_I, m_Phi_E;
      //! averaged eigenvalues to calculate DFFS
      double m_rho_I, m_rho_E;
      //! temporary storage
      mutable blitz::Array<double, 1> m_diff_I, m_diff_E;
      mutable blitz::Array<double, 1> m_proj_I, m_proj_E;

  };
}}

#endif // BOB_MACHINE_BICMACHINE_H
