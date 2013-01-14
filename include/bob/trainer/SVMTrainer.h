/**
 * @file bob/trainer/SVMTrainer.h
 * @date Sat Dec 17 14:41:56 2011 +0100
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief C++ bindings to libsvm (training bits)
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

#ifndef BOB_TRAINER_SVMTRAINER_H 
#define BOB_TRAINER_SVMTRAINER_H

#include <vector>
#include "bob/machine/SVM.h"

namespace bob { namespace trainer {

  /**
   * This class emulates the behavior of the command line utility called
   * svm-train, from libsvm. These bindings do not support:
   *
   * * Precomputed Kernels
   * * Regression Problems
   * * Different weights for every label (-wi option in svm-train)
   *
   * Fell free to implement those and remove these remarks.
   */
  class SVMTrainer {

    public: //api

      /**
       * Builds a new trainer setting the default parameters as defined in the
       * command line application svm-train.
       */
      SVMTrainer(
          bob::machine::SupportVector::svm_t svm_type=bob::machine::SupportVector::C_SVC,
          bob::machine::SupportVector::kernel_t kernel_type=bob::machine::SupportVector::RBF,
          int degree=3, //for poly
          double gamma=0., //for poly/rbf/sigmoid
          double coef0=0., //for poly/sigmoid
          double cache_size=100, //in MB
          double eps=1.e-3, //stopping criteria epsilon
          double C=1., //for C_SVC, EPSILON_SVR and NU_SVR
          double nu=0.5, //for NU_SVC, ONE_CLASS and NU_SVR
          double p=0.1, //for EPSILON_SVR, this is the "epsilon" value there
          bool shrinking=true, //use the shrinking heuristics
          bool probability=false //do probability estimates
          );
      /** TODO: Support for weight cost in multi-class classification? **/

      /**
       * Destructor virtualisation
       */
      virtual ~SVMTrainer();

      /**
       * Trains a new machine for multi-class classification. If the number of
       * classes in data is 2, then the assigned labels will be -1 and +1. If
       * the number of classes is greater than 2, labels are picked starting
       * from 1 (i.e., 1, 2, 3, 4, etc.). If what you want is regression, the
       * size of the input data array should be 1.
       */
      boost::shared_ptr<bob::machine::SupportVector> train
        (const std::vector<blitz::Array<double,2> >& data) const;

      /**
       * This version accepts scaling parameters that will be applied
       * column-wise to the input data.
       */
      boost::shared_ptr<bob::machine::SupportVector> train
        (const std::vector<blitz::Array<double,2> >& data, 
         const blitz::Array<double,1>& input_subtract,
         const blitz::Array<double,1>& input_division) const;

      /**
       * Getters and setters for all parameters
       */
      bob::machine::SupportVector::svm_t getSvmType() const { return (bob::machine::SupportVector::svm_t)m_param.svm_type; }
      void setSvmType(bob::machine::SupportVector::svm_t v) { m_param.svm_type = v; }

      bob::machine::SupportVector::kernel_t getKernelType() const { return (bob::machine::SupportVector::kernel_t)m_param.kernel_type; }
      void setKernelType(bob::machine::SupportVector::kernel_t v) { m_param.kernel_type = v; }

      int getDegree() const { return m_param.degree; }
      void setDegree(int v) { m_param.degree = v; }

      double getGamma() const { return m_param.gamma; }
      void setGamma(double v) { m_param.gamma = v; }

      double getCoef0() const { return m_param.coef0; }
      void setCoef0(double v) { m_param.coef0 = v; }

      double getCacheSizeInMB() const { return m_param.cache_size; }
      void setCacheSizeInMb(double v) { m_param.cache_size = v; }

      double getStopEpsilon() const { return m_param.eps; }
      void setStopEpsilon(double v) { m_param.eps = v; }

      double getCost() const { return m_param.C; }
      void setCost(double v) { m_param.C = v; }

      double getNu() const { return m_param.nu; }
      void setNu(double v) { m_param.nu = v; }

      double getLossEpsilonSVR() const { return m_param.p; }
      void setLossEpsilonSVR(double v) { m_param.p = v; }

      bool getUseShrinking() const { return m_param.shrinking; }
      void setUseShrinking(bool v) { m_param.shrinking = v; }

      bool getProbabilityEstimates() const 
      { return m_param.probability; }
      void setProbabilityEstimates(bool v) 
      { m_param.probability = v; }

    private: //representation

      svm_parameter m_param; ///< training parametrization for libsvm
      
  };

}}

#endif /* BOB_TRAINER_SVMTRAINER_H */
