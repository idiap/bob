/**
 * @file bob/trainer/FisherLDATrainer.h
 * @date Sat Jun 4 21:38:59 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Implements a multi-class Fisher/LDA linear machine Training using
 * Singular Value Decomposition (SVD). For more information on Linear Machines
 * and associated methods, please consult Bishop, Machine Learning and Pattern
 * Recognition chapter 4.
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

#ifndef BOB_TRAINER_FISHERLDATRAINER_H
#define BOB_TRAINER_FISHERLDATRAINER_H

#include <vector>
#include <bob/machine/LinearMachine.h>

namespace bob { namespace trainer {

  /**
   * @ingroup TRAINER
   * @{
   */

  /**
   * @brief Sets a linear machine to perform the Fisher/LDA decomposition.\n
   *
   * References:\n
   * 1. Bishop, Machine Learning and Pattern Recognition chapter 4.\n
   * 2. http://en.wikipedia.org/wiki/Linear_discriminant_analysis
   */
  class FisherLDATrainer {
    public: //api

      /**
       * @brief Initializes a new Fisher/LDA trainer. The training stage will
       * place the resulting fisher components in the linear machine and set it
       * up to extract the variable means automatically.
       *
       * @param strip_to_rank Specifies what is the final size of the prepared
       * LinearMachine. The default setting (<code>true</code>), makes the
       * trainer return only the K-1 eigen-values/vectors limiting the output
       * to the rank of Sw^1 Sb. If you set this value to <code>false</code>,
       * the it returns all eigen-values/vectors of Sw^1 Sb, including the ones
       * that are supposed to be zero.
       *
       * @param use_pinv If set (to <code>true</code>), then use the
       * pseudo-inverse to compute Sw^-1 and then a generalized eigen-value
       * decomposition instead of using the default (more stable) version of
       * the eigen-value decomposition, starting from Sb and Sw.
       */
      FisherLDATrainer(bool use_pinv = false, bool strip_to_rank = true);

      /**
       * @brief Destructor
       */
      virtual ~FisherLDATrainer();

      /**
       * @brief Copy constructor.
       */
      FisherLDATrainer(const FisherLDATrainer& other);

      /**
       * @brief Assignment operator
       */
      FisherLDATrainer& operator=(const FisherLDATrainer& other);

      /**
       * @brief Equal to
       */
      bool operator==(const FisherLDATrainer& other) const;
      
      /**
       * @brief Not equal to
       */
      bool operator!=(const FisherLDATrainer& other) const;

      /**
       * @brief Similar to
       */
      bool is_similar_to(const FisherLDATrainer& other, 
          const double r_epsilon=1e-5, const double a_epsilon=1e-8) const;

      /**
       * @brief Gets the pseudo-inverse flag
       */
      bool getUsePseudoInverse () const { return m_use_pinv; }

      /**
       * @brief Sets the pseudo-inverse flag
       */
      void setUsePseudoInverse (bool v) { m_use_pinv = v; }

      /**
       * @brief Gets the strip-to-rank flag
       */
      bool getStripToRank () const { return m_strip_to_rank; }

      /**
       * @brief Sets the strip-to-rank flag
       */
      void setStripToRank (bool v) { m_strip_to_rank = v; }

      /**
       * @brief Trains the LinearMachine to perform Fisher/LDA discrimination.
       * The resulting machine will have the eigen-vectors of the
       * Sigma-1 * Sigma_b product, arranged by decreasing energy.
       *
       * Each input arrayset represents data from a given input class.
       *
       * Note we set only the N-1 eigen vectors in the linear machine since the
       * last eigen value should be zero anyway. You can compress the machine
       * output further using resize() if necessary.
       */
      void train(bob::machine::LinearMachine& machine,
          const std::vector<blitz::Array<double,2> >& X) const;

      /**
       * @brief Trains the LinearMachine to perform Fisher/LDA discrimination.
       * The resulting machine will have the eigen-vectors of the
       * Sigma-1 * Sigma_b product, arranged by decreasing energy. You don't
       * need to sort the results. Also returns the eigen values of the
       * covariance matrix product so you can use that to choose which
       * components to keep.
       *
       * Each input arrayset represents data from a given input class.
       *
       * Note we set only the N-1 eigen vectors in the linear machine since the
       * last eigen value should be zero anyway. You can compress the machine
       * output further using resize() if necessary.
       */
      void train(bob::machine::LinearMachine& machine,
          blitz::Array<double,1>& eigen_values,
          const std::vector<blitz::Array<double,2> >& X) const;

      /**
       * @brief Returns the expected size of the output given the data.
       *
       * This number could be either K-1 (K = number of classes) or the number
       * of columns (features) in X, depending on the seetting of
       * <code>strip_to_rank</code>.
       */
      size_t output_size(const std::vector<blitz::Array<double,2> >& X) const;

    private:
      bool m_use_pinv; ///< use the 'pinv' method for LDA
      bool m_strip_to_rank; ///< return rank or full matrix
  };

  /**
   * @}
   */
}}

#endif /* BOB_TRAINER_FISHERLDATRAINER_H */
