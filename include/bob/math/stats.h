/**
 * @file bob/math/stats.h
 * @date Mon Jun 20 11:47:58 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Some statistical base methods
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

#ifndef BOB_MATH_STATS_H 
#define BOB_MATH_STATS_H

#include <blitz/array.h>
#include <bob/core/assert.h>
#include <vector>

namespace bob { namespace math {
/**
 * @ingroup MATH
 * @{
 */

    /**
     * @brief Computes the scatter matrix of a 2D array considering data is
     * organized row-wise (each sample is a row, each feature is a column).
     * Outputs the sample mean M and the scatter matrix S.
     *
     * @warning No checks are performed on the array sizes and is recommended
     * only in scenarios where you have previously checked conformity and is
     * focused only on speed.
     *
     * This version of the method also returns the sample mean of the array.
     */
    template<typename T>
    void scatter_(const blitz::Array<T,2>& A, blitz::Array<T,2>& S, 
        blitz::Array<T,1>& M) {
      blitz::firstIndex i;
      blitz::secondIndex j;
      blitz::Range a = blitz::Range::all();

      M = blitz::mean(A(j,i),j);
      S = 0;

      blitz::Array<T,1> buffer(A.extent(1));
      for (int z=0; z<A.extent(0); ++z) {
        buffer = A(z,a) - M;
        S += buffer(i) * buffer(j); //outer product
      }
    }

    /**
     * @brief Computes the scatter matrix of a 2D array considering data is
     * organized row-wise (each sample is a row, each feature is a column).
     * Outputs the sample mean M and the scatter matrix S.
     *
     * The input and output data have their sizes checked and this method will
     * raise an appropriate exception if that is not cased. If you know that
     * the input and output matrices conform, use the scatter_() variant.
     *
     * This version of the method also returns the sample mean of the array.
     */
    template<typename T>
    void scatter(const blitz::Array<T,2>& A, blitz::Array<T,2>& S, 
        blitz::Array<T,1>& M) {

      // Check output
      bob::core::array::assertSameDimensionLength(A.extent(1), M.extent(0));
      bob::core::array::assertSameDimensionLength(A.extent(1), S.extent(0));
      bob::core::array::assertSameDimensionLength(A.extent(1), S.extent(1));

      scatter_<T>(A, S, M);
    }

    /**
     * @brief Computes the scatter matrix of a 2D array considering data is
     * organized row-wise (each sample is a row, each feature is a column).
     * Outputs the sample scatter matrix S.
     *
     * @warning No checks are performed on the array sizes and is recommended
     * only in scenarios where you have previously checked conformity and is
     * focused only on speed.
     */
    template<typename T>
    void scatter_(const blitz::Array<T,2>& A, blitz::Array<T,2>& S) {
      blitz::Array<T,1> M;
      scatter_<T>(A, S, M);
    }

    /**
     * @brief Computes the scatter matrix of a 2D array considering data is
     * organized row-wise (each sample is a row, each feature is a column).
     * Outputs the sample scatter matrix S.
     *
     * The input and output data have their sizes checked and this method will
     * raise an appropriate exception if that is not cased. If you know that
     * the input and output matrices conform, use the scatter_() variant.
     */
    template<typename T>
    void scatter(const blitz::Array<T,2>& A, blitz::Array<T,2>& S) {
      blitz::Array<T,1> M;
      scatter<T>(A, S, M);
    }


    namespace detail {
      /** 
       * @brief Evaluates, in a single loop, the overall (or grand) mean 'm',
       * the individual class means 'm_k' and computes the total number of 
       * elements in each class 'N'. 
       */ 
      template <typename T>
      void evalMeans(const std::vector<blitz::Array<T,2> >& data, 
        blitz::Array<T,1>& m, blitz::Array<T,2>& m_k, 
        blitz::Array<T,1>& N) 
      { 
        blitz::Range a = blitz::Range::all(); 
        for (size_t k=0; k<data.size(); ++k) { //class loop 
          N(k) = data[k].extent(0); 
          for (int example=0; example<data[k].extent(0); ++example) { 
            blitz::Array<T,1> buffer(data[k](example,a)); 
            m_k(a,k) += buffer; 
            m += buffer; 
          }
          
          m_k(a,k) /= N(k); 
        }
        
        m /= sum(N); 
      }   

    }

    /**
     * @brief Calculates the within and between class scatter matrices Sw and 
     * Sb. Returns those matrices and the overall means vector (m).
     *
     * Strategy implemented:
     * 1. Evaluate the overall mean (m), class means (m_k) and the total class
     *    counts (N).
     * 2. Evaluate Sw and Sb using normal loops.
     *
     * Note that Sw and Sb, in this implementation, will be normalized by N-1
     * (number of samples) and K (number of classes). This procedure makes
     * the eigen values scaled by (N-1)/K, effectively increasing their 
     * values. The main motivation for this normalization are numerical 
     * precision concerns with the increasing number of samples causing a 
     * rather large Sw matrix. A normalization strategy mitigates this 
     * problem. The eigen vectors will see no effect on this normalization as
     * they are normalized in the euclidean sense (||a|| = 1) so that does 
     * not change those.
     *
     * This method was designed based on the previous design at 
     * torch3Vision 2.1, by SM.
     *
     * @warning No checks are performed on the array sizes and is recommended
     * only in scenarios where you have previously checked conformity and is
     * focused only on speed.
     */
    template <typename T>
    void scatters_(const std::vector<blitz::Array<T,2> >& data,
      blitz::Array<T,2>& Sw, blitz::Array<T,2>& Sb,
      blitz::Array<T,1>& m)
    {
      // checks for data shape should have been done before...
      const int n_features = data[0].extent(1);

      m = 0; //overall mean
      blitz::Array<T,2> m_k(n_features, data.size());
      m_k = 0; //class means
      blitz::Array<T,1> N(data.size());
      N = 0; //class counts

      // Compute the means
      detail::evalMeans(data, m, m_k, N); 

      blitz::firstIndex i;
      blitz::secondIndex j;
      blitz::Range a = blitz::Range::all();

      // between class scatter Sb
      Sb = 0;
      blitz::Array<T,1> buffer(n_features); //tmp buffer for speed-up
      for (size_t k=0; k<data.size(); ++k) { //class loop
        buffer = m - m_k(a,k);
        Sb += N(k) * buffer(i) * buffer(j); //Bishop's Eq. 4.46
      }

      // within class scatter Sw
      Sw = 0;
      for (size_t k=0; k<data.size(); ++k) { //class loop
        for (int example=0; example<data[k].extent(0); ++example) {
          buffer = data[k](example,a) - m_k(a,k);
          Sw += buffer(i) * buffer(j); //outer product
        }
      }
    }

    /**
     * @brief Calculates the within and between class scatter matrices Sw and 
     * Sb. Returns those matrices and the overall means vector (m).
     *
     * Strategy implemented:
     * 1. Evaluate the overall mean (m), class means (m_k) and the total class
     *    counts (N).
     * 2. Evaluate Sw and Sb using normal loops.
     *
     * Note that Sw and Sb, in this implementation, will be normalized by N-1
     * (number of samples) and K (number of classes). This procedure makes
     * the eigen values scaled by (N-1)/K, effectively increasing their 
     * values. The main motivation for this normalization are numerical 
     * precision concerns with the increasing number of samples causing a 
     * rather large Sw matrix. A normalization strategy mitigates this 
     * problem. The eigen vectors will see no effect on this normalization as
     * they are normalized in the euclidean sense (||a|| = 1) so that does 
     * not change those.
     *
     * This method was designed based on the previous design at 
     * torch3Vision 2.1, by SM.
     */
    template <typename T>
    void scatters(const std::vector<blitz::Array<T,2> >& data,
      blitz::Array<T,2>& Sw, blitz::Array<T,2>& Sb,
      blitz::Array<T,1>& m)
    {
      // Check output
      for (size_t i=0; i<data.size(); ++i)
        bob::core::array::assertSameDimensionLength(data[i].extent(1), m.extent(0));
      bob::core::array::assertSameDimensionLength(m.extent(0), Sw.extent(0));
      bob::core::array::assertSameDimensionLength(m.extent(0), Sw.extent(1));
      bob::core::array::assertSameDimensionLength(m.extent(0), Sb.extent(0));
      bob::core::array::assertSameDimensionLength(m.extent(0), Sb.extent(1));

      scatters_<T>(data, Sw, Sb, m);
    }

    /**
     * @brief Calculates the within and between class scatter matrices Sw and 
     * Sb. Returns those matrices.
     *
     * Strategy implemented:
     * 1. Evaluate the overall mean (m), class means (m_k) and the total class
     *    counts (N).
     * 2. Evaluate Sw and Sb using normal loops.
     *
     * Note that Sw and Sb, in this implementation, will be normalized by N-1
     * (number of samples) and K (number of classes). This procedure makes
     * the eigen values scaled by (N-1)/K, effectively increasing their 
     * values. The main motivation for this normalization are numerical 
     * precision concerns with the increasing number of samples causing a 
     * rather large Sw matrix. A normalization strategy mitigates this 
     * problem. The eigen vectors will see no effect on this normalization as
     * they are normalized in the euclidean sense (||a|| = 1) so that does 
     * not change those.
     *
     * This method was designed based on the previous design at 
     * torch3Vision 2.1, by SM.
     */
    template<typename T>
    void scatters_(const std::vector<blitz::Array<T,2> >& data,
      blitz::Array<T,2>& Sw, blitz::Array<T,2>& Sb)
    {
      blitz::Array<T,1> M(data[0].extent(1));
      scatters_<T>(data, Sw, Sb, M);
    }

    /**
     * @brief Calculates the within and between class scatter matrices Sw and 
     * Sb. Returns those matrices.
     *
     * Strategy implemented:
     * 1. Evaluate the overall mean (m), class means (m_k) and the total class
     *    counts (N).
     * 2. Evaluate Sw and Sb using normal loops.
     *
     * Note that Sw and Sb, in this implementation, will be normalized by N-1
     * (number of samples) and K (number of classes). This procedure makes
     * the eigen values scaled by (N-1)/K, effectively increasing their 
     * values. The main motivation for this normalization are numerical 
     * precision concerns with the increasing number of samples causing a 
     * rather large Sw matrix. A normalization strategy mitigates this 
     * problem. The eigen vectors will see no effect on this normalization as
     * they are normalized in the euclidean sense (||a|| = 1) so that does 
     * not change those.
     *
     * This method was designed based on the previous design at 
     * torch3Vision 2.1, by SM.
     *
     * @warning No checks are performed on the array sizes and is recommended
     * only in scenarios where you have previously checked conformity and is
     * focused only on speed.
     */
    template<typename T>
    void scatters(const std::vector<blitz::Array<T,2> >& data,
      blitz::Array<T,2>& Sw, blitz::Array<T,2>& Sb)
    {
      blitz::Array<T,1> M(data[0].extent(1));
      scatters<T>(data, Sw, Sb, M);
    }

/**
 * @}
 */
}}

#endif /* BOB_MATH_STATS_H */
