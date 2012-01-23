/**
 * @file cxx/measure/measure/GaborJetSimilarity.h
 * @date
 * @author
 *
 * @brief A set of methods that evaluates error from score sets
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
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

#ifndef BOB_MEASURE_GABOR_JET_SIMILARITY_H
#define BOB_MEASURE_GABOR_JET_SIMILARITY_H

#include <core/array_assert.h>
#include <blitz/array.h>
#include <numeric>

namespace bob { namespace measure {

  //! base class for Gabor jet similarity functions
  class GaborJetSimilarity{
    protected:
      GaborJetSimilarity(){}
      virtual ~GaborJetSimilarity(){}

    public:
      //! The similarity between two Gabor jets
      virtual double similarity(const blitz::Array<double,2>& jet1, const blitz::Array<double,2>& jet2) = 0;

  };

  //! The default Gabor jet similarity function, which is the normalized scalar product,
  //! also known as the cosine measure
  class ScalarProductSimilarity : GaborJetSimilarity{
    public:
      ScalarProductSimilarity() : GaborJetSimilarity(){}
      virtual ~ScalarProductSimilarity(){}

      virtual double similarity(const blitz::Array<double,2>& jet1, const blitz::Array<double,2>& jet2);
  };

  //! The default Gabor jet similarity function, which is the normalized scalar product,
  //! also known as the cosine measure
  class CanberraSimilarity : GaborJetSimilarity{
    public:
      CanberraSimilarity() : GaborJetSimilarity(){}
      virtual ~CanberraSimilarity(){}

      virtual double similarity(const blitz::Array<double,2>& jet1, const blitz::Array<double,2>& jet2);
  };
} }

#endif // BOB_MEASURE_GABOR_JET_SIMILARITY_H
