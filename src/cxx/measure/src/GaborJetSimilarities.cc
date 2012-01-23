/**
 * @file cxx/measure/src/GaborJetSimilarities.cc
 * @date
 * @author
 *
 * @brief Implements the Gabor jet similarity functions
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

#include "measure/GaborJetSimilarities.h"

double bob::measure::ScalarProductSimilarity::similarity(const blitz::Array<double,2>& jet1, const blitz::Array<double,2>& jet2){
  const blitz::Array<double,1> j1 = jet1(0,blitz::Range::all()), j2 = jet2(0,blitz::Range::all());
  bob::core::array::assertCZeroBaseContiguous(j1);
  bob::core::array::assertCZeroBaseContiguous(j2);
  bob::core::array::assertSameShape(j1,j2);
  return std::inner_product(j1.begin(), j1.end(), j2.begin(), 0.);
}

double bob::measure::CanberraSimilarity::similarity(const blitz::Array<double,2>& jet1, const blitz::Array<double,2>& jet2){
  const blitz::Array<double,1> j1 = jet1(0,blitz::Range::all()), j2 = jet2(0,blitz::Range::all());
  bob::core::array::assertCZeroBaseContiguous(j1);
  bob::core::array::assertCZeroBaseContiguous(j2);
  bob::core::array::assertSameShape(j1,j2);
  double sim = 0.;
  unsigned size = j1.shape()[0];
  for (unsigned j = size; j--;){
    sim += 1. - std::abs(j1(j) - j2(j)) / (j1(j) + j2(j));
  }
  return sim / size;
}
