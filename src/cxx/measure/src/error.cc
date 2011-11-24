/**
 * @file cxx/measure/src/error.cc
 * @date Wed Apr 20 08:02:30 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Implements the error evaluation routines
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

#include <stdexcept>
#include "measure/error.h"
#include "core/blitz_compat.h"

namespace err = Torch::measure;

std::pair<double, double> err::farfrr(const blitz::Array<double,1>& negatives, 
    const blitz::Array<double,1>& positives, double threshold) {
  blitz::sizeType total_negatives = negatives.extent(blitz::firstDim);
  blitz::sizeType total_positives = positives.extent(blitz::firstDim);
  blitz::sizeType false_accepts = blitz::count(negatives >= threshold);
  blitz::sizeType false_rejects = blitz::count(positives < threshold);
  if (!total_negatives) total_negatives = 1; //avoids division by zero
  if (!total_positives) total_positives = 1; //avoids division by zero
  return std::make_pair(false_accepts/(double)total_negatives,
      false_rejects/(double)total_positives);
}

double eer_predicate(double far, double frr) {
  return std::abs(far - frr);
}

double err::eerThreshold(const blitz::Array<double,1>& negatives,
    const blitz::Array<double,1>& positives) {
  return err::minimizingThreshold(negatives, positives, eer_predicate);
}

/**
 * Provides a functor predicate for weighted error calculation
 */
class weighted_error {
  
  double m_weight; ///< The weighting factor

  public: //api

  weighted_error(double weight): m_weight(weight) { 
    if (weight > 1.0) m_weight = 1.0;
    if (weight < 0.0) m_weight = 0.0;
  }

  inline double operator() (double far, double frr) const {
    return (m_weight*far) + ((1-m_weight)*frr);
  }

};

double err::minWeightedErrorRateThreshold
(const blitz::Array<double,1>& negatives,
 const blitz::Array<double,1>& positives, double cost) {
  weighted_error predicate(cost);
  return err::minimizingThreshold(negatives, positives, predicate);
}

blitz::Array<double,2> err::roc(const blitz::Array<double,1>& negatives,
 const blitz::Array<double,1>& positives, size_t points) {
  double min = std::min(blitz::min(negatives), blitz::min(positives));
  double max = std::max(blitz::max(negatives), blitz::max(positives));
  double step = (max-min)/(points-1);
  blitz::Array<double,2> retval(2, points);
  for (int i=0; i<(int)points; ++i) {
    std::pair<double, double> ratios =
      err::farfrr(negatives, positives, min + i*step);
    //note: inversion to preserve X x Y ordering (FRR x FAR)
    retval(0,i) = ratios.second;
    retval(1,i) = ratios.first;
  }
  return retval;
}

/**
 * The input to this function is a cumulative probability.  The output from
 * this function is the Normal deviate that corresponds to that probability.
 * For example: 
 *
 *  INPUT | OUTPUT
 * -------+--------
 *  0.001 | -3.090
 *  0.01  | -2.326
 *  0.1   | -1.282
 *  0.5   |  0.0
 *  0.9   |  1.282
 *  0.99  |  2.326
 *  0.999 |  3.090
 */
static double _ppndf(double p) {
  //some constants we need for the calculation. 
  //these come from the NIST implementation...
  static const double SPLIT = 0.42;
  static const double A0 = 2.5066282388;
  static const double A1 = -18.6150006252;
  static const double A2 = 41.3911977353;
  static const double A3 = -25.4410604963;
  static const double B1 = -8.4735109309;
  static const double B2 = 23.0833674374;
  static const double B3 = -21.0622410182;
  static const double B4 = 3.1308290983;
  static const double C0 = -2.7871893113;
  static const double C1 = -2.2979647913;
  static const double C2 = 4.8501412713;
  static const double C3 = 2.3212127685;
  static const double D1 = 3.5438892476;
  static const double D2 = 1.6370678189;
  static const double eps = 2.2204e-16;

  double retval;

  if (p >= 1.0) p = 1 - eps;
  if (p <= 0.0) p = eps;

  double q = p - 0.5;

  if (std::abs(q) <= SPLIT) {
    double r = q * q;
    retval = q * (((A3 * r + A2) * r + A1) * r + A0) /
      ((((B4 * r + B3) * r + B2) * r + B1) * r + 1.0);
  }
  else {
    //r = sqrt (log (0.5 - abs(q)));
    double r = (q > 0.0  ?  1.0 - p : p);
    if (r <= 0.0) throw std::underflow_error("measure::ppndf(): r <= 0.0!");
    r = sqrt ((-1.0) * log (r));
    retval = (((C3 * r + C2) * r + C1) * r + C0) / ((D2 * r + D1) * r + 1.0);
    if (q < 0) retval *= -1.0;
  }

  return retval;
}

namespace blitz {
  BZ_DECLARE_FUNCTION(_ppndf) ///< A blitz::Array binding
}

double err::ppndf (double value) { return _ppndf(value); }

blitz::Array<double,2> err::det(const blitz::Array<double,1>& negatives,
    const blitz::Array<double,1>& positives, size_t points) {
  blitz::Array<double,2> retval(2, points);
  retval = blitz::_ppndf(err::roc(negatives, positives, points));
  return retval;
}

blitz::Array<double,2> err::epc
(const blitz::Array<double,1>& dev_negatives,
 const blitz::Array<double,1>& dev_positives, 
 const blitz::Array<double,1>& test_negatives,
 const blitz::Array<double,1>& test_positives, size_t points) {
  double step = 1.0/((double)points-1.0);
  blitz::Array<double,2> retval(2, points);
  for (int i=0; i<(int)points; ++i) {
    double alpha = (double)i*step;
    retval(0,i) = alpha;
    double threshold = err::minWeightedErrorRateThreshold(dev_negatives, 
        dev_positives, alpha);
    std::pair<double, double> ratios =
      err::farfrr(test_negatives, test_positives, threshold);
    retval(1,i) = (ratios.first + ratios.second) / 2;
  }
  return retval;
}
