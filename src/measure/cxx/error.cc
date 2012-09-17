/**
 * @file measure/cxx/error.cc
 * @date Wed Apr 20 08:02:30 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Implements the error evaluation routines
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

#include <stdexcept>
#include <algorithm>
#include "bob/measure/error.h"
#include "bob/core/blitz_compat.h"
#include "bob/core/Exception.h"

namespace err = bob::measure;

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

double err::farThreshold(const blitz::Array<double,1>& negatives,
  const blitz::Array<double,1>&, double far_value) {
  // check the parameters are valid
  if (far_value < 0. || far_value > 1.){
    std::ostringstream s;
    throw bob::core::InvalidArgumentException("far_value", far_value, 0., 1.);
  }
  if (negatives.size() < 2){
    throw bob::core::InvalidArgumentException("The number of negatives must at least be two!");
  }

  // sort negative scores ascendingly
  std::vector<double> negatives_(negatives.shape()[0]);
  std::copy(negatives.begin(), negatives.end(), negatives_.begin());
  std::sort(negatives_.begin(), negatives_.end(), std::less<double>());

  // compute position of the threshold
  double crr = 1.-far_value; // (Correct Rejection Rate; = 1 - FAR)
  double crr_index = crr * negatives_.size();
  // compute the index above the current CRR value
  int index = (int)std::floor(crr_index);

  // correct index if we have multiple score values at the requested position
  while (index && negatives_[index] == negatives_[index-1]) --index;

  // we compute a correction term
  double correction;
  if (index){
    // assure that we are in the middle of two cases
    correction = 0.5 * (negatives_[index] - negatives_[index-1]);
  } else {
    // add an overall correction term
    correction = 0.5 * (negatives_.back() - negatives_.front()) / negatives_.size();
  }

  return negatives_[index] - correction;
}

double err::frrThreshold(const blitz::Array<double,1>&,
  const blitz::Array<double,1>& positives, double frr_value) {

  // check the parameters are valid
  if (frr_value < 0. || frr_value > 1.){
    std::ostringstream s;
    throw bob::core::InvalidArgumentException("frr_value", frr_value, 0., 1.);
  }
  if (positives.size() < 2){
    throw bob::core::InvalidArgumentException("The number of positives must at least be two!");
  }

  // sort positive scores descendingly
  std::vector<double> positives_(positives.shape()[0]);
  std::copy(positives.begin(), positives.end(), positives_.begin());
  std::sort(positives_.begin(), positives_.end(), std::greater<double>());

  // compute position of the threshold
  double car = 1.-frr_value; // (Correct Acceptance Rate; = 1 - FRR)
  double car_index = car * positives_.size();
  // compute the index above the current CRR value
  int index = (int)std::floor(car_index);

  // correct index if we have multiple score values at the requested position
  while (index && positives_[index] == positives_[index-1]) --index;

  // we compute a correction term to assure that we are in the middle of two cases
  // we compute a correction term
  double correction;
  if (index){
    // assure that we are in the middle of two cases
    correction = 0.5 * (positives_[index-1] - positives_[index]);
  } else {
    // add an overall correction term
    correction = 0.5 * (positives_.front() - positives_.back()) / positives_.size();
  }

  return positives_[index] + correction;
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
    return (m_weight*far) + ((1.0-m_weight)*frr);
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
  double step = (max-min)/((double)points-1.0);
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
 * This function computes the ROC coordinates for the given positive and
 * negative values at the given FAR positions.
 *
 * @param negatives Impostor scores
 * @param positives Client scores
 * @param far_list  The list of FAR values where the FRR should be calculated
 *
 * @return The ROC curve with the FAR in the first row and the FRR in the second.
 */
blitz::Array<double,2> err::roc_for_far(const blitz::Array<double,1>& negatives,
 const blitz::Array<double,1>& positives, const blitz::Array<double,1>& far_list) {
  int n_points = far_list.extent(0);

  // sort negative scores ascendingly
  std::vector<double> negatives_(negatives.extent(0));;
  std::copy(negatives.begin(), negatives.end(), negatives_.begin());
  std::sort(negatives_.begin(), negatives_.end());
  // sort positive scores ascendingly
  std::vector<double> positives_(positives.extent(0));;
  std::copy(positives.begin(), positives.end(), positives_.begin());
  std::sort(positives_.begin(), positives_.end());

  // do some magic to compute the FRR list
  blitz::Array<double,2> retval(2, n_points);

  // index into the FAR and FRR list
  int far_index = n_points-1;
  int pos_index = 0, neg_index = 0;
  int n_pos = positives_.size(), n_neg = negatives_.size();

  // iterators into the result lists
  std::vector<double>::const_iterator pos_it = positives_.begin(), neg_it = negatives_.begin();
  // do some fast magic to compute the FRR values ;-)
  do{
    // check whether the current positive value is less than the current negative one
    if (*pos_it <= *neg_it){
      // increase the positive count
      ++pos_index;
      // go to the next positive value
      ++pos_it;
    }else{
      // increase the negative count
      ++neg_index;
      // go to the next negative value
      ++neg_it;
    }
    // check, if we have reached a new FAR limit,
    // i.e. if the relative number of negative similarities is greater than 1-FAR (which is the CRR)
    if ((double)neg_index / (double)n_neg > 1. - far_list(far_index)){
      // copy the far value
      retval(0,far_index) = far_list(far_index);
      // calculate the CAR (i.e., 1.-frr) for the current FAR
      retval(1,far_index) = 1. - (double)pos_index / (double)n_pos;
      // go to the next FAR value
      --far_index;
    }

  // do this, as long as there are elements in both lists left and not all FRR elements where calculated yet
  } while (pos_it != positives_.end() && neg_it != negatives_.end() && far_index >= 0);

  // check if all CAR values have been set
  if (far_index >= 0){
    // walk to the end of both lists; at least one of both lists should already have reached its limit.
    pos_index += positives_.end() - pos_it;
    neg_index += negatives_.end() - neg_it;
    // fill in the remaining elements of the CAR list
    do {
      // copy the FAR value
      retval(0,far_index) = far_list(far_index);
      // check if the criterion is fulfilled (should be, as long as the lowest far is not below 0)
      if ((double)neg_index / (double)n_neg > 1. - far_list(far_index)){
        // calculate the CAR (i.e., 1.-FRR) for the current FAR
        retval(1,far_index) = 1. - (double)pos_index / (double)n_pos;
      } else {
        // set CAR to zero (this should never happen, but might be due to numerical issues)
        retval(1,far_index) = 0.;
      }
    } while (far_index--);
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
