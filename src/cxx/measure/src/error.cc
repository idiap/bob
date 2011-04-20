/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Implements the error evaluation routines 
 */

#include "error/eval.h"
#include "core/blitz_compat.h"

namespace err = Torch::error;

std::pair<double, double> err::farfrr(const blitz::Array<double,1>& negatives, 
    const blitz::Array<double,1>& positives, double threshold) {
  blitz::sizeType total_negatives = negatives.extent(blitz::firstDim);
  blitz::sizeType total_positives = positives.extent(blitz::firstDim);
  blitz::sizeType false_accepts = total_negatives - blitz::count(negatives < threshold);
  blitz::sizeType false_rejects = total_positives - blitz::count(positives >= threshold);
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
    retval(0,i) = ratios.first;
    retval(1,i) = ratios.second;
  }
  return retval;
}

blitz::Array<double,2> err::epc
(const blitz::Array<double,1>& dev_negatives,
 const blitz::Array<double,1>& dev_positives, 
 const blitz::Array<double,1>& test_negatives,
 const blitz::Array<double,1>& test_positives, size_t points) {
  double step = 1.0/(points-1);
  blitz::Array<double,2> retval(2, points);
  for (int i=0; i<(int)points; ++i) {
    double alpha = i*step;
    retval(0,i) = alpha;
    double threshold = err::minWeightedErrorRateThreshold(dev_negatives, 
        dev_positives, alpha);
    std::pair<double, double> ratios =
      err::farfrr(test_negatives, test_positives, threshold);
    retval(1,i) = (ratios.first + ratios.second) / 2;
  }
  return retval;
}
