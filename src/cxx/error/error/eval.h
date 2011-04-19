/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 * @date Tue 19 Apr 07:42:35 2011 
 *
 * @brief A set of methods that evaluates error from score sets
 */

#ifndef TORCH_ERROR_EVALUATOR_H 
#define TORCH_ERROR_EVALUATOR_H

#include <blitz/array.h>
#include <utility>

namespace Torch { namespace error {

  /**
   * Calculates the FA ratio and the FR ratio given positive and negative
   * scores and a threshold. 'positives' holds the score information for
   * samples that are labelled to belong to a certain class (a.k.a., "signal"
   * or "client"). 'negatives' holds the score information for samples that are
   * labelled *not* to belong to the class (a.k.a., "noise" or "impostor").
   *
   * It is expected that 'positive' scores are, at least by design, greater
   * than 'negative' scores. So, every positive value that falls bellow the
   * threshold is considered a false-rejection (FR). 'negative' samples that
   * fall above the threshold are considered a false-accept (FA). 
   *
   * Positives that fall on the threshold (exactly) are considered correctly
   * classified. Negatives that fall on the threshold (exactly) are considered
   * *incorrectly* classified. This equivalent to setting the comparision like
   * this pseudo-code:
   *
   * foreach (positive as K) if K < threshold: falseRejectionCount += 1
   * foreach (negative as K) if K >= threshold: falseAcceptCount += 1
   *
   * The 'threshold' value does not necessarily have to fall in the range
   * covered by the input scores (negatives and positives altogether), but if
   * it does not, the output will be either (1.0, 0.0) or (0.0, 1.0)
   * depending on the side the threshold falls.
   *
   * The output is in form of a std::pair of two double-precision real numbers.
   * The numbers range from 0 to 1. The first element of the pair is the
   * false-accept ratio. The second element of the pair is the false-rejection
   * ratio.
   *
   * It is possible that scores are inverted in the negative/positive sense. In
   * some setups the designer may have setup the system so 'positive' samples
   * have a smaller score than the 'negative' ones. In this case, make sure you
   * normalize the scores so positive samples have greater scores before
   * feeding them into this method.
   */
  std::pair<double, double> farfrr(const blitz::Array<double,1>& negatives, 
      const blitz::Array<double,1>& positives, double threshold);

  /**
   * This method returns a blitz::Array composed of booleans that pin-point
   * which positives where correctly classified in a 'positive' score sample,
   * given a threshold. It runs the formula:
   *
   * foreach (element k in positive) 
   *   if positive[k] >= threshold: returnValue[k] = true
   *   else: returnValue[k] = false
   */
  inline blitz::Array<bool,1> classifyPositives
    (const blitz::Array<double,1>& positives, double threshold) {
      return blitz::Array<bool,1>(positives >= threshold);
    }

  /**
   * This method returns a blitz::Array composed of booleans that pin-point
   * which negatives where correctly classified in a 'negative' score sample,
   * given a threshold. It runs the formula:
   *
   * foreach (element k in negative) 
   *   if negative[k] < threshold: returnValue[k] = true
   *   else: returnValue[k] = false
   */
  inline blitz::Array<bool,1> classifyNegatives
    (const blitz::Array<double,1>& negatives, double threshold) {
      return blitz::Array<bool,1>(negatives < threshold);
    }

  /**
   * This method can calculate a threshold based on a set of scores (positives
   * and negatives) given a certain minimization criteria, input as a
   * functional predicate. For a discussion on 'positive' and 'negative' see
   * Torch::error::fafr().
   *
   * The predicate method gives back the current minimum given false-acceptance
   * (FA) and false-rejection (FR) ratios for the input data. As a predicate,
   * it has to be a non-member method or a pre-configured functor where we can
   * use operator(). The API for the method is:
   *
   * double predicate(double fa_ratio, double fr_ratio);
   *
   * The minimization is carried out in a recursive manner. First, we identify
   * the threshold that minimizes the predicate given a set of N (N=100)
   * thresholds between the min(negatives, positives) and the max(negatives,
   * positives). If the minima lies in a range of values, the center value is
   * picked up. 
   *
   * In a second round of minimization new minimum and maximum bounds are
   * defined based on the center value plus/minus the step (max-min/N) and a
   * new minimization round is restarted for N samples within the new bounds.
   *
   * The procedure continues until all calculated predicates in a given round
   * give the same minima. At this point, the center threshold is picked up and
   * returned.
   */
  double minimizingThreshold(const blitz::Array<double,1>& negatives,
      const blitz::Array<double,1>& positives,
      double (*predicate)(double, double));

  /**
   * Calculates the threshold that is, as close as possible, to the
   * equal-error-rate (EER) given the input data. The EER should be the point
   * where the FAR equals the FRR. Graphically, this would be equivalent to the
   * intersection between the R.O.C. (or D.E.T.) curves and the identity.
   */
  double eerThreshold(const blitz::Array<double,1>& negatives,
      const blitz::Array<double,1>& positives);

  /**
   * Calculates the threshold that minimizes the half-total error rate (HTER),
   * given the input data. An optional parameter 'cost' determines the relative
   * importance between false-accepts and false-rejections. This number should
   * be between 0 and 1 and will be clipped to those extremes.
   *
   * The value to minimize becomes:
   *
   * HTER_cost = [cost * FAR] + [(1-cost) * FRR]
   *
   * The higher the cost, the higher the importance given to *not* making
   * mistakes classifying negatives/noise/impostors.
   */
  double minHterThreshold(const blitz::Array<double,1>& negatives,
      const blitz::Array<double,1>& positives, double cost=0.5);

  /**
   * Calculates the ROC curve given a set of positive and negative scores and a
   * number of desired points. Returns a two-dimensional blitz::Array of
   * doubles that express the X and Y coordinates in this order.
   */
  blitz::Array<double,2> roc
    (const blitz::Array<double,1>& negatives,
     const blitz::Array<double,1>& positives, size_t points);

  /**
   * Calculates the DET curve given a set of positive and negative scores and a
   * number of desired points. Returns a two-dimensional blitz::Array of
   * doubles that express the X and Y coordinates in this order.
   */
  blitz::Array<double,2> det
    (const blitz::Array<double,1>& negatives,
     const blitz::Array<double,1>& positives, size_t points);

  /**
   * Calculates the EPC curve given a set of positive and negative scores and a
   * number of desired points. Returns a two-dimensional blitz::Array of
   * doubles that express the X and Y coordinates in this order.
   *
   * The EPC curve defines the minimum HTER with a varying cost between 0 and
   * 1. 
   */
  blitz::Array<double,2> epc
    (const blitz::Array<double,1>& negatives,
     const blitz::Array<double,1>& positives, size_t points);

}}

#endif /* TORCH_ERROR_EVALUATOR_H */
