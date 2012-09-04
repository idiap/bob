/**
 * @file python/measure/src/error.cc
 * @date Wed Apr 20 08:19:36 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Implements python bindings to the bob configuration system
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

#include "bob/measure/error.h"
#include "bob/core/python/ndarray.h"

using namespace boost::python;

static tuple farfrr(
    const blitz::Array<double,1>& negatives,
    const blitz::Array<double,1>& positives,
    double threshold
){
  std::pair<double, double> retval = bob::measure::farfrr(negatives, positives, threshold);
  return make_tuple(retval.first, retval.second);
}

static blitz::Array<bool,1> bob_correctly_classified_positives(bob::python::const_ndarray positives, double threshold){
  return bob::measure::correctlyClassifiedPositives(positives.cast<double,1>(), threshold);
}

static blitz::Array<bool,1> bob_correctly_classified_negatives(bob::python::const_ndarray negatives, double threshold){
  return bob::measure::correctlyClassifiedNegatives(negatives.cast<double,1>(), threshold);
}


static double bob_eer_threshold(bob::python::const_ndarray negatives, bob::python::const_ndarray positives){
  return bob::measure::eerThreshold(negatives.cast<double,1>(), positives.cast<double,1>());
}

static double bob_min_weighted_error_rate_threshold(bob::python::const_ndarray negatives, bob::python::const_ndarray positives, const double costs){
  return bob::measure::minWeightedErrorRateThreshold(negatives.cast<double,1>(), positives.cast<double,1>(), costs);
}


static double bob_min_hter_threshold(bob::python::const_ndarray negatives, bob::python::const_ndarray positives){
  return bob::measure::minHterThreshold(negatives.cast<double,1>(), positives.cast<double,1>());
}

static double bob_far_threshold(bob::python::const_ndarray negatives, bob::python::const_ndarray positives, double far_value){
  return bob::measure::farThreshold(negatives.cast<double,1>(), positives.cast<double,1>(), far_value);
}

static blitz::Array<double,2> bob_roc(bob::python::const_ndarray negatives, bob::python::const_ndarray positives, int n_points){
  return bob::measure::roc(negatives.cast<double,1>(), positives.cast<double,1>(), n_points);
}

static blitz::Array<double,2> bob_roc_for_far(bob::python::const_ndarray negatives, bob::python::const_ndarray positives, bob::python::const_ndarray far_list){
  return bob::measure::roc_for_far(negatives.cast<double,1>(), positives.cast<double,1>(), far_list.cast<double,1>());
}

static blitz::Array<double,2> bob_det(bob::python::const_ndarray negatives, bob::python::const_ndarray positives, int n_points){
  return bob::measure::det(negatives.cast<double,1>(), positives.cast<double,1>(), n_points);
}

static blitz::Array<double,2> bob_epc(bob::python::const_ndarray dev_negatives, bob::python::const_ndarray dev_positives,
                                      bob::python::const_ndarray test_negatives, bob::python::const_ndarray test_positives, int n_points){
  return bob::measure::epc(dev_negatives.cast<double,1>(), dev_positives.cast<double,1>(), test_negatives.cast<double,1>(), test_positives.cast<double,1>(), n_points);
}

void bind_measure_error() {
  def(
    "farfrr",
    &farfrr,
    (arg("negatives"), arg("positives"), arg("threshold")),
    "Calculates the FA ratio and the FR ratio given positive and negative scores and a threshold. 'positives' holds the score information for samples that are labelled to belong to a certain class (a.k.a., 'signal' or 'client'). 'negatives' holds the score information for samples that are labelled *not* to belong to the class (a.k.a., 'noise' or 'impostor').\n\nIt is expected that 'positive' scores are, at least by design, greater than 'negative' scores. So, every positive value that falls bellow the threshold is considered a false-rejection (FR). 'negative' samples that fall above the threshold are considered a false-accept (FA).\n\nPositives that fall on the threshold (exactly) are considered correctly classified. Negatives that fall on the threshold (exactly) are considered *incorrectly* classified. This equivalent to setting the comparision like this pseudo-code:\n\nforeach (positive as K) if K < threshold: falseRejectionCount += 1\nforeach (negative as K) if K >= threshold: falseAcceptCount += 1\n\nThe 'threshold' value does not necessarily have to fall in the range covered by the input scores (negatives and positives altogether), but if it does not, the output will be either (1.0, 0.0) or (0.0, 1.0) depending on the side the threshold falls.\n\nThe output is in form of a std::pair of two double-precision real numbers. The numbers range from 0 to 1. The first element of the pair is the false-accept ratio. The second element of the pair is the false-rejection ratio.\n\nIt is possible that scores are inverted in the negative/positive sense. In some setups the designer may have setup the system so 'positive' samples have a smaller score than the 'negative' ones. In this case, make sure you normalize the scores so positive samples have greater scores before feeding them into this method."
  );

  def(
    "correctly_classified_positives",
    &bob_correctly_classified_positives,
    (arg("positives"), arg("threshold")),
    "This method returns a blitz::Array composed of booleans that pin-point which positives where correctly classified in a 'positive' score sample, given a threshold. It runs the formula: foreach (element k in positive) if positive[k] >= threshold: returnValue[k] = true else: returnValue[k] = false"
  );

  def(
    "correctly_classified_negatives",
    &bob_correctly_classified_negatives,
    (arg("negatives"), arg("threshold")),
    "This method returns a blitz::Array composed of booleans that pin-point which negatives where correctly classified in a 'negative' score sample, given a threshold. It runs the formula: foreach (element k in negative) if negative[k] < threshold: returnValue[k] = true else: returnValue[k] = false"
  );

  def(
    "eer_threshold",
    &bob_eer_threshold,
    (arg("negatives"), arg("positives")),
    "Calculates the threshold that is as close as possible to the equal-error-rate (EER) given the input data. The EER should be the point where the FAR equals the FRR. Graphically, this would be equivalent to the intersection between the ROC (or DET) curves and the identity."
  );

  def(
    "min_weighted_error_rate_threshold",
    &bob_min_weighted_error_rate_threshold,
    (arg("negatives"), arg("positives"), arg("cost")),
    "Calculates the threshold that minimizes the error rate, given the input data. An optional parameter 'cost' determines the relative importance between false-accepts and false-rejections. This number should be between 0 and 1 and will be clipped to those extremes. The value to minimize becomes: ER_cost = [cost * FAR] + [(1-cost) * FRR]. The higher the cost, the higher the importance given to *not* making mistakes classifying negatives/noise/impostors."
  );

  def(
    "min_hter_threshold",
    &bob_min_hter_threshold,
    (arg("negatives"), arg("positives")),
    "Calculates the min_weighted_error_rate_threshold() when the cost is 0.5."
  );

  def(
    "far_threshold",
    &bob_far_threshold,
    (arg("negatives"), arg("positives"), arg("far_value")=0.001),
    "Calculates the score threshold at the position, the far reaches the given limit"
  );

  def(
    "roc",
    &bob_roc,
    (arg("negatives"), arg("positives"), arg("n_points")),
    "Calculates the ROC curve given a set of positive and negative scores and a desired number of points. Returns a two-dimensional blitz::Array of doubles that express the X (FRR) and Y (FAR) coordinates in this order. The points in which the ROC curve are calculated are distributed uniformily in the range [min(negatives, positives), max(negatives, positives)]."
  );

  def(
    "roc_for_far",
    &bob_roc_for_far,
    (arg("negatives"), arg("positives"), arg("far_list")),
    "Calculates the ROC curve given a set of positive and negative scores and the FAR values for which the CAR should be computed. The resulting ROC curve holds a copy of the given FAR values (row 0), and the corresponding FRR values (row 1)."
  );

  def(
    "ppndf",
    &bob::measure::ppndf,
    (arg("value")),
    "Returns the Deviate Scale equivalent of a false rejection/acceptance ratio.\n\nThe algorithm that calculates the deviate scale is based on function ppndf() from the NIST package DETware version 2.1, freely available on the internet. Please consult it for more details."
  );

  def(
    "det",
    &bob_det,
    (arg("negatives"), arg("positives"), arg("n_points")),
    "Calculates the DET curve given a set of positive and negative scores and a desired number of points. Returns a two-dimensional blitz::Array of doubles that express on its rows:\n\n0. X axis values in the normal deviate scale for the false-rejections\n1. Y axis values in the normal deviate scale for the false-accepts\n\nYou can plot the results using your preferred tool to first create a plot using rows 0 and 1 from the returned value and then replace the X/Y axis annotation using a pre-determined set of tickmarks as recommended by NIST.\n\nThe algorithm that calculates the deviate scale is based on function ppndf() from the NIST package DETware version 2.1, freely available on the internet. Please consult it for more details.\n\nBy 20.04.2011, you could find such package here: http://www.itl.nist.gov/iad/mig/tools/"
  );

  def(
    "epc",
    &bob_epc,
    (arg("dev_negatives"), arg("dev_positives"), arg("test_negatives"), arg("test_positives"), arg("n_points")),
    "Calculates the EPC curve given a set of positive and negative scores and a desired number of points. Returns a two-dimensional blitz::Array of doubles that express the X (cost) and Y (HTER on the test set given the min. HTER threshold on the development set) coordinates in this order. Please note that, in order to calculate the EPC curve, one needs two sets of data comprising a development set and a test set. The minimum weighted error is calculated on the development set and then applied to the test set to evaluate the half-total error rate at that position.\n\n The EPC curve plots the HTER on the test set for various values of 'cost'. For each value of 'cost', a threshold is found that provides the minimum weighted error (see min_weighted_error_rate_threshold()) on the development set. Each threshold is consecutively applied to the test set and the resulting HTER values are plotted in the EPC.\n\n The cost points in which the EPC curve are calculated are distributed uniformily in the range [0.0, 1.0]."
  );

}
