/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Wed 30 Mar 11:34:22 2011 
 *
 * @brief Implements python bindings to the Torch configuration system 
 */

#include "measure/error.h"
#include "core/python/ndarray.h"

using namespace boost::python;
namespace err = Torch::measure;
namespace tp = Torch::python;

/**
 * A nicer python wrapper for the FAR x FRR computation
 */
static tuple farfrr(tp::const_ndarray negatives, tp::const_ndarray positives,
    double threshold) {
  std::pair<double, double> retval = err::farfrr(negatives.bz<double,1>(),
      positives.bz<double,1>(), threshold);
  return make_tuple(retval.first, retval.second);
}

void bind_measure_error() {
 def("farfrr", &farfrr, (arg("negatives"), arg("positives"), arg("threshold")),
     "Calculates the FA ratio and the FR ratio given positive and negative scores and a threshold. 'positives' holds the score information for samples that are labelled to belong to a certain class (a.k.a., 'signal' or 'client'). 'negatives' holds the score information for samples that are labelled *not* to belong to the class (a.k.a., 'noise' or 'impostor').\n\nIt is expected that 'positive' scores are, at least by design, greater than 'negative' scores. So, every positive value that falls bellow the threshold is considered a false-rejection (FR). 'negative' samples that fall above the threshold are considered a false-accept (FA).\n\nPositives that fall on the threshold (exactly) are considered correctly classified. Negatives that fall on the threshold (exactly) are considered *incorrectly* classified. This equivalent to setting the comparision like this pseudo-code:\n\nforeach (positive as K) if K < threshold: falseRejectionCount += 1\nforeach (negative as K) if K >= threshold: falseAcceptCount += 1\n\nThe 'threshold' value does not necessarily have to fall in the range covered by the input scores (negatives and positives altogether), but if it does not, the output will be either (1.0, 0.0) or (0.0, 1.0) depending on the side the threshold falls.\n\nThe output is in form of a std::pair of two double-precision real numbers. The numbers range from 0 to 1. The first element of the pair is the false-accept ratio. The second element of the pair is the false-rejection ratio.\n\nIt is possible that scores are inverted in the negative/positive sense. In some setups the designer may have setup the system so 'positive' samples have a smaller score than the 'negative' ones. In this case, make sure you normalize the scores so positive samples have greater scores before feeding them into this method.");
 def("correctlyClassifiedPositives", &err::correctlyClassifiedPositives, (arg("positives"), arg("threshold")), "This method returns a blitz::Array composed of booleans that pin-point which positives where correctly classified in a 'positive' score sample, given a threshold. It runs the formula: foreach (element k in positive) if positive[k] >= threshold: returnValue[k] = true else: returnValue[k] = false");
 def("correctlyClassifiedNegatives", &err::correctlyClassifiedNegatives, (arg("negatives"), arg("threshold")), "This method returns a blitz::Array composed of booleans that pin-point which negatives where correctly classified in a 'negative' score sample, given a threshold. It runs the formula: foreach (element k in negative) if negative[k] < threshold: returnValue[k] = true else: returnValue[k] = false");
 def("eerThreshold", &err::eerThreshold, (arg("negatives"), arg("positives")), "Calculates the threshold that is, as close as possible, to the equal-error-rate (EER) given the input data. The EER should be the point where the FAR equals the FRR. Graphically, this would be equivalent to the intersection between the R.O.C. (or D.E.T.) curves and the identity.");
 def("minWeightedErrorRateThreshold", &err::minWeightedErrorRateThreshold, (arg("negatives"), arg("positives"), arg("cost")), "Calculates the threshold that minimizes the error rate, given the input data. An optional parameter 'cost' determines the relative importance between false-accepts and false-rejections. This number should be between 0 and 1 and will be clipped to those extremes. The value to minimize becomes: ER_cost = [cost * FAR] + [(1-cost) * FRR]. The higher the cost, the higher the importance given to *not* making mistakes classifying negatives/noise/impostors.");
 def("minHterThreshold", &err::minHterThreshold, (arg("negatives"), arg("positives")), "Calculates the minWeightedErrorRateThreshold() when the cost is 0.5.");
 def("roc", &err::roc, (arg("negatives"), arg("positives"), arg("points")), "Calculates the ROC curve given a set of positive and negative scores and a number of desired points. Returns a two-dimensional blitz::Array of doubles that express the X (FRR) and Y (FAR) coordinates in this order. The points in which the ROC curve are calculated are distributed uniformily in the range [min(negatives, positives), max(negatives, positives)].");
 def("ppndf", &err::ppndf, (arg("value")), "Returns the Deviate Scale equivalent of a false rejection/acceptance ratio.\n\nThe algorithm that calculates the deviate scale is based on function ppndf() from the NIST package DETware version 2.1, freely available on the internet. Please consult it for more details.");
 def("det", &err::det, (arg("negatives"), arg("positives"), arg("points")), "Calculates the DET curve given a set of positive and negative scores and a number of desired points. Returns a two-dimensional blitz::Array of doubles that express on its rows:\n\n0. X axis values in the normal deviate scale for the false-rejections\n1. Y axis values in the normal deviate scale for the false-aceptsn\nYou can plot the results using your preferred tool to first create a plot using rows 0 and 1 from the returned value and then replace the X/Y axis annotation using a pre-determined set of tickmarks as recommended by NIST.\n\nThe algorithm that calculates the deviate scale is based on function ppndf() from the NIST package DETware version 2.1, freely available on the internet. Please consult it for more details.\n\nBy 20.04.2011, you could find such package here: http://www.itl.nist.gov/iad/mig/tools/");
  def("epc", &err::epc, (arg("dev_negatives"), arg("dev_positives"), arg("test_negatives"), arg("test_positives"), arg("points")), "Calculates the EPC curve given a set of positive and negative scores and a number of desired points. Returns a two-dimensional blitz::Array of doubles that express the X (cost) and Y (HTER on the test set given the min. HTER threshold on the development set) coordinates in this order. Please note that, in order to calculate the EPC curve, one needs two sets of data comprising a development set and a test set. The minimum weighted error is calculated on the development set and then applied to the test set to evaluate the half-total error rate at that position.\n\n The EPC curve plots the HTER on the test set for various values of 'cost'. For each value of 'cost', a threshold is found that provides the minimum weighted error (see minWeightedErrorRateThreshold()) on the development set. Each threshold is consecutively applied to the test set and the resulting HTER values are plotted in the EPC.\n\n The cost points in which the EPC curve are calculated are distributed uniformily in the range [0.0, 1.0].");
}
