/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Implements the error evaluation routines 
 */

#include "error/eval.h"

namespace err = Torch::error;

std::pair<double, double> err::farfrr(const blitz::Array<double,1>& negatives, 
    const blitz::Array<double,1>& positives, double threshold) {
}

double err::minimizingThreshold(const blitz::Array<double,1>& negatives,
    const blitz::Array<double,1>& positives,
    double (*predicate)(double, double)) {
}

double err::eerThreshold(const blitz::Array<double,1>& negatives,
    const blitz::Array<double,1>& positives) {
}

double err::minHterThreshold(const blitz::Array<double,1>& negatives,
    const blitz::Array<double,1>& positives, double cost) {
}

blitz::Array<double,2> err::roc(const blitz::Array<double,1>& negatives,
 const blitz::Array<double,1>& positives, size_t points) {
}

blitz::Array<double,2> err::det(const blitz::Array<double,1>& negatives,
 const blitz::Array<double,1>& positives, size_t points) {
}

blitz::Array<double,2> err::epc(const blitz::Array<double,1>& negatives,
 const blitz::Array<double,1>& positives, size_t points) {
}
