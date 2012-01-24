/**
 * @file python/measure/src/gabor_jet_similarities.cc
 * @date
 * @author
 *
 * @brief Implements python bindings to the bob configuration system
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

#include <boost/python.hpp>
#include <boost/python/tuple.hpp>
#include "measure/GaborJetSimilarities.h"

static boost::python::tuple bob_disparity(const bob::measure::DisparitySimilarity& self){
  return boost::python::make_tuple(self.disparity().first, self.disparity().second);
}

void bind_measure_gabor_jet_similarities(){
  boost::python::class_<bob::measure::ScalarProductSimilarity, boost::shared_ptr<bob::measure::ScalarProductSimilarity> >(
      "ScalarProductSimilarity",
      "This class computes the similarity of two Gabor jets as the normalized scalar product (also known as the cosine measure)"
    )

    .def(
      "__call__",
      &bob::measure::ScalarProductSimilarity::similarity,
      (boost::python::arg("self"), boost::python::arg("jet1"), boost::python::arg("jet2")),
      "Computes the similarity."
    );


  boost::python::class_<bob::measure::CanberraSimilarity, boost::shared_ptr<bob::measure::CanberraSimilarity> >(
      "CanberraSimilarity",
      "This class computes the similarity of two Gabor jets as the Canberra similarity measure: \\sum_j |a_j - a_j'| / (a_j + a_j'))"
    )

    .def(
      "__call__",
      &bob::measure::CanberraSimilarity::similarity,
      (boost::python::arg("self"), boost::python::arg("jet1"), boost::python::arg("jet2")),
      "Computes the similarity."
    );


  boost::python::class_<bob::measure::DisparitySimilarity, boost::shared_ptr<bob::measure::DisparitySimilarity> >(
      "DisparitySimilarity",
      "This class computes the similarity of two Gabor jets by computing the disparity between the two jets and use this to correct phase differences in the calculation of the similarity",
      boost::python::no_init
    )

    .def(
      boost::python::init<const bob::ip::GaborWaveletTransform&>(
        (boost::python::arg("gwt") = bob::ip::GaborWaveletTransform()),
        "Initializes the similarity measure with parameters from the given transform (or default transform, if no other is given)"
      )
    )

    .def(
      "__call__",
      &bob::measure::DisparitySimilarity::similarity,
      (boost::python::arg("self"), boost::python::arg("jet1"), boost::python::arg("jet2")),
      "Computes the similarity."
    )

    .def
    (
      "disparity",
      &bob_disparity,
      boost::python::arg("self"),
      "Returns the disparity estimated by the last call to similarity"
    );

}
