/**
 * @file machine/cxx/test/gabor.cc
 * @date 2012-03-05
 * @author Manuel Guenther <Manuel.Guenther@idiap.ch>
 *
 * @brief Tests Gabor graphs and Gabor jet similarity functions.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE machine-GaborGraph Tests
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include <blitz/array.h>

#include "bob/machine/GaborGraphMachine.h"
#include "bob/machine/GaborJetSimilarities.h"

#include "bob/core/logging.h"
#include "bob/io/utils.h"


static const double epsilon = 1e-8;

template <int D>
void test_identical(const blitz::TinyVector<int,D>& shape, const blitz::TinyVector<int,D>& reference){
  for (int i = D; i--;)
    BOOST_CHECK_EQUAL(shape[i], reference[i]);

}

void test_identical(const blitz::Array<int,2>& values, const blitz::Array<int,2>& reference){
  test_identical(values.shape(), reference.shape());
  for (int a = values.shape()[0]; a--;)
    for (int b = values.shape()[1]; b--;)
      BOOST_CHECK_EQUAL(values(a,b), reference(a,b));
}

void test_close(const blitz::Array<double,3>& values, const blitz::Array<double,3>& reference){
  test_identical(values.shape(), reference.shape());
  for (int a = values.shape()[0]; a--;)
    for (int b = values.shape()[1]; b--;)
      for (int c = values.shape()[2]; c--;)
        BOOST_CHECK_SMALL(values(a,b,c) - reference(a,b,c), epsilon);
}


// #define GENERATE_NEW_REFERENCE_FILES
BOOST_AUTO_TEST_CASE( test_gabor_graph_machine )
{
  // Get path to the XML Schema definition
  char* data = getenv("BOB_TESTDATA_DIR");
  if (!data){
    bob::core::error << "Environment variable $BOB_TESTDATA_DIR "
        "is not set. Have you setup your working environment correctly?" << std::endl;
    throw std::runtime_error("test failed");
  }
  std::string data_dir(data);

  // create machine creating a regular grid
  blitz::TinyVector<int,2> first(10,10), last(90,90), step(10,10);
  bob::machine::GaborGraphMachine machine(first, last, step);

  // check node positions
  boost::filesystem::path node_position_file = boost::filesystem::path(data_dir) / "grid_positions.hdf5";
#ifdef GENERATE_NEW_REFERENCE_FILES
  bob::io::save(node_position_file.string(), machine.nodes());
#else // GENERATE_NEW_REFERENCE_FILES
  blitz::Array<int,2> node_positions = bob::io::load<int,2>(node_position_file.string());
  test_identical(machine.nodes(), node_positions);
#endif // GENERATE_NEW_REFERENCE_FILES

  // Load original image
  boost::filesystem::path image_file = boost::filesystem::path(data_dir) / "image.pgm";
  blitz::Array<uint8_t,2> uint8_image = bob::io::load<uint8_t,2>(image_file.string());
  blitz::Array<std::complex<double>,2> image = bob::core::array::cast<std::complex<double> >(uint8_image);

  // perform Gabor wavelet transform
  bob::ip::GaborWaveletTransform gwt;
  blitz::Array<double,4> jet_image(image.shape()[0], image.shape()[1], 2, gwt.numberOfKernels());
  gwt.computeJetImage(image, jet_image, true);

  // extract the graph from the jet image
  blitz::Array<double,3> graph(machine.numberOfNodes(), 2, gwt.numberOfKernels());
  machine.extract(jet_image, graph);

  // check if the jets are still the same
  boost::filesystem::path graph_jets_file = boost::filesystem::path(data_dir) / "graph_jets.hdf5";
#ifdef GENERATE_NEW_REFERENCE_FILES
  bob::io::save(graph_jets_file.string(), graph);
  blitz::Array<double,3> graph_jets = graph;
#else // GENERATE_NEW_REFERENCE_FILES
  blitz::Array<double,3> graph_jets = bob::io::load<double,3>(graph_jets_file.string());
  test_close(graph, graph_jets);
#endif // GENERATE_NEW_REFERENCE_FILES


  // compute similarities of the graph to itself and check that they are unity
  std::vector<boost::shared_ptr<bob::machine::GaborJetSimilarity> > sim_fcts;
  sim_fcts.push_back(boost::shared_ptr<bob::machine::GaborJetSimilarity>(new bob::machine::GaborJetSimilarity(bob::machine::GaborJetSimilarity::SCALAR_PRODUCT)));
  sim_fcts.push_back(boost::shared_ptr<bob::machine::GaborJetSimilarity>(new bob::machine::GaborJetSimilarity(bob::machine::GaborJetSimilarity::CANBERRA)));
  sim_fcts.push_back(boost::shared_ptr<bob::machine::GaborJetSimilarity>(new bob::machine::GaborJetSimilarity(bob::machine::GaborJetSimilarity::DISPARITY, gwt)));
  sim_fcts.push_back(boost::shared_ptr<bob::machine::GaborJetSimilarity>(new bob::machine::GaborJetSimilarity(bob::machine::GaborJetSimilarity::PHASE_DIFF,gwt)));
  sim_fcts.push_back(boost::shared_ptr<bob::machine::GaborJetSimilarity>(new bob::machine::GaborJetSimilarity(bob::machine::GaborJetSimilarity::PHASE_DIFF_PLUS_CANBERRA,gwt)));

  for (int i = sim_fcts.size(); i--;){
    double similarity = machine.similarity(graph, graph_jets, *sim_fcts[i]);
    BOOST_CHECK_CLOSE(similarity, 1., epsilon);
  }
}
