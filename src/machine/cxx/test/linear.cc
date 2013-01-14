/**
 * @file machine/cxx/test/linear.cc
 * @date Mon Jun 20 17:58:16 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Tests linear machine loading/unloading and execution
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
#define BOOST_TEST_MODULE Linear Machine Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include <blitz/array.h>
#include <stdint.h>

#include "bob/machine/LinearMachine.h"
#include "bob/machine/Exception.h"
#include "bob/core/logging.h"
#include "bob/core/array_copy.h"
#include "bob/io/HDF5File.h"
#include "bob/math/linear.h"

/**
 * Evaluates the presumed output of a linear machine through a different path.
 */
static blitz::Array<double,1> presumed (const blitz::Array<double,1>& input) {
  blitz::Array<double,1> buffer(bob::core::array::ccopy(input));
  
  blitz::Array<double,2> weights(3,2);
  weights = 0.4, 0.1, 0.4, 0.2, 0.2, 0.7;
  blitz::Array<double,1> biases(weights.extent(1));
  biases = 0.3, -3.0;
  blitz::Array<double,1> isub(weights.extent(0));
  isub = 0, 0.5, 0.5;
  blitz::Array<double,1> idiv(weights.extent(0));
  idiv = 0.5, 1.0, 1.0;
  
  buffer -= isub;
  buffer /= idiv;

  blitz::firstIndex i;
  blitz::secondIndex j;

  blitz::Array<double,1> output(weights.extent(1));
  bob::math::prod(buffer, weights, output);
  output += biases;
  output = blitz::tanh(output);
  return output;
}

BOOST_AUTO_TEST_CASE( test_empty_initialization )
{
  bob::machine::LinearMachine M(2,1);
  BOOST_CHECK( blitz::all(M.getWeights() == 0.0) );
  BOOST_CHECK_EQUAL( M.getWeights().shape()[0], 2 );
  BOOST_CHECK_EQUAL( M.getWeights().shape()[1], 1 );
  BOOST_CHECK( blitz::all(M.getBiases() == 0.0) );
  BOOST_CHECK_EQUAL( M.getBiases().shape()[0], 1 );
}

BOOST_AUTO_TEST_CASE( test_initialization )
{
  blitz::Array<double,2> weights(3,2);
  weights = 0.4, 0.1, 0.4, 0.2, 0.2, 0.7;
  bob::machine::LinearMachine M(weights);

  blitz::Array<double,1> biases(2);
  biases = 0.3, -3.0;
  M.setBiases(biases);

  blitz::Array<double,1> isub(3);
  isub = 0, 0.5, 0.5;
  M.setInputSubtraction(isub);

  blitz::Array<double,1> idiv(3);
  idiv = 0.5, 1.0, 1.0;
  M.setInputDivision(idiv);

  M.setActivation(bob::machine::TANH);
  
  //now load the same machine from the file and compare
  char *testdata_cpath = getenv("BOB_TESTDATA_DIR");
  if( !testdata_cpath || !strcmp( testdata_cpath, "") ) {
    bob::core::error << "Environment variable $BOB_TESTDATA_DIR " <<
      "is not set. " << "Have you setup your working environment " <<
      "correctly?" << std::endl;
    throw bob::core::Exception();
  }
  boost::filesystem::path testdata(testdata_cpath);
  testdata /= "linear-test.hdf5";
  bob::io::HDF5File config(testdata.string(), bob::io::HDF5File::in);
  bob::machine::LinearMachine N(config);

  BOOST_CHECK( blitz::all(M.getWeights() == N.getWeights()) );
  BOOST_CHECK( blitz::all(M.getBiases() == N.getBiases()) );
  BOOST_CHECK( blitz::all(M.getInputSubtraction() == N.getInputSubtraction()) );
  BOOST_CHECK( blitz::all(M.getInputDivision() == N.getInputDivision()) );
  BOOST_CHECK_EQUAL( M.getActivation(), N.getActivation() );
}

BOOST_AUTO_TEST_CASE( test_error_check )
{
  //loads a known machine from the file
  char *testdata_cpath = getenv("BOB_TESTDATA_DIR");
  if( !testdata_cpath || !strcmp( testdata_cpath, "") ) {
    bob::core::error << "Environment variable $BOB_TESTDATA_DIR " <<
      "is not set. " << "Have you setup your working environment " <<
      "correctly?" << std::endl;
    throw bob::core::Exception();
  }
  boost::filesystem::path testdata(testdata_cpath);
  testdata /= "linear-test.hdf5";
  bob::io::HDF5File config(testdata.string(), bob::io::HDF5File::in);
  bob::machine::LinearMachine M(config);

  blitz::Array<double,2> W(2,3);
  W = 0.4, 0.1, 0.4, 0.2, 0.2, 0.7;

  blitz::Array<double,1> X(5);
  X = 0.3, -3.0, 2.7, -18, 52;

  BOOST_CHECK_THROW(M.setWeights(W), bob::machine::NInputsMismatch);
  BOOST_CHECK_THROW(M.setBiases(X), bob::machine::NOutputsMismatch);
  BOOST_CHECK_THROW(M.setInputSubtraction(X), bob::machine::NInputsMismatch);
  BOOST_CHECK_THROW(M.setInputDivision(X), bob::machine::NInputsMismatch);
}

BOOST_AUTO_TEST_CASE( test_correctness )
{
  //loads a known machine from the file
  char *testdata_cpath = getenv("BOB_TESTDATA_DIR");
  if( !testdata_cpath || !strcmp( testdata_cpath, "") ) {
    bob::core::error << "Environment variable $BOB_TESTDATA_DIR " <<
      "is not set. " << "Have you setup your working environment " <<
      "correctly?" << std::endl;
    throw bob::core::Exception();
  }
  boost::filesystem::path testdata(testdata_cpath);
  testdata /= "linear-test.hdf5";
  bob::io::HDF5File config(testdata.string(), bob::io::HDF5File::in);
  bob::machine::LinearMachine M(config);

  blitz::Array<double,2> in(4,3);
  in = 1, 1, 1, 
       0.5, 0.2, 200,
       -27, 35.77, 0,
       12, 0, 0;

  blitz::Array<double,1> maxerr(2);
  maxerr = 1e-10, 1e-10;

  blitz::Range a = blitz::Range::all();
  for (int i=0; i<in.extent(0); ++i) {
    blitz::Array<double,1> output(M.outputSize());
    M.forward(in(i,a), output);
    BOOST_CHECK(blitz::all(blitz::abs(presumed(in(i,a)) - output) < maxerr));
  }
}
