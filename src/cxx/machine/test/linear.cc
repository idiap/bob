/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Mon 20 Jun 2011 17:20:39 CEST
 *
 * @brief Tests linear machine loading/unloading and execution
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Linear Machine Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include <blitz/array.h>
#include <stdint.h>

#include "machine/LinearMachine.h"
#include "machine/Exception.h"
#include "core/logging.h"
#include "database/HDF5File.h"

BOOST_AUTO_TEST_CASE( test_empty_initialization )
{
  Torch::machine::LinearMachine M(2,1);
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
  Torch::machine::LinearMachine M(weights);

  blitz::Array<double,1> biases(2);
  biases = 0.3, -3.0;
  M.setBiases(biases);

  blitz::Array<double,1> isub(3);
  isub = 0, 0.5, 0.5;
  M.setInputSubtraction(isub);

  blitz::Array<double,1> idiv(3);
  idiv = 0.5, 1.0, 1.0;
  M.setInputDivision(idiv);

  M.setActivation(Torch::machine::LinearMachine::TANH);
  
  //now load the same machine from the file and compare
  char *testdata_cpath = getenv("TORCH_MACHINE_TESTDATA_DIR");
  if( !testdata_cpath || !strcmp( testdata_cpath, "") ) {
    Torch::core::error << "Environment variable $TORCH_MACHINE_TESTDATA_DIR " <<
      "is not set. " << "Have you setup your working environment " <<
      "correctly?" << std::endl;
    throw Torch::core::Exception();
  }
  boost::filesystem::path testdata(testdata_cpath);
  testdata /= "linear-test.hdf5";
  Torch::database::HDF5File config(testdata.string(), Torch::database::HDF5File::in);
  Torch::machine::LinearMachine N(config);

  BOOST_CHECK( blitz::all(M.getWeights() == N.getWeights()) );
  BOOST_CHECK( blitz::all(M.getBiases() == N.getBiases()) );
  BOOST_CHECK( blitz::all(M.getInputSubraction() == N.getInputSubraction()) );
  BOOST_CHECK( blitz::all(M.getInputDivision() == N.getInputDivision()) );
  BOOST_CHECK_EQUAL( M.getActivation(), N.getActivation() );
}

BOOST_AUTO_TEST_CASE( test_error_check )
{
  //loads a known machine from the file
  char *testdata_cpath = getenv("TORCH_MACHINE_TESTDATA_DIR");
  if( !testdata_cpath || !strcmp( testdata_cpath, "") ) {
    Torch::core::error << "Environment variable $TORCH_MACHINE_TESTDATA_DIR " <<
      "is not set. " << "Have you setup your working environment " <<
      "correctly?" << std::endl;
    throw Torch::core::Exception();
  }
  boost::filesystem::path testdata(testdata_cpath);
  testdata /= "linear-test.hdf5";
  Torch::database::HDF5File config(testdata.string(), Torch::database::HDF5File::in);
  Torch::machine::LinearMachine M(config);

  blitz::Array<double,2> W(2,3);
  W = 0.4, 0.1, 0.4, 0.2, 0.2, 0.7;

  blitz::Array<double,1> X(5);
  X = 0.3, -3.0, 2.7, -18, 52;

  BOOST_CHECK_THROW(M.setWeights(W), Torch::machine::NInputsMismatch);
  BOOST_CHECK_THROW(M.setBiases(X), Torch::machine::NOutputsMismatch);
  BOOST_CHECK_THROW(M.setInputSubtraction(X), Torch::machine::NInputsMismatch);
  BOOST_CHECK_THROW(M.setInputDivision(X), Torch::machine::NInputsMismatch);
}
