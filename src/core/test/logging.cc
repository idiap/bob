/**
 * @file test/logging.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Multiple C++ tests for the logging infrastructure.
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Logging Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>

#include "core/logging.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>

using namespace Torch::core;

//tests if the streams are correctly initialized and will work correctly
BOOST_AUTO_TEST_CASE( test_basic )
{
  TDEBUG1("NOT SUPPOSED TO BE PRINTED!");
  setenv("TORCH_DEBUG", "3", 1); ///< after this, all messages should be printed
  TDEBUG1("This is a debug message, level 1. " << "I can also stream!");
  TDEBUG2("This is a debug message, level 2. ");
  TDEBUG3("This is a debug message, level 3. ");
  info << "This is an info message." << std::endl;
  warn << "This is a warning message." << std::endl;
  error << "This is an error message." << std::endl;
  unsetenv("TORCH_DEBUG");
}

/**
 * Returns the contents of a file in a single string. Useful for testing the
 * output of logging.
 */
std::string get_contents(const std::string& fname) {
  std::string cmd;
  if (boost::filesystem::extension(fname) == ".gz") cmd = "zcat ";
  else cmd = "cat ";
  cmd += fname;
  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe) return "ERROR";
  char buffer[128];
  std::string result = "";
  while(!feof(pipe)) {
      if(fgets(buffer, 128, pipe) != NULL) result += buffer;
  }
  pclose(pipe);
  return result;
}

/**
 * Generates a unique temporary filename
 */
std::string temp_file() {
  char tmp_name[12] = "test_XXXXXX";
  mkstemp(tmp_name);
  return tmp_name;
}

//tests if I can easily switch streams 
BOOST_AUTO_TEST_CASE( test_switch )
{
  std::string testfile = temp_file();
  std::string gztestfile = testfile + ".gz";
  std::string teststring = "** info test **";

  info.reset(testfile);
  info << teststring << std::endl;
  info.reset(gztestfile);

  //at this point checks if "testfile" is filled
  BOOST_CHECK(boost::filesystem::exists(testfile));
  BOOST_CHECK_EQUAL(get_contents(testfile), teststring + "\n");
  boost::filesystem::remove(testfile);

  info << teststring << std::endl;
  info.reset("null");
  
  //at this point checks if "testfile.gz" is filled
  BOOST_CHECK(boost::filesystem::exists(gztestfile));
  BOOST_CHECK_EQUAL(get_contents(gztestfile), teststring + "\n");
  boost::filesystem::remove(gztestfile);

  info << "NOT SUPPOSED TO BE PRINTED!" << std::endl;
}
