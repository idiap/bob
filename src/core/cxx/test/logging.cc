/**
 * @file core/cxx/test/logging.cc
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * @brief Multiple C++ tests for the logging infrastructure.
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

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Logging Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include <boost/shared_array.hpp>

#include "bob/core/logging.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>

using namespace bob::core;

/**
 * Generates a unique temporary filename
 */
std::string temp_file() {
  std::string tpl = bob::core::tmpdir();
  tpl += "/bobtest_core_loggingXXXXXX";
  boost::shared_array<char> char_tpl(new char[tpl.size()+1]);
  strcpy(char_tpl.get(), tpl.c_str());
  int fd = mkstemp(char_tpl.get());
  close(fd);
  boost::filesystem::remove(char_tpl.get());
  return char_tpl.get();
}

//tests if the streams are correctly initialized and will work correctly
BOOST_AUTO_TEST_CASE( test_basic )
{
  TDEBUG1("NOT SUPPOSED TO BE PRINTED!");
  setenv("BOB_DEBUG", "3", 1); ///< after this, all messages should be printed
  TDEBUG1("This is a debug message, level 1. " << "I can also stream!");
  TDEBUG2("This is a debug message, level 2. ");
  TDEBUG3("This is a debug message, level 3. ");
  info << "This is an info message." << std::endl;
  warn << "This is a warning message." << std::endl;
  error << "This is an error message." << std::endl;
  unsetenv("BOB_DEBUG");
}

/**
 * Returns the contents of a file in a single string. Useful for testing the
 * output of logging.
 */
std::string get_contents(const std::string& fname) {
  std::string cmd;
  if (boost::filesystem::path(fname).extension() == ".gz") cmd = "zcat ";
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

//tests if I can read from files
BOOST_AUTO_TEST_CASE( test_input )
{
  std::string testfilename = temp_file();
  bob::core::OutputStream ofile(testfilename);
  std::string testdata = "12345678,a_single_sentence";
  ofile << testdata << std::endl;
  ofile.close();
  bob::core::InputStream ifile(testfilename);
  std::string back;
  ifile >> back;
  BOOST_CHECK_EQUAL(testdata, back);
  //error << "File saved at: " << testfilename << std::endl;
  boost::filesystem::remove(testfilename);
}

//tests if I can read from compressed files
BOOST_AUTO_TEST_CASE( test_compressed_input )
{
  std::string testfilename = temp_file() + ".gz";
  bob::core::OutputStream ofile(testfilename);
  std::string testdata = "12345678,a_single_sentence";
  ofile << testdata << std::endl;
  ofile.close();
  bob::core::InputStream ifile(testfilename);
  std::string back;
  ifile >> back;
  BOOST_CHECK_EQUAL(testdata, back);
  //error << "File saved at: " << testfilename << std::endl;
  boost::filesystem::remove(testfilename);
}
