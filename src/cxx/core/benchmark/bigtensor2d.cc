/**
 * @file cxx/core/benchmark/bigtensor2d.cc
 * @date Wed Apr 13 16:35:10 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
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
// Standard includes
#include <stdio.h>
#include <iostream>
using namespace std;

// Blitz
#include <blitz/array.h>
#include <blitz/array/reduce.h>

// Timer
#include <boost/timer.hpp>

// Boost program options
#include <boost/program_options.hpp>
namespace po = boost::program_options;

// Boost random
#include <boost/random.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>


/**
 * This reduction gets one template parameter which is the type of input data.
 * For this kind of operation this would normally be T or double and
 * returns a second type which is a tuple of 2 numbers (mean, std dev). This is
 * generalization of the method which could return any number of elements. 
 * 
 * Here are other options for return values:
 * std::pair<>
 * boost::tuple<> #this is a generalized std::pair w/ any number of components
 */
template<typename T> class ReduceMeanStdDev {
  
public:
  
  //You need the following public typedefs and statics as blitz use them
  //internally.
    typedef T T_sourcetype;
  typedef blitz::TinyVector<T,2> T_resulttype;
  typedef T_resulttype T_numtype;
  
  static const bool canProvideInitialValue = true;
  static const bool needIndex = false;
  static const bool needInit = false;
  
  ReduceMeanStdDev() { reset(); }
  
  ReduceMeanStdDev(blitz::TinyVector<T,2> initialValue)
  { m_result = initialValue; }
  
  //accumulates, doesn't tell the index position
  inline bool operator()(T x) const {
    m_result[0] += x;
    m_result[1] += x*x;
    return true;
  }
  
  //accumulates, tells us the index position
  inline bool operator()(T x, int=0) const {
    m_result[0] += x;
    m_result[1] += x*x;
    return true;
  }
  
  //gets the result, tells us how many items we have seen
  inline blitz::TinyVector<T,2> result(int count) const {
    m_result /= count;
    m_result[1] -= m_result[0] * m_result[0];
    // TODO This version doesn't check that if (stdv(j) == 0) stdv(j) = 1 else stdv(j) = sqrt(stdv(j))
    m_result[1] = sqrt(m_result[1]);
    return m_result;
  }
  
  void reset() const { m_result = 0; }
  
  void reset(blitz::TinyVector<T,2> initialValue) { 
    m_result = initialValue; 
  }
  
  static const char* name() { return "meanvar"; }
  
  protected: //representation    
  mutable blitz::TinyVector<T,2> m_result;
};

/**
 * This is the bit that declares the reduction for blitz++
 * Warning: Reductions *must* be declared inside the blitz namespace...
 */
namespace blitz {
  BZ_DECL_ARRAY_FULL_REDUCE(meanvar, ReduceMeanStdDev)
  BZ_DECL_ARRAY_PARTIAL_REDUCE(meanvar, ReduceMeanStdDev)
}

/// Random number generator
static boost::mt19937 rng;
/// Uniform real distribution
boost::uniform_real<> distribution(0.f, 1.f);
/// Generate a float between 0 and 1 (1 included)
boost::variate_generator<boost::mt19937&, boost::uniform_real<> > die(rng, distribution);

/// Blitz function to randomize an array
static float randomize(float)
{
  float d = die();
  return d; 
}

namespace blitz {
    BZ_DECLARE_FUNCTION(randomize)
}

/// Maximum number of tests
#define NB_MAX_TEST 20

/// Help message
const char *help = "\
progname: bigblitz2d.cc\n\
code2html: This program is a bob program to show how to manip a big blitz.\n\
version: bob\n\
date: Feb 2010\n\
author: Sebastien Marcel (marcel@idiap.ch) and Andre Anjos (andre.anjos@idiap.ch) 2010\n";

/// Verbosity level
int verbose;

enum Test_type {
  MEAN,
  MEAN_STDV_COLUMN,
  MEAN_STDV
};

/// Structure that stores the description and duration of each test
struct unit_test
{
  Test_type type;
  double duration;
  char name[500];
  
  bool operator <(const unit_test& test) const {
    return this->duration < test.duration;
  }
};

/// Test results 
static unit_test tests[NB_MAX_TEST];
/// Next available test
static int i_test = 0;

/// Number of times each test is executed
static int times = 1;

static void test_mean_onebyone_row(unit_test& test, float* array, int n_rows, int n_cols, int times);
static void test_mean_onebyone_col( unit_test& test, float* array, int n_rows, int n_cols, int times);
static void test_mean_stdv_by_row( unit_test& test, float* array, int n_rows, int n_cols, int times);

static void test_mean_onebyone_row( unit_test& test, blitz::Array< float, 2 >& array, int times);
static void test_mean_onebyone_col( unit_test& test, blitz::Array< float, 2 >& array, int times);
static void test_mean_complet_reduction( unit_test& test, blitz::Array< float, 2 >& array, int times);
static void test_mean_stdv_by_row( unit_test& test, blitz::Array< float, 2 >& array, int times);
static void test_mean_stdv_slice_col(unit_test& test, blitz::Array<float, 2>& array, int times);
static void test_mean_stdv_slice_row(unit_test& test, blitz::Array<float, 2>& array, int times);
static void test_mean_stdv_partial_reduction(unit_test& test, blitz::Array<float, 2>& array, int times);
static void test_mean_stdv_slices_custom_full_reduction(unit_test& test, blitz::Array<float, 2>& array, int times);
static void test_mean_stdv_custom_partial_reduction(unit_test& test, blitz::Array<float, 2>& array, int times);
static void test_mean_stdv_custom_full_reduction(unit_test& test, blitz::Array<float, 2>& array, int times);

/** 
 * Execute a test and store the result in @c tests
 * 
 * @param array  blitz array
 * @param functionB  test function
 */
static void do_test(blitz::Array<float, 2>& array, void (*functionB)(unit_test&, blitz::Array<float, 2>&, int)) {
  if (i_test < 0 || i_test >= NB_MAX_TEST) {
    std::cerr << "Error: \"tests\" is full. (see NB_MAX_TEST)" << std::endl;
    return ;
  }
  
  functionB(tests[i_test], array, times);
  tests[i_test].duration /= times;
  printf("%s: %.3f ms\n", tests[i_test].name, tests[i_test].duration);
  i_test++;
}

/** 
 * Execute a test and store the result in @c tests
 * 
 * @param array  C array
 * @param n_rows  number of rows
 * @param n_cols  number of columns
 * @param functionC  test function
 */
static void do_test(float* array, int n_rows, int n_cols, void (*functionC)(unit_test&, float*, int, int, int)) {
  if (i_test < 0 || i_test >= NB_MAX_TEST) {
    std::cerr << "Error: \"tests\" is full. (see NB_MAX_TEST)" << std::endl;
    return ;
  }
  
  functionC(tests[i_test], array, n_rows, n_cols, times);
  tests[i_test].duration /= times;
  printf("%s: %.3f ms\n", tests[i_test].name, tests[i_test].duration);
  i_test++;
}

/**
 * Display a result table
 * 
 * @param tests @c unit_test array
 * @param tests_size size of @c tests
 * @param type type of unit_test to display
 */
static void display_results(unit_test* tests, int tests_size, Test_type type) {
  unit_test tests_[NB_MAX_TEST];
  int max_index = 0;
  
  for(int i = 0; i < tests_size; i++) {
    if (tests[i].type == type) {
      tests_[max_index] = tests[i];
      max_index++;
    }
  }
  
  if (max_index == 0) {
    std::cout << "No result for this type" << std::endl;
    return;
  }
  
  std::cout << "     msec | speedup | test" << std::endl;
  sort<unit_test*>(tests_, tests_ + max_index);
  
  double reference  = tests_[0].duration;
  
  for(int i = 0 ; i < max_index ; i++)
  {
    double ratio = reference / tests_[i].duration;
    printf("%9.3f | %7.3f | %s\n", tests_[i].duration, ratio, tests_[i].name);
  }
}


int main(int argc, char **argv)
{
  int64_t seed;
  int n_rows;
  int n_cols;

  // Parse command line
  po::options_description desc(help);
  desc.add_options()
    ("help,h", "produce help message")
    ("seed,s", po::value<int64_t>(&seed)->default_value(-1), "seed for random generator")
    ("rows,r", po::value<int>(&n_rows)->default_value(5000), "number of rows")
    ("cols,c", po::value<int>(&n_cols)->default_value(5000), "number of columns")
    ("times,t", po::value<int>(&times)->default_value(1), "number of times each test is executed")
    ("verbose,v", po::value<int>(&verbose)->default_value(0), "verbose level");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 1;
  }
  
  // Print some information 
  printf("Array size: %d x %d = %d elements\n", n_rows, n_cols, n_rows * n_cols);
  std::cout << "Array memory size: ";
  
  size_t memory_size = n_rows * n_cols * sizeof(float);
  if (memory_size < 1024.f) {
    std::cout << memory_size << " B";
  }
  else if (memory_size < (1024.f * 1024.f)) {
    printf("%.2f kB", (float)(memory_size / 1024.f));
  }
  else if (memory_size < (1024.f * 1024.f * 1024.f)) {
    printf("%.2f MB", (float)((memory_size / 1024.f) / 1024.f));
  }
  else {
    printf("%.2f GB", (float)(((memory_size / 1024.f) / 1024.f) / 1024.f));
  }
 
  std::cout << std::endl;
  
  // Initialize the random number generator
  if(seed < 0) 
  {
    int64_t s = std::time(0);
    rng.seed(s);
    std::cout << "Random seed = " << s << std::endl; 
  }
  else
  {
    std::cout << "Setting manual seed = " << seed << std::endl; 
    rng.seed(seed);
  }

  std::cout << std::endl;
  
  std::cout << "Creating Blitz array" << std::endl;
  blitz::Array<float, 2> blitz_array(n_rows, n_cols);
  
  std::cout << "Randomizing ..." << std::endl;
  blitz_array = randomize(blitz_array);
  
  if(verbose > 2) {
    cout << "array: " << blitz_array << endl;
  }
  
  std::cout << "Creating C array" << std::endl;
  float *array = new float[n_rows * n_cols];
  
  // Copy Blitz array to C array
  for(int i = 0 ; i < n_rows; ++i)
  {
    for (int j = 0; j < n_cols; j++) {
      array[i * n_cols + j] = blitz_array(i, j);
    }
  }

  
  // Do the tests
  do_test(array, n_rows, n_cols, &test_mean_onebyone_row);
  do_test(array, n_rows, n_cols, &test_mean_onebyone_col);
  do_test(array, n_rows, n_cols, &test_mean_stdv_by_row);
  
  do_test(blitz_array, &test_mean_onebyone_row);
  do_test(blitz_array, &test_mean_onebyone_col);
  do_test(blitz_array, &test_mean_complet_reduction);
  do_test(blitz_array, &test_mean_stdv_by_row);
  do_test(blitz_array, &test_mean_stdv_slice_row);
  do_test(blitz_array, &test_mean_stdv_slice_col);
  do_test(blitz_array, &test_mean_stdv_partial_reduction);
  do_test(blitz_array, &test_mean_stdv_slices_custom_full_reduction);
  do_test(blitz_array, &test_mean_stdv_custom_partial_reduction);
  do_test(blitz_array, &test_mean_stdv_custom_full_reduction);
  
  
  std::cout << std::endl;
  std::cout << "Summary tables:" << std::endl;
  std::cout << std::endl;
  
  std::cout << "Mean:" << std::endl;
  display_results(tests, i_test, MEAN);
  
  std::cout << std::endl << "Mean and stdv along each column" << std::endl;
  display_results(tests, i_test, MEAN_STDV_COLUMN);
  
  std::cout << std::endl << "Mean and stdv" << std::endl;
  display_results(tests, i_test, MEAN_STDV);
  std::cout << std::endl;
  return(0);
}

/**
 * Compute the mean by adding elements one-by-one (row)
 * 
 */
static void test_mean_onebyone_row(unit_test& test, float* array, int n_rows, int n_cols, int times) {
  test.type = MEAN;
  
  float mean = 0.0;
  
  boost::timer timer;
  for(;times > 0; --times) {
    int size = n_rows * n_cols;
    for(int i = 0 ; i < size ; i++)
    {
      mean += array[i];
    }
    
    mean /= (float) size;
  }
  test.duration = timer.elapsed() * 1000;
  sprintf(test.name, "C: Computing the mean one-by-one (row)");
  
  if(verbose > 1) {
    cout << "Mean: " << mean << endl;
  }
}

/**
 * Compute the mean by adding elements one-by-one (column)
 * 
 */
static void test_mean_onebyone_col(unit_test& test, float* array, int n_rows, int n_cols, int times) {
  test.type = MEAN;
  
  float mean = 0.0;
  
  boost::timer timer;
  for(;times > 0; --times) {
    int size = n_rows * n_cols;
    for (int j = 0; j < n_cols; j++) {
      float* col = &array[j];
      float* end = col + size;
      for (; col < end; col += n_cols)
      {
        mean += *col;
      }
    }
    
    mean /= (float) size;
  }
  test.duration = timer.elapsed() * 1000;
  sprintf(test.name, "C: Computing the mean one-by-one (column)");
  
  if(verbose > 1) {
    cout << "Mean: " << mean << endl;
  }
}

/**
 * Compute the mean and standard deviation along each column (iterations over columns and rows)
 */
static void test_mean_stdv_by_row(unit_test& test, float* array, int n_rows, int n_cols, int times) {
  test.type = MEAN_STDV_COLUMN;
  
  float* mean_ = new float[n_cols];
  float* stdv_ = new float[n_cols];

  memset(mean_, 0, n_cols * sizeof(float));
  memset(stdv_, 0, n_cols * sizeof(float));

  boost::timer timer;
  for(;times > 0; --times) {
    float *row = array;
    for (int i = 0; i < n_rows; i++) {
      for (int j = 0; j < n_cols; j++) {
        float z = row[j];
        mean_[j] += z;
        stdv_[j] += z*z;
      }
      row += n_cols;
    }

    for (int j = 0; j < n_cols; j++) {
      float z = mean_[j] / (float) n_rows;
      mean_[j] = z;
      stdv_[j] /= (float) n_rows;
      stdv_[j] -= z*z;
      float zz = stdv_[j];
      if(zz <= 0.0) {
        stdv_[j] = 1.0;
      }
      else {
        stdv_[j] = sqrt(zz);
      }
    }
  }
  
  test.duration = timer.elapsed() * 1000;
  sprintf(test.name, "C: Computing mean and stdv along each column");
  
  if(verbose > 1)
  {
    blitz::Array<float, 1> t_mean(mean_, blitz::shape(n_cols), blitz::neverDeleteData);
    blitz::Array<float, 1> t_stdv(stdv_, blitz::shape(n_cols), blitz::neverDeleteData);
    cout << "Mean: " << t_mean(blitz::Range(0, 9)) << endl;
    cout << "Stdv: " << t_stdv(blitz::Range(0, 9)) << endl;
  }


  //
  delete[] mean_;
  delete[] stdv_;
}

/**
 * Compute the mean by adding elements one-by-one (row)
 */
static void test_mean_onebyone_row(unit_test& test, blitz::Array<float, 2>& array, int times) {
  test.type = MEAN;
  
  int n_rows = array.length(0);
  int n_cols = array.length(1);

  float mean = 0.f;
  boost::timer timer;
  for(;times > 0; --times) {
    for (int i = 0; i < n_rows; i++) {
      for (int j = 0; j < n_cols; j++) {
        mean += array(i, j);
      }
    }
    
    mean /= (float) (n_rows * n_cols);
  }
  test.duration = timer.elapsed() * 1000;
  sprintf(test.name, "B: Computing the mean one-by-one (row)");
  
  if(verbose > 1) {
    cout << "Mean: " << mean << endl;
  }
}

/**
 * Compute the mean by adding elements one-by-one (column)
 */
static void test_mean_onebyone_col(unit_test& test, blitz::Array<float, 2>& array, int times) {
  test.type = MEAN;
  
  int n_rows = array.length(0);
  int n_cols = array.length(1);

  float mean = 0.f;
  boost::timer timer;
  for(;times > 0; --times) {
    for (int j = 0; j < n_cols; j++) {
      for (int i = 0; i < n_rows; i++) {
        mean += array(i, j);
      }
    }
    
    mean /= (float) (n_rows * n_cols);
  }
  test.duration = timer.elapsed() * 1000;
  sprintf(test.name, "B: Computing the mean one-by-one (column)");
  
  if(verbose > 1) {
    cout << "Mean: " << mean << endl;
  }
}

/**
 * Compute mean and stdv along each column (iterations over columns and rows)
 */
static void test_mean_stdv_by_row(unit_test& test, blitz::Array<float, 2>& array, int times) {
  test.type = MEAN_STDV_COLUMN;
  
  int n_rows = array.length(0);
  int n_cols = array.length(1);
  
  blitz::Array<float, 1> mean(n_cols);
  blitz::Array<float, 1> stdv(n_cols);
  
  mean = 0;
  stdv = 0;
  
  boost::timer timer;
  for(;times > 0; --times) {
    for (int i = 0; i < n_rows; i++) {
      for (int j = 0; j < n_cols; j++) {
        float z = array(i, j);
        mean(j) += z;
        stdv(j) += z*z;
      }
    }
    
    for (int j = 0; j < n_cols; j++) {
      float z = mean(j) / (float) n_rows;
      mean(j) = z;
      stdv(j) /= (float) n_rows;
      stdv(j) -= z*z;
      float zz = stdv(j);
      if (zz <= 0.0) {
        stdv(j) = 1.0;
      }
      else {
        stdv(j) = sqrt(zz);
      }
    }
  }
  test.duration = timer.elapsed() * 1000;
  sprintf(test.name, "B: Computing mean and stdv along each column (iterations over columns and rows)");
  
  if(verbose > 1)
  {
    std::cout << "Mean: " << mean(blitz::Range(0, 9)) << std::endl;
    std::cout << "Stdv: " << stdv(blitz::Range(0, 9)) << std::endl;
  }
}


/**
 * Compute the mean by complete reduction (blitz::mean function)
 */
static void test_mean_complet_reduction(unit_test& test, blitz::Array<float, 2>& array, int times) {
  test.type = MEAN;
  
  float mean = 0;
  
  boost::timer timer;
  for(;times > 0; --times) {
    mean = blitz::mean(array);
  }
  test.duration = timer.elapsed() * 1000;
  sprintf(test.name, "B: Computing the mean by complete reduction (blitz::mean function)");
  
  if(verbose > 1) {
    cout << "Mean: " << mean << endl;
  }
}

/**
 * Compute mean and stdv along each column using slices (blitz::mean(slice) where slice is one column)
 */
static void test_mean_stdv_slice_col(unit_test& test, blitz::Array<float, 2>& array, int times) {
  test.type = MEAN_STDV_COLUMN;
  
  //int n_rows = array.length(0);
  int n_cols = array.length(1);
  
  blitz::Array<float, 1> mean(n_cols);
  blitz::Array<float, 1> stdv(n_cols);
  
  mean = 0;
  stdv = 0;
  
  boost::timer timer;
  for(;times > 0; --times) {
    for (int j = 0; j < n_cols; j++) {
      blitz::Array<float,1> slice(array(blitz::Range::all(), j));
      float z = blitz::mean(slice);
      mean(j) = z;
      stdv(j) = blitz::mean(blitz::pow2(slice));
      stdv(j) -= z*z;
    }
    
    // TODO This version doesn't check that if (stdv(j) == 0) stdv(j) = 1 else stdv(j) = sqrt(stdv(j))
    stdv = sqrt(stdv);
  }
  
  test.duration = timer.elapsed() * 1000;
  sprintf(test.name, "B: Computing mean and stdv along each column using slices (blitz::mean(slice) where slice is one column)");
  
  if(verbose > 1)
  {
    std::cout << "Mean: " << mean(blitz::Range(0, 9)) << std::endl;
    std::cout << "Stdv: " << stdv(blitz::Range(0, 9)) << std::endl;
  }
}

/**
 * Compute mean and stdv along each column using slices (mean+=slice where slice is one row)
 */
static void test_mean_stdv_slice_row(unit_test& test, blitz::Array<float, 2>& array, int times) {
  test.type = MEAN_STDV_COLUMN;
  
  int n_rows = array.length(0);
  int n_cols = array.length(1);
  
  blitz::Array<float, 1> mean(n_cols);
  blitz::Array<float, 1> stdv(n_cols);
  
  mean = 0;
  stdv = 0;
  
  boost::timer timer;
  for(;times > 0; --times) {
    for(int i = 0; i < n_rows; ++i) {
      blitz::Array<float,1> slice(array(i, blitz::Range::all()));
      mean += slice;
      stdv += blitz::pow2(slice);
    }
    
    mean /= (float) n_rows;
    stdv /= (float) n_rows;
    stdv -= blitz::pow2(mean);
    // TODO This version doesn't check that if (stdv(j) == 0) stdv(j) = 1 else stdv(j) = sqrt(stdv(j))
    stdv = sqrt(stdv);
  }
  test.duration = timer.elapsed() * 1000;
  sprintf(test.name, "B: Computing mean and stdv along each column using slices (mean+=slice where slice is one row)");
    
  if(verbose > 1)
  {
    std::cout << "Mean: " << mean(blitz::Range(0, 9)) << std::endl;
    std::cout << "Stdv: " << stdv(blitz::Range(0, 9)) << std::endl;
  }
}

/**
 * Compute mean and stdv along each column using partial reduction
 */
static void test_mean_stdv_partial_reduction(unit_test& test, blitz::Array<float, 2>& array, int times) {
  test.type = MEAN_STDV_COLUMN;
  
  //int n_rows = array.length(0);
  int n_cols = array.length(1);
  
  blitz::Array<float, 1> mean(n_cols);
  blitz::Array<float, 1> stdv(n_cols);
  
  mean = 0;
  stdv = 0;
  
  boost::timer timer;
  for(;times > 0; --times) {
    blitz::firstIndex i;
    blitz::secondIndex j;
    
    // !!! Warning !!! reduction is only possible over the last dimension
    // The workaround is to interchange the dimensions prior reduction (see http://www.oonumerics.org/blitz/docs/blitz_3.html section 3.14 on Partial Reductions)
    // TODO Is it faster to do
    //    blitz:Array<float,2> tmp = array(j, i)
    //    blitz::mean(tmp, i), j);
    //    sqrt(blitz::mean(blitz::pow2(tmp, i)), j) - mean*mean);
    //  or
    //    blitz::mean(array(j, i), j);
    //    sqrt(blitz::mean(blitz::pow2(array(j, i)), j) - mean*mean);
    mean = blitz::mean(array(j, i), j);
    // TODO This version doesn't check that if (stdv(j) == 0) stdv(j) = 1 else stdv(j) = sqrt(stdv(j))
    stdv = sqrt(blitz::mean(blitz::pow2(array(j, i)), j) - mean*mean);
  }
  test.duration = timer.elapsed() * 1000;
  sprintf(test.name, "B: Computing mean and stdv along each column using partial reduction");
  
  if(verbose > 1)
  {
    std::cout << "Mean: " << mean(blitz::Range(0, 9)) << std::endl;
    std::cout << "Stdv: " << stdv(blitz::Range(0, 9)) << std::endl;
  }
}


/**
 * Compute mean and stdv along each column using column slices and custom full reduction
 */
static void test_mean_stdv_slices_custom_full_reduction(unit_test& test, blitz::Array<float, 2>& array, int times) {
  test.type = MEAN_STDV_COLUMN;
  
  //int n_rows = array.length(0);
  int n_cols = array.length(1);
  
  blitz::Array<float, 1> mean(n_cols);
  blitz::Array<float, 1> stdv(n_cols);
  
  mean = 0;
  stdv = 0;
  boost::timer timer;
  for(;times > 0; --times) {
    for (int j = 0; j < n_cols; j++) {
      blitz::Array<float,1> slice(array(blitz::Range::all(),j));
      blitz::TinyVector<float,2> tmp = blitz::meanvar(slice);
      mean(j) = tmp[0];
      stdv(j) = tmp[1];
    }
  }
  
  test.duration = timer.elapsed() * 1000;
  sprintf(test.name, "B: Computing mean and stdv along each column using column slices and custom full reduction");
  
  if(verbose > 1)
  {
    std::cout << "Mean: " << mean(blitz::Range(0, 9)) << std::endl;
    std::cout << "Stdv: " << stdv(blitz::Range(0, 9)) << std::endl;
  }
}

/**
 * Compute mean and stdv along each column using custom partial reduction 
 */
static void test_mean_stdv_custom_partial_reduction(unit_test& test, blitz::Array<float, 2>& array, int times) {
  test.type = MEAN_STDV_COLUMN;
  
  int n_cols = array.length(1);
  blitz::Array<blitz::TinyVector<float, 2>, 1> muvar(n_cols);
  
  blitz::firstIndex i;
  blitz::secondIndex j;
  
  boost::timer timer;
  for(;times > 0; --times) {
    // !!! Warning !!! reduction is only possible over the last dimension
    // The workaround is to interchange the dimensions prior reduction (see http://www.oonumerics.org/blitz/docs/blitz_3.html section 3.14 on Partial Reductions)
    muvar = blitz::meanvar(array(j, i), j); //partial reduction
  }
  test.duration = timer.elapsed() * 1000;
  sprintf(test.name, "B: Computing mean and stdv along each column using custom partial reduction");
  
  if(verbose > 1)
  {
    std::cout << "Mu: " << muvar.extractComponent(float(),0,2)(blitz::Range(0, 9)) << std::endl;
    std::cout << "Var: " << muvar.extractComponent(float(),1,2)(blitz::Range(0, 9)) << std::endl;
  }
}

/**
 * Compute mean and stdv using custum full reduction
 */
static void test_mean_stdv_custom_full_reduction(unit_test& test, blitz::Array<float, 2>& array, int times) {
  test.type = MEAN_STDV;
  
  int n_cols = array.length(1);
  blitz::Array<blitz::TinyVector<float, 2>, 1> muvar(n_cols);
  
  blitz::firstIndex i;
  blitz::secondIndex j;
  
  boost::timer timer;
  for(;times > 0; --times) {
    muvar = blitz::meanvar(array);
  }
  test.duration = timer.elapsed() * 1000;
  sprintf(test.name, "B: Computing mean and stdv using custom full reduction");
  
  if(verbose > 1)
  {
    std::cout << "Mu: " << muvar.extractComponent(float(),0,2)(blitz::Range(0, 9)) << std::endl;
    std::cout << "Var: " << muvar.extractComponent(float(),1,2)(blitz::Range(0, 9)) << std::endl;
  }
}
