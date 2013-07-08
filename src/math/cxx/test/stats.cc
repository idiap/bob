/**
 * @file math/cxx/test/stats.cc
 * @date Mon Jul  8 12:08:50 CEST 2013
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Test the computation of the scatter matrices
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
#define BOOST_TEST_MODULE math-stats Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include <bob/math/stats.h>
#include <vector>

struct T {
  double eps;

  T(): eps(1e-6)
  {
  }

  ~T() {}
};

template<typename T, typename U, int d>  
void check_dimensions( blitz::Array<T,d>& t1, blitz::Array<U,d>& t2) 
{
  BOOST_REQUIRE_EQUAL(t1.dimensions(), t2.dimensions());
  for( int i=0; i<t1.dimensions(); ++i)
    BOOST_CHECK_EQUAL(t1.extent(i), t2.extent(i));
}


template<typename T>  
void checkBlitzClose( blitz::Array<T,2>& t1, blitz::Array<T,2>& t2, 
  const double eps )
{
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    for( int j=0; j<t1.extent(1); ++j)
      BOOST_CHECK_SMALL( fabs( t2(i,j)-t1(i,j) ), eps);
}

/**
 * @brief Evaluates, for testing purposes, the St scatter which is the total
 * scatter for the LDA problem. The total scatter St relates to the within
 * class scatter Sw and the between class scatter Sb in the following manner:
 * St = Sw + Sb (Bishop's Equation 4.45).
 */
static void evalTotalScatter(const std::vector<blitz::Array<double,2> >& data,
    blitz::Array<double,2>& St, blitz::Array<double,1>& m)
{
  int n_features = data[0].extent(1);
  blitz::Array<double,1> buffer(n_features);

  blitz::firstIndex i;
  blitz::secondIndex j;

  // within class scatter Sw
  St = 0;
  blitz::Range a = blitz::Range::all();
  for (size_t k=0; k<data.size(); ++k) { //class loop
    for (int example=0; example<data[k].extent(0); ++example) {
      buffer = data[k](example,a) - m;
      St += buffer(i) * buffer(j); //outer product
    }
  }
} 

BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( test_stats )
{
  for (int loop=0; loop < 10; ++loop) {
    // size of the data
    int M = (rand() % 64 + 1); 
    int N = (rand() % 64 + 1); 

    // set up simple 1D random array
    blitz::Array<double,2> t1(M,N);
    blitz::Array<double,2> t2(M,N);
    for (int i=0; i < M; ++i)
      for (int j=0; j < N; ++j) {
        t1(i,j) = (rand()/(double)RAND_MAX)*10.;
        t2(i,j) = (rand()/(double)RAND_MAX)*10.;
      }

    std::vector<blitz::Array<double,2> > data;
    data.push_back(t1);
    data.push_back(t2);

    // Evaluate scatters and mean    
    blitz::Array<double,1> mean(N);
    blitz::Array<double,2> Sw(N,N);
    blitz::Array<double,2> Sb(N,N);
    bob::math::scatters(data, Sw, Sb, mean);

    // Evaluate total scatter
    blitz::Array<double,2> St(N,N);
    evalTotalScatter(data, St, mean);

    // Check that St == Sw + Sb
    St -= (Sw + Sb);
    blitz::Array<double,2> z(N,N);
    z = 0.;
    checkBlitzClose(St, z, eps);
  }
}

BOOST_AUTO_TEST_SUITE_END()

