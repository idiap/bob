/**
 * @file src/core/test/tensor.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Extensive Tensor tests 
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Core-Tensor Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include "core/Tensor.h"

struct T {

  Torch::DoubleTensor dt;
  Torch::FloatTensor ft;
  Torch::LongTensor lt;
  Torch::IntTensor it;
  Torch::ShortTensor st, st2;
  Torch::CharTensor ct;

  T(): dt(3,5), ft(5), lt(3,5), it(5, 5, 5), st(10, 9, 8, 7), st2(2, 2, 2, 2), ct(5) {
    dt.fill(1);
    ft.fill(2);
    lt.fill(3);
    it.fill(4);
    st.fill(5);
    st2.fill(0);
    ct.set(0, 'T');
    ct.set(1, 'O');
    ct.set(2, 'R');
    ct.set(3, 'C');
    ct.set(4, 'H');
  }

  ~T() {}

};

template<typename TTensor, typename TVal> void check_fill_1d(const TTensor& t,
    const TVal& v) {
  BOOST_REQUIRE_EQUAL(t.nDimension(), 1);
  for (int i=0; i<t.size(0); ++i) BOOST_CHECK_EQUAL(t.get(i), v);
}
template<typename TTensor, typename TVal> void check_fill_2d(const TTensor& t,
    const TVal& v) {
  BOOST_REQUIRE_EQUAL(t.nDimension(), 2);
  for (int i=0; i<t.size(0); ++i) 
    for (int j=0; j<t.size(1); ++j) BOOST_CHECK_EQUAL(t.get(i, j), v);
}
template<typename TTensor, typename TVal> void check_fill_3d(const TTensor& t,
    const TVal& v) {
  BOOST_REQUIRE_EQUAL(t.nDimension(), 3);
  for (int i=0; i<t.size(0); ++i) 
    for (int j=0; j<t.size(1); ++j)
      for (int k=0; k<t.size(2); ++k) BOOST_CHECK_EQUAL(t.get(i, j, k), v);
}
template<typename TTensor, typename TVal> void check_fill_4d(const TTensor& t,
    const TVal& v) {
  BOOST_REQUIRE_EQUAL(t.nDimension(), 4);
  for (int i=0; i<t.size(0); ++i) 
    for (int j=0; j<t.size(1); ++j)
      for (int k=0; k<t.size(2); ++k)
        for (int l=0; l<t.size(3); ++l) 
          BOOST_CHECK_EQUAL(t.get(i, j, k, l), v);
}
void check_dimensions(Torch::Tensor& t, unsigned s1, unsigned s2=0, 
                      unsigned s3=0, unsigned s4=0) {
  if (!s2) { //1-D
    BOOST_CHECK_EQUAL(t.nDimension(), 1);
  }
  else if(!s3) { //2-D
    BOOST_CHECK_EQUAL(t.nDimension(), 2);
  }
  else if(!s4) { //3-D
    BOOST_CHECK_EQUAL(t.nDimension(), 3);
  }
  else { //4-D
    BOOST_CHECK_EQUAL(t.nDimension(), 4);
  }
  if (s1) BOOST_CHECK_EQUAL(t.size(0), s1);
  if (s2) BOOST_CHECK_EQUAL(t.size(1), s2);
  if (s3) BOOST_CHECK_EQUAL(t.size(2), s3);
  if (s4) BOOST_CHECK_EQUAL(t.size(3), s4);
}

BOOST_FIXTURE_TEST_SUITE( test_setup, T )

//this will check whether every tensor was initialized as expected
BOOST_AUTO_TEST_CASE( test_init )
{
  BOOST_CHECK_EQUAL(dt.getDatatype(), Torch::Tensor::Double);
  BOOST_CHECK_EQUAL(dt.typeSize(), sizeof(double));
  BOOST_CHECK_EQUAL(dt.sizeAll(), 15);
  check_dimensions(dt, 3, 5);
  check_fill_2d(dt, 1.0);
  BOOST_CHECK_EQUAL(ft.getDatatype(), Torch::Tensor::Float);
  BOOST_CHECK_EQUAL(ft.typeSize(), sizeof(float));
  BOOST_CHECK_EQUAL(ft.sizeAll(), 5);
  check_dimensions(ft, 5);
  check_fill_1d(ft, 2.0);
  BOOST_CHECK_EQUAL(lt.getDatatype(), Torch::Tensor::Long);
  BOOST_CHECK_EQUAL(lt.typeSize(), sizeof(long));
  BOOST_CHECK_EQUAL(lt.sizeAll(), 15);
  check_dimensions(lt, 3, 5);
  check_fill_2d(lt, 3.0);
  BOOST_CHECK_EQUAL(it.getDatatype(), Torch::Tensor::Int);
  BOOST_CHECK_EQUAL(it.typeSize(), sizeof(int));
  BOOST_CHECK_EQUAL(it.sizeAll(), 125);
  check_dimensions(it, 5, 5, 5);
  check_fill_3d(it, 4);
  BOOST_CHECK_EQUAL(st.getDatatype(), Torch::Tensor::Short);
  BOOST_CHECK_EQUAL(st.typeSize(), sizeof(short));
  BOOST_CHECK_EQUAL(st.sizeAll(), 5040);
  check_dimensions(st, 10, 9, 8, 7);
  check_fill_4d(st, 5);
  BOOST_CHECK_EQUAL(ct.getDatatype(), Torch::Tensor::Char);
  BOOST_CHECK_EQUAL(ct.typeSize(), sizeof(char));
  BOOST_CHECK_EQUAL(ct.sizeAll(), 5);
  check_dimensions(ct, 5);
  BOOST_CHECK_EQUAL(ct.get(0), 'T');
  BOOST_CHECK_EQUAL(ct.get(1), 'O');
  BOOST_CHECK_EQUAL(ct.get(2), 'R');
  BOOST_CHECK_EQUAL(ct.get(3), 'C');
  BOOST_CHECK_EQUAL(ct.get(4), 'H');
}

BOOST_AUTO_TEST_CASE( test_resize )
{
  dt.resize(5);
  check_dimensions(dt, 5);
  dt.resize(200, 5, 1, 10);
  check_dimensions(dt, 200, 5, 1, 10);
  dt.resize(1);
  check_dimensions(dt, 1);
}

BOOST_AUTO_TEST_CASE( test_copy )
{
  T other;
  other.dt.resize(1, 2, 3);
  other.dt.fill(100);
  dt.copy(&other.dt);
  check_dimensions(dt, 1, 2, 3);
  check_fill_3d(dt, 100.0);
}

BOOST_AUTO_TEST_CASE( test_set )
{
  dt(0, 0) = 18;
  BOOST_CHECK_EQUAL(dt.get(0, 0), 18.0);
  dt.set(1, 1, 52);
  BOOST_CHECK_EQUAL(dt.get(1, 1), 52.0);

  ft(0) = 3;
  BOOST_CHECK_EQUAL(ft.get(0), 3.0);
  ft.set(4, 'c'); //crazy, but possible!
  BOOST_CHECK_EQUAL(ft.get(4), 'c');

  lt(2, 2) = 14.5;
  BOOST_CHECK_EQUAL(lt.get(2, 2), 14L);
  lt.set(2, 4, 0xffffL);
  BOOST_CHECK_EQUAL(lt.get(2, 4), 0xffffL);
}

BOOST_AUTO_TEST_CASE( test_bracket_stride )
{
  unsigned int counter = 0;
  for (int i=0; i<st2.size(0); ++i)
    for (int j=0; j<st2.size(1); ++j)
      for (int k=0; k<st2.size(2); ++k)
        for (int l=0; l<st2.size(3); ++l)
          st2(i, j, k, l) = counter++;

  int stride_i = st2.stride(0);
  int stride_j = st2.stride(1);
  int stride_k = st2.stride(2);
  int stride_l = st2.stride(3);
  
  for (int i=0; i<st2.size(0); ++i)
    for (int j=0; j<st2.size(1); ++j)
      for (int k=0; k<st2.size(2); ++k)
        for (int l=0; l<st2.size(3); ++l)
          BOOST_CHECK_EQUAL( st2(i, j, k, l), st2(i*stride_i + j*stride_j + k*stride_k + l*stride_l));
}


BOOST_AUTO_TEST_CASE( test_transpose )
{
  unsigned int counter = 0;
  for (int i=0; i<st.size(0); ++i)
    for (int j=0; j<st.size(1); ++j)
      for (int k=0; k<st.size(2); ++k)
        for (int l=0; l<st.size(3); ++l)
          st(i, j, k, l) = counter++;
  Torch::ShortTensor t2;
  t2.transpose(&st, 0, 1);
  BOOST_CHECK(t2.isReference());
  check_dimensions(t2, 9, 10, 8, 7);
  for (int i=0; i<t2.size(1); ++i)
    for (int j=0; j<t2.size(0); ++j)
      for (int k=0; k<t2.size(2); ++k)
        for (int l=0; l<t2.size(3); ++l)
          BOOST_CHECK_EQUAL(t2.get(j, i, k, l), st.get(i, j, k, l));
  Torch::ShortTensor t3;
  t3.transpose(&t2, 0, 1); //back to normal
  BOOST_CHECK(t3.isReference());
  check_dimensions(t3, 10, 9, 8, 7);
  for (int i=0; i<t3.size(0); ++i)
    for (int j=0; j<t3.size(1); ++j)
      for (int k=0; k<t3.size(2); ++k)
        for (int l=0; l<t3.size(3); ++l)
          BOOST_CHECK_EQUAL(t3.get(i, j, k, l), st.get(i, j, k, l));
}

BOOST_AUTO_TEST_CASE( test_select )
{
  unsigned int counter = 0;
  for (int i=0; i<st.size(0); ++i)
    for (int j=0; j<st.size(1); ++j)
      for (int k=0; k<st.size(2); ++k)
        for (int l=0; l<st.size(3); ++l)
          st(i, j, k, l) = counter++;
  Torch::ShortTensor t2;
  t2.select(&st, 0, 2); 
  BOOST_CHECK(t2.isReference());
  check_dimensions(t2, 9, 8, 7);
  for (int j=0; j<t2.size(0); ++j)
    for (int k=0; k<t2.size(1); ++k)
      for (int l=0; l<t2.size(2); ++l)
        BOOST_CHECK_EQUAL(t2.get(j, k, l), st.get(2, j, k, l));
  Torch::ShortTensor t3;
  t3.select(&t2, 0, 4);
  BOOST_CHECK(t3.isReference());
  check_dimensions(t3, 8, 7);
  for (int k=0; k<t3.size(0); ++k)
    for (int l=0; l<t3.size(1); ++l)
      BOOST_CHECK_EQUAL(t3.get(k, l), st.get(2, 4, k, l));
  Torch::ShortTensor t4;
  t4.select(&t3, 0, 6);
  BOOST_CHECK(t4.isReference());
  check_dimensions(t4, 7);
  for (int l=0; l<t4.size(0); ++l)
    BOOST_CHECK_EQUAL(t4.get(l), st.get(2, 4, 6, l));
}

BOOST_AUTO_TEST_CASE( test_narrow )
{
  unsigned int counter = 0;
  for (int i=0; i<st.size(0); ++i)
    for (int j=0; j<st.size(1); ++j)
      for (int k=0; k<st.size(2); ++k)
        for (int l=0; l<st.size(3); ++l)
          st(i, j, k, l) = counter++;
  Torch::ShortTensor t2;
  t2.narrow(&st, 0, 0, 5);
  BOOST_CHECK(t2.isReference());
  check_dimensions(t2, 5, 9, 8, 7);
  for (int i=0; i<t2.size(0); ++i)
    for (int j=0; j<t2.size(1); ++j)
      for (int k=0; k<t2.size(2); ++k)
        for (int l=0; l<t2.size(3); ++l)
          BOOST_CHECK_EQUAL(t2.get(i, j, k, l), st.get(i, j, k, l));
  Torch::ShortTensor t3;
  t3.narrow(&t2, 2, 0, 4);
  BOOST_CHECK(t3.isReference());
  check_dimensions(t3, 5, 9, 4, 7);
  for (int i=0; i<t3.size(0); ++i)
    for (int j=0; j<t3.size(1); ++j)
      for (int k=0; k<t3.size(2); ++k)
        for (int l=0; l<t3.size(3); ++l)
          BOOST_CHECK_EQUAL(t3.get(i, j, k, l), st.get(i, j, k, l));
}

BOOST_AUTO_TEST_SUITE_END()
