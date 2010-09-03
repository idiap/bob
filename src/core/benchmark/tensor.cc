/**
 * @file tensor.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Extensive Tensor benchmark tests 
 */

#include <iostream>
#include <stdint.h>
#include <boost/timer.hpp>
#include <boost/format.hpp>
#include "core/Tensor.h"

const char* type2string(const Torch::Tensor& t) {
  static const char* d = "Double";
  static const char* f = "Float";
  static const char* l = "Long";
  static const char* i = "Int";
  static const char* s = "Short";
  static const char* c = "Char";
  static const char* u = "Unknown";
  switch (t.getDatatype()) {
    case Torch::Tensor::Double:
      return d;
      break;
    case Torch::Tensor::Float:
      return f;
      break;
    case Torch::Tensor::Long:
      return l;
      break;
    case Torch::Tensor::Int:
      return i;
      break;
    case Torch::Tensor::Short:
      return s;
      break;
    case Torch::Tensor::Char:
      return c;
      break;
    default:
      break;
  }
  return u;
}

template<typename T> void benchmark_allocation
(const unsigned times, const unsigned s) {
  boost::timer timer;
  for (unsigned n=0; n<times; ++n) {
    Torch::Tensor* t = new T(s, s, s, s);
    delete t;
  }
  double elapsed = timer.elapsed();
  T tmp;
  float tsize = s*s*s*s*tmp.typeSize();
  boost::format value("%.2f b");
  if (tsize > 1024) {
    tsize /= 1024;
    value = boost::format("%.2f kb");
  }
  if (tsize > 1024) {
    tsize /= 1024;
    value = boost::format("%.2f Mb");
  }
  if (tsize > 1024) {
    tsize /= 1024;
    value = boost::format("%.2f Gb");
  }

  std::cout << "total: " << elapsed << " s; tensor size: "
            << value % tsize << "; per alloc.: " << (1000000*elapsed)/times
            << " us";
}

void benchmark_set(const unsigned int times, Torch::Tensor& t) {
  short v = 0;
  const unsigned S[] = {t.size(0), t.size(1), t.size(2), t.size(3)};
  boost::timer timer;
  for (unsigned n=0; n<times; ++n) 
    for(unsigned int i=0; i<S[0]; ++i)  
      for(unsigned int j=0; j<S[1]; ++j)  
        for(unsigned int k=0; k<S[2]; ++k)  
          for(unsigned int l=0; l<S[3]; ++l) {
            t.set(i, j, k, l, v++);
          }
  double elapsed = timer.elapsed();
  uint64_t total = times * t.sizeAll();
  std::cout << "total: " << elapsed << " s; per element: "
            << ((1000000.0)*(elapsed/total)) << " us";
}

template<typename T> double benchmark_get(const unsigned int times, const T& t) {
  double v = 0;
  const unsigned S[] = {t.size(0), t.size(1), t.size(2), t.size(3)};
  boost::timer timer;
  switch(t.nDimension())
  {
    case 1:
      for (unsigned n=0; n<times; ++n) 
        for(unsigned int i=0; i<S[0]; ++i)  
          v += t.get(i);
      break;
    case 2:
      for (unsigned n=0; n<times; ++n) 
        for(unsigned int i=0; i<S[0]; ++i)  
          for(unsigned int j=0; j<S[1]; ++j)  
            v += t.get(i, j);
      break;
    case 3:
      for (unsigned n=0; n<times; ++n) 
        for(unsigned int i=0; i<S[0]; ++i)  
          for(unsigned int j=0; j<S[1]; ++j)  
            for(unsigned int k=0; k<S[2]; ++k)  
              v += t.get(i, j, k);
      break;
    case 4:
      for (unsigned n=0; n<times; ++n) 
        for(unsigned int i=0; i<S[0]; ++i)  
          for(unsigned int j=0; j<S[1]; ++j)  
            for(unsigned int k=0; k<S[2]; ++k)  
              for(unsigned int l=0; l<S[3]; ++l) 
                v += t.get(i, j, k, l);
      break;
    default:
      break;
  }
  double elapsed = timer.elapsed();
  uint64_t total = times * t.sizeAll();
  std::cout << "total: " << elapsed << " s; per element: "
            << ((1000000.0)*(elapsed/total)) << " us";
  return v;
}

template<typename T> void benchmark_narrow(const unsigned int times, const T& t) {
  const unsigned S[] = {t.size(0), t.size(1), t.size(2), t.size(3)};
  T narrowed;
  boost::timer timer;
  for (unsigned n=0; n<times; ++n) narrowed.narrow(&t, 0, 0, S[0]/2);
  double elapsed = timer.elapsed();
  std::cout << "total: " << elapsed << " s; per operation: "
            << ((1000000.0)*(elapsed/times)) << " us";
}

template<typename T> void 
benchmark_add(const unsigned int times, const T& t1, const T& t2) {
  const unsigned S[] = {t1.size(0), t1.size(1), t1.size(2), t1.size(3)};
  T result1(S[0]);
  T result2(S[0], S[1]);
  T result3(S[0], S[1], S[2]);
  T result4(S[0], S[1], S[2], S[3]);
  boost::timer timer;
  switch(t1.nDimension())
  {
    case 1:
      for (unsigned n=0; n<times; ++n) 
        for(unsigned int i=0; i<S[0]; ++i)  
          result1(i) = t1.get(i) + t2.get(i);
      break;
    case 2:
      for (unsigned n=0; n<times; ++n) 
        for(unsigned int i=0; i<S[0]; ++i)  
          for(unsigned int j=0; j<S[1]; ++j)  
            result2(i, j) = t1.get(i, j) + t2.get(i, j);
      break;
    case 3:
      for (unsigned n=0; n<times; ++n) 
        for(unsigned int i=0; i<S[0]; ++i)  
          for(unsigned int j=0; j<S[1]; ++j)  
            for(unsigned int k=0; k<S[2]; ++k)  
              result3(i, j, k) = t1.get(i, j, k) + t2.get(i, j, k);
      break;
    case 4:
      for (unsigned n=0; n<times; ++n) 
        for(unsigned int i=0; i<S[0]; ++i)  
          for(unsigned int j=0; j<S[1]; ++j)  
            for(unsigned int k=0; k<S[2]; ++k)  
              for(unsigned int l=0; l<S[3]; ++l)
                result4(i, j, k, l) = t1.get(i, j, k, l) + t2.get(i, j, k, l);
      break;
    default:
      break;
  }
  double elapsed = timer.elapsed();
  std::cout << "total: " << elapsed << " s; per time: "
            << ((1000.0)*(elapsed/times)) << " ms";
}

int main(int argc, char** argv) {
  boost::format H("%19s | ");
  
  //Allocation tests
  const unsigned TS[] = {1, 10, 100};
  const unsigned AR[] = {2500000, 2500000, 200000};
  for (unsigned k=0; k<3; ++k) {
    std::cout << H % "Alloc (Double/4d)";
    benchmark_allocation<Torch::DoubleTensor>(AR[k], TS[k]);
    std::cout << std::endl;
    std::cout << H % "Alloc (Float/4d)";
    benchmark_allocation<Torch::FloatTensor>(AR[k], TS[k]);
    std::cout << std::endl;
    std::cout << H % "Alloc (Long/4d)";
    benchmark_allocation<Torch::LongTensor>(AR[k], TS[k]);
    std::cout << std::endl;
    std::cout << H % "Alloc (Int/4d)";
    benchmark_allocation<Torch::IntTensor>(AR[k], TS[k]);
    std::cout << std::endl;
    std::cout << H % "Alloc (Short/4d)";
    benchmark_allocation<Torch::ShortTensor>(AR[k], TS[k]);
    std::cout << std::endl;
    std::cout << H % "Alloc (Char/4d)";
    benchmark_allocation<Torch::CharTensor>(AR[k], TS[k]);
    std::cout << std::endl;
  }

  //Set tests //would narrowing be slower?
  const unsigned N = 20;
  const unsigned S[] = {100, 100, 10, 10};
  std::cout << H % "Set (4d)";
  Torch::FloatTensor dt4(S[0], S[1], S[2], S[3]);
  benchmark_set(N, dt4);
  std::cout << std::endl;
  std::cout << H % "Set (4d/narrowed)";
  Torch::FloatTensor dt4_narrow;
  dt4_narrow.narrow(&dt4, 3, 0, 5);
  benchmark_set(2*N, dt4_narrow);
  std::cout << std::endl;
  std::cout << H % "Set (3d)";
  Torch::FloatTensor dt3(S[0], S[1], S[2]);
  benchmark_set(250*N, dt3);
  std::cout << std::endl;
  std::cout << H % "Set (3d/narrowed)";
  Torch::FloatTensor dt3_narrow;
  dt3_narrow.narrow(&dt3, 2, 0, 5);
  benchmark_set(400*N, dt3_narrow);
  std::cout << std::endl;
  std::cout << H % "Set (2d)";
  Torch::FloatTensor dt2(S[0], S[1]);
  benchmark_set(2500*N, dt2);
  std::cout << std::endl;
  std::cout << H % "Set (2d/narrowed)";
  Torch::FloatTensor dt2_narrow;
  dt2_narrow.narrow(&dt2, 1, 0, 50);
  benchmark_set(4000*N, dt2_narrow);
  std::cout << std::endl;
  std::cout << H % "Set (1d)";
  Torch::FloatTensor dt1(S[0]);
  benchmark_set(250000*N, dt1);
  std::cout << std::endl;
  std::cout << H % "Set (1d/narrowed)";
  Torch::FloatTensor dt1_narrow;
  dt1_narrow.narrow(&dt1, 0, 0, 50);
  benchmark_set(250000*N, dt1_narrow);
  std::cout << std::endl;

  //Retrieval tests
  std::cout << H % "Get (4d)";
  benchmark_get(N, dt4);
  std::cout << std::endl;
  std::cout << H % "Get (4d/narrowed)";
  benchmark_get(2*N, dt4_narrow);
  std::cout << std::endl;
  std::cout << H % "Get (3d)";
  benchmark_get(250*N, dt3);
  std::cout << std::endl;
  std::cout << H % "Get (3d/narrowed)";
  benchmark_get(400*N, dt3_narrow);
  std::cout << std::endl;
  std::cout << H % "Get (2d)";
  benchmark_get(2500*N, dt2);
  std::cout << std::endl;
  std::cout << H % "Get (2d/narrowed)";
  benchmark_get(4000*N, dt2_narrow);
  std::cout << std::endl;
  std::cout << H % "Get (1d)";
  benchmark_get(250000*N, dt1);
  std::cout << std::endl;
  std::cout << H % "Get (1d/narrowed)";
  benchmark_get(250000*N, dt1_narrow);
  std::cout << std::endl;

  //Narrow tests
  std::cout << H % "Narrow (4d)";
  benchmark_narrow(10000000, dt4);
  std::cout << std::endl;
  std::cout << H % "Narrow (3d)";
  benchmark_narrow(10000000, dt3);
  std::cout << std::endl;
  std::cout << H % "Narrow (2d)";
  benchmark_narrow(10000000, dt2);
  std::cout << std::endl;
  std::cout << H % "Narrow (1d)";
  benchmark_narrow(10000000, dt1);
  std::cout << std::endl;

  //Operation tests: add
  boost::format fsize4("Add (%d,%d,%d,%d)");
  std::string size = str(fsize4 % dt4.size(0) % dt4.size(1) % dt4.size(2) 
      % dt4.size(3));
  std::cout << H % size;
  benchmark_add(10, dt4, dt4);
  std::cout << std::endl;
  boost::format fsize3("Add (%d,%d,%d)");
  size = str(fsize3 % dt3.size(0) % dt3.size(1) % dt3.size(2));
  std::cout << H % size;
  benchmark_add(5000, dt3, dt3);
  std::cout << std::endl;
  boost::format fsize2("Add (%d,%d)");
  size = str(fsize2 % dt2.size(0) % dt2.size(1));
  std::cout << H % size;
  benchmark_add(50000, dt2, dt2);
  std::cout << std::endl;
  boost::format fsize1("Add (%d)");
  size = str(fsize1 % dt1.size(0));
  std::cout << H % size;
  benchmark_add(2500000, dt1, dt1);
  std::cout << std::endl;
}
