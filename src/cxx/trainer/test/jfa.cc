/**
 * @file cxx/trainer/test/jfa.cc
 * @date Wed Aug 3 12:23:57 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Test the jfa trainer
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

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Trainer-jfa Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include <stdint.h>

#include "core/cast.h"
#include "trainer/JFATrainer.h"
#include "machine/JFAMachine.h"


struct T {
  double eps;
  blitz::Array<double,2> F, N;
  blitz::Array<double,1> m, E, d;
  blitz::Array<double,2> v, u, z, y, x;
  blitz::Array<uint32_t,1> spk_ids;

  T(): eps(1e-4), F(4,6), N(4,2), m(6), E(6), 
       d(6), v(2,6), u(2,6), z(2,6), y(2,2), x(4,2), spk_ids(4)
  { 
    F = 0.3833, 0.6173, 0.5755, 0.5301, 0.2751, 0.2486,
        0.4516, 0.2277, 0.8044, 0.9861, 0.0300, 0.5357,
        0.0871, 0.8021, 0.9891, 0.0669, 0.9394, 0.0182,
        0.6838, 0.7837, 0.5341, 0.8854, 0.8990, 0.6259;
    N = 0.1379, 0.2178, 0.1821, 0.0418, 0.1069, 0.6164, 0.9397, 0.3545;
    m = 0.1806, 0.0451, 0.7232, 0.3474, 0.6606, 0.3839;
    E = 0.6273, 0.0216, 0.9106, 0.8006, 0.7458, 0.8131;
    d = 0.4106, 0.9843, 0.9456, 0.6766, 0.9883, 0.7668;
    v = 0.3367, 0.6624, 0.2442, 0.2955, 0.6802, 0.5278,
        0.4116, 0.6026, 0.7505, 0.5835, 0.5518, 0.5836;
    u = 0.5118, 0.0826, 0.7196, 0.9962, 0.3545, 0.9713,
        0.3464, 0.8865, 0.4547, 0.4134, 0.2177, 0.1257;
    z = 0.3089, 0.7261, 0.7829, 0.6938, 0.0098, 0.8432,
        0.9223, 0.7710, 0.0427, 0.3782, 0.7043, 0.7295;
    y = 0.2243, 0.2691, 0.6730, 0.4775;
    x = 0.9976, 0.1375,
        0.8116, 0.3900,
        0.4857, 0.9274,
        0.8944, 0.9175;
    spk_ids = 0, 0, 1, 1;
  }

  ~T() {}
};

template<typename T, typename U, int d>  
void check_dimensions(const blitz::Array<T,d>& t1, const blitz::Array<U,d>& t2) 
{
  BOOST_REQUIRE_EQUAL(t1.dimensions(), t2.dimensions());
  for( int i=0; i<t1.dimensions(); ++i)
    BOOST_CHECK_EQUAL(t1.extent(i), t2.extent(i));
}

template<typename T, typename U>  
void checkBlitzEqual( const blitz::Array<T,2>& t1, const blitz::Array<U,2>& t2)
{
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    for( int j=0; j<t1.extent(1); ++j)
      BOOST_CHECK_EQUAL(t1(i,j), bob::core::cast<T>(t2(i,j)));
}

template<typename T, typename U>  
void checkBlitzEqual( const blitz::Array<T,3>& t1, const blitz::Array<U,3>& t2) 
{
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    for( int j=0; j<t1.extent(1); ++j)
      for( int k=0; k<t1.extent(2); ++k)
        BOOST_CHECK_EQUAL(t1(i,j,k), bob::core::cast<T>(t2(i,j,k)));
}

template<typename T>  
void checkBlitzClose( const blitz::Array<T,1>& t1, const blitz::Array<T,1>& t2, 
  const double eps )
{
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    BOOST_CHECK_SMALL( fabs( t2(i)-t1(i) ), eps);
}

template<typename T>  
void checkBlitzClose( const blitz::Array<T,2>& t1, const blitz::Array<T,2>& t2, 
  const double eps )
{
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    for( int j=0; j<t1.extent(1); ++j)
      BOOST_CHECK_SMALL( fabs( t2(i,j)-t1(i,j) ), eps);
}



BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( test_estimateXandU )
{
  // estimateXandU
  bob::trainer::jfa::estimateXandU(F,N,m,E,d,v,u,z,y,x,spk_ids);

  // JFA cookbook reference
  blitz::Array<double,2> x_ref(4,2);
  x_ref = 0.2143, 3.1979,
      1.8275, 0.1227,
      -1.3861, 5.3326,
      0.2359,  -0.7914;

  checkBlitzClose(x, x_ref, eps);
}

BOOST_AUTO_TEST_CASE( test_estimateYandV )
{
  // estimateXandU
  bob::trainer::jfa::estimateYandV(F,N,m,E,d,v,u,z,y,x,spk_ids);

  // JFA cookbook reference
  blitz::Array<double,2> y_ref(2,2);
  y_ref = 0.9630, 1.3868,
      0.04255, -0.3721;

  checkBlitzClose(y, y_ref, eps);
}

BOOST_AUTO_TEST_CASE( test_estimateZandD )
{
  // estimateXandU
  bob::trainer::jfa::estimateZandD(F,N,m,E,d,v,u,z,y,x,spk_ids);

  // JFA cookbook reference
  blitz::Array<double,2> z_ref(2,6);
  z_ref = 0.3256, 1.8633, 0.6480, 0.8085, -0.0432, 0.2885,
      -0.3324, -0.1474, -0.4404, -0.4529, 0.0484, -0.5848;

  checkBlitzClose(z, z_ref, eps);
}

BOOST_AUTO_TEST_CASE( test_JFATrainer_updateYandV )
{
  std::vector<blitz::Array<double,2> > Ft;
  blitz::Array<double,2> F1 = F(blitz::Range(0,1),blitz::Range::all()).transpose(1,0);
  blitz::Array<double,2> F2 = F(blitz::Range(2,3),blitz::Range::all()).transpose(1,0);
  Ft.push_back(F1);
  Ft.push_back(F2);

  std::vector<blitz::Array<double,2> > Nt;
  blitz::Array<double,2> N1 = N(blitz::Range(0,1),blitz::Range::all()).transpose(1,0);
  blitz::Array<double,2> N2 = N(blitz::Range(2,3),blitz::Range::all()).transpose(1,0);
  Nt.push_back(N1);
  Nt.push_back(N2);

  blitz::Array<double,2> vt = v.transpose(1,0);
  blitz::Array<double,2> ut = u.transpose(1,0);

  std::vector<blitz::Array<double,1> > zt;
  blitz::Array<double,1> z1 = z(0,blitz::Range::all());
  blitz::Array<double,1> z2 = z(1,blitz::Range::all());
  zt.push_back(z1);
  zt.push_back(z2);

  std::vector<blitz::Array<double,1> > yt;
  blitz::Array<double,1> y1(2);
  blitz::Array<double,1> y2(2);
  y1 = 0;
  y2 = 0;
  yt.push_back(y1);
  yt.push_back(y2);

  std::vector<blitz::Array<double,2> > xt;
  blitz::Array<double,2> x1 = x(blitz::Range(0,1),blitz::Range::all()).transpose(1,0);
  blitz::Array<double,2> x2 = x(blitz::Range(2,3),blitz::Range::all()).transpose(1,0);
  xt.push_back(x1);
  xt.push_back(x2);

  // updateYandV
  boost::shared_ptr<bob::machine::GMMMachine> ubm(new bob::machine::GMMMachine(2,3));
  ubm->setMeanSupervector(m);
  ubm->setVarianceSupervector(E);
  bob::machine::JFABaseMachine jfa_base_m(ubm, 2, 2);
  jfa_base_m.setU(ut);
  jfa_base_m.setV(vt);
  jfa_base_m.setD(d);
  bob::trainer::JFABaseTrainer jfa_base_t(jfa_base_m);
  jfa_base_t.setStatistics(Nt,Ft);
  jfa_base_t.setSpeakerFactors(xt,yt,zt);
  jfa_base_t.precomputeSumStatisticsN();
  jfa_base_t.precomputeSumStatisticsF();

  jfa_base_t.updateY();
  jfa_base_t.updateV();

  // JFA cookbook reference
  // v_ref
  blitz::Array<double,2> v_ref(6,2);
  v_ref = 0.7228, 0.7892,
          0.6475, 0.6080,
          0.8631, 0.8416,
          1.6512, 1.6068,
          0.0500, 0.0101,
          0.4325, 0.6719;
  // y_ref
  blitz::Array<double,1> y1_ref(2);
  y1_ref = 0.9630, 1.3868;
  blitz::Array<double,1> y2_ref(2);
  y2_ref = 0.0426, -0.3721;

  checkBlitzClose(jfa_base_m.getV(), v_ref, eps);
  checkBlitzClose(jfa_base_t.getY()[0], y1_ref, eps);
  checkBlitzClose(jfa_base_t.getY()[1], y2_ref, eps);
}

BOOST_AUTO_TEST_CASE( test_JFATrainer_updateXandU )
{
  std::vector<blitz::Array<double,2> > Ft;
  blitz::Array<double,2> F1 = F(blitz::Range(0,1),blitz::Range::all()).transpose(1,0);
  blitz::Array<double,2> F2 = F(blitz::Range(2,3),blitz::Range::all()).transpose(1,0);
  Ft.push_back(F1);
  Ft.push_back(F2);

  std::vector<blitz::Array<double,2> > Nt;
  blitz::Array<double,2> N1 = N(blitz::Range(0,1),blitz::Range::all()).transpose(1,0);
  blitz::Array<double,2> N2 = N(blitz::Range(2,3),blitz::Range::all()).transpose(1,0);
  Nt.push_back(N1);
  Nt.push_back(N2);

  blitz::Array<double,2> vt = v.transpose(1,0);
  blitz::Array<double,2> ut = u.transpose(1,0);

  std::vector<blitz::Array<double,1> > zt;
  blitz::Array<double,1> z1 = z(0,blitz::Range::all());
  blitz::Array<double,1> z2 = z(1,blitz::Range::all());
  zt.push_back(z1);
  zt.push_back(z2);

  std::vector<blitz::Array<double,1> > yt;
  blitz::Array<double,1> y1 = y(0,blitz::Range::all());
  blitz::Array<double,1> y2 = y(1,blitz::Range::all());
  yt.push_back(y1);
  yt.push_back(y2);

  std::vector<blitz::Array<double,2> > xt;
  blitz::Array<double,2> x1(2,2);
  x1 = 0;
  blitz::Array<double,2> x2(2,2);
  x2 = 0;
  xt.push_back(x1);
  xt.push_back(x2);

  // updateXandU
  boost::shared_ptr<bob::machine::GMMMachine> ubm(new bob::machine::GMMMachine(2,3));
  ubm->setMeanSupervector(m);
  ubm->setVarianceSupervector(E);
  bob::machine::JFABaseMachine jfa_base_m(ubm, 2, 2);
  jfa_base_m.setU(ut);
  jfa_base_m.setV(vt);
  jfa_base_m.setD(d);
  bob::trainer::JFABaseTrainer jfa_base_t(jfa_base_m);
  jfa_base_t.setStatistics(Nt,Ft);
  jfa_base_t.setSpeakerFactors(xt,yt,zt);
  jfa_base_t.precomputeSumStatisticsN();
  jfa_base_t.precomputeSumStatisticsF();

  jfa_base_t.updateX();
  jfa_base_t.updateU();

  // JFA cookbook reference
  // u_ref
  blitz::Array<double,2> u_ref(6,2);
  u_ref = 0.6729, 0.3408,
          0.0544, 1.0653,
          0.5399, 1.3035,
          2.4995, 0.4385,
          0.1292, -0.0576,
          1.1962, 0.0117;
  // x_ref
  blitz::Array<double,2> x1_ref(2,2);
  x1_ref = 0.2143, 1.8275,
           3.1979, 0.1227;
  blitz::Array<double,2> x2_ref(2,2);
  x2_ref = -1.3861, 0.2359,
            5.3326, -0.7914;

  checkBlitzClose(jfa_base_m.getU(), u_ref, eps);
  checkBlitzClose(jfa_base_t.getX()[0], x1_ref, eps);
  checkBlitzClose(jfa_base_t.getX()[1], x2_ref, eps);
}

BOOST_AUTO_TEST_CASE( test_JFATrainer_updateZandD )
{
  std::vector<blitz::Array<double,2> > Ft;
  blitz::Array<double,2> F1 = F(blitz::Range(0,1),blitz::Range::all()).transpose(1,0);
  blitz::Array<double,2> F2 = F(blitz::Range(2,3),blitz::Range::all()).transpose(1,0);
  Ft.push_back(F1);
  Ft.push_back(F2);

  std::vector<blitz::Array<double,2> > Nt;
  blitz::Array<double,2> N1 = N(blitz::Range(0,1),blitz::Range::all()).transpose(1,0);
  blitz::Array<double,2> N2 = N(blitz::Range(2,3),blitz::Range::all()).transpose(1,0);
  Nt.push_back(N1);
  Nt.push_back(N2);

  blitz::Array<double,2> vt = v.transpose(1,0);
  blitz::Array<double,2> ut = u.transpose(1,0);

  std::vector<blitz::Array<double,1> > zt;
  blitz::Array<double,1> z1(6);
  z1 = 0;
  blitz::Array<double,1> z2(6);
  z2 = 0;
  zt.push_back(z1);
  zt.push_back(z2);

  std::vector<blitz::Array<double,1> > yt;
  blitz::Array<double,1> y1 = y(0,blitz::Range::all());
  blitz::Array<double,1> y2 = y(1,blitz::Range::all());
  yt.push_back(y1);
  yt.push_back(y2);

  std::vector<blitz::Array<double,2> > xt;
  blitz::Array<double,2> x1 = x(blitz::Range(0,1),blitz::Range::all()).transpose(1,0);
  blitz::Array<double,2> x2 = x(blitz::Range(2,3),blitz::Range::all()).transpose(1,0);
  xt.push_back(x1);
  xt.push_back(x2);

  // updateZandD
  boost::shared_ptr<bob::machine::GMMMachine> ubm(new bob::machine::GMMMachine(2,3));
  ubm->setMeanSupervector(m);
  ubm->setVarianceSupervector(E);
  bob::machine::JFABaseMachine jfa_base_m(ubm, 2, 2);
  jfa_base_m.setU(ut);
  jfa_base_m.setV(vt);
  jfa_base_m.setD(d);
  bob::trainer::JFABaseTrainer jfa_base_t(jfa_base_m);
  jfa_base_t.setStatistics(Nt,Ft);
  jfa_base_t.setSpeakerFactors(xt,yt,zt);
  jfa_base_t.precomputeSumStatisticsN();
  jfa_base_t.precomputeSumStatisticsF();

  jfa_base_t.updateZ();
  jfa_base_t.updateD();

  // JFA cookbook reference
  // d_ref
  blitz::Array<double,1> d_ref(6);
  d_ref = 0.3110, 1.0138, 0.8297, 1.0382, 0.0095, 0.6320;
  // z_ref
  blitz::Array<double,1> z1_ref(6);
  z1_ref = 0.3256, 1.8633, 0.6480, 0.8085, -0.0432, 0.2885;
  blitz::Array<double,1> z2_ref(6);
  z2_ref = -0.3324, -0.1474, -0.4404, -0.4529, 0.0484, -0.5848;

  checkBlitzClose(jfa_base_m.getD(), d_ref, eps);
  checkBlitzClose(jfa_base_t.getZ()[0], z1_ref, eps);
  checkBlitzClose(jfa_base_t.getZ()[1], z2_ref, eps);
}

BOOST_AUTO_TEST_CASE( test_JFATrainer_train )
{
  std::vector<blitz::Array<double,2> > Ft;
  blitz::Array<double,2> F1 = F(blitz::Range(0,1),blitz::Range::all()).transpose(1,0);
  blitz::Array<double,2> F2 = F(blitz::Range(2,3),blitz::Range::all()).transpose(1,0);
  Ft.push_back(F1);
  Ft.push_back(F2);

  std::vector<blitz::Array<double,2> > Nt;
  blitz::Array<double,2> N1 = N(blitz::Range(0,1),blitz::Range::all()).transpose(1,0);
  blitz::Array<double,2> N2 = N(blitz::Range(2,3),blitz::Range::all()).transpose(1,0);
  Nt.push_back(N1);
  Nt.push_back(N2);

  blitz::Array<double,2> vt = v.transpose(1,0);
  blitz::Array<double,2> ut = u.transpose(1,0);

  std::vector<blitz::Array<double,1> > zt;
  blitz::Array<double,1> z1(6);
  z1 = 0;
  blitz::Array<double,1> z2(6);
  z2 = 0;
  zt.push_back(z1);
  zt.push_back(z2);

  std::vector<blitz::Array<double,1> > yt;
  blitz::Array<double,1> y1 = y(0,blitz::Range::all());
  blitz::Array<double,1> y2 = y(1,blitz::Range::all());
  yt.push_back(y1);
  yt.push_back(y2);

  std::vector<blitz::Array<double,2> > xt;
  blitz::Array<double,2> x1 = x(blitz::Range(0,1),blitz::Range::all()).transpose(1,0);
  blitz::Array<double,2> x2 = x(blitz::Range(2,3),blitz::Range::all()).transpose(1,0);
  xt.push_back(x1);
  xt.push_back(x2);

  // train
  boost::shared_ptr<bob::machine::GMMMachine> ubm(new bob::machine::GMMMachine(2,3));
  ubm->setMeanSupervector(m);
  ubm->setVarianceSupervector(E);
  bob::machine::JFABaseMachine jfa_base_m(ubm, 2, 2);
  jfa_base_m.setU(ut);
  jfa_base_m.setV(vt);
  jfa_base_m.setD(d);
  bob::trainer::JFABaseTrainer jfa_base_t(jfa_base_m);
  jfa_base_t.train(Nt,Ft,1);
}

BOOST_AUTO_TEST_CASE( test_JFATrainer_enrol )
{
  std::vector<blitz::Array<double,2> > Ft;
  blitz::Array<double,2> F1 = F(blitz::Range(0,1),blitz::Range::all()).transpose(1,0);
  blitz::Array<double,2> F2 = F(blitz::Range(2,3),blitz::Range::all()).transpose(1,0);
  Ft.push_back(F1);
  Ft.push_back(F2);

  std::vector<blitz::Array<double,2> > Nt;
  blitz::Array<double,2> N1 = N(blitz::Range(0,1),blitz::Range::all()).transpose(1,0);
  blitz::Array<double,2> N2 = N(blitz::Range(2,3),blitz::Range::all()).transpose(1,0);
  Nt.push_back(N1);
  Nt.push_back(N2);

  blitz::Array<double,2> vt = v.transpose(1,0);
  blitz::Array<double,2> ut = u.transpose(1,0);

  std::vector<blitz::Array<double,1> > zt;
  blitz::Array<double,1> z1 = z(0,blitz::Range::all());
  blitz::Array<double,1> z2 = z(1,blitz::Range::all());
  zt.push_back(z1);
  zt.push_back(z2);

  std::vector<blitz::Array<double,1> > yt;
  blitz::Array<double,1> y1 = y(0,blitz::Range::all());
  blitz::Array<double,1> y2 = y(1,blitz::Range::all());
  yt.push_back(y1);
  yt.push_back(y2);

  std::vector<blitz::Array<double,2> > xt;
  blitz::Array<double,2> x1 = x(blitz::Range(0,1),blitz::Range::all()).transpose(1,0);
  blitz::Array<double,2> x2 = x(blitz::Range(2,3),blitz::Range::all()).transpose(1,0);
  xt.push_back(x1);
  xt.push_back(x2);

  // enrol
  boost::shared_ptr<bob::machine::GMMMachine> ubm(new bob::machine::GMMMachine(2,3));
  ubm->setMeanSupervector(m);
  ubm->setVarianceSupervector(E);
  boost::shared_ptr<bob::machine::JFABaseMachine> jfa_base_m(new bob::machine::JFABaseMachine(ubm, 2, 2));
  jfa_base_m->setU(ut);
  jfa_base_m->setV(vt);
  jfa_base_m->setD(d);
  bob::trainer::JFABaseTrainer jfa_base_t(*jfa_base_m);

  bob::machine::JFAMachine jfa_m(jfa_base_m);

  bob::trainer::JFATrainer jfa_t(jfa_m, jfa_base_t);
  jfa_t.enrol(N1,F1,5);

  double score;
  bob::machine::GMMStats sample(2,3);
  sample.T = 50;
  sample.log_likelihood = -233;
  sample.n = N1(blitz::Range::all(),0);
  for(int g=0; g<2; ++g) {
    blitz::Array<double,1> f = sample.sumPx(g,blitz::Range::all());
    f = F1(blitz::Range(g*3,(g+1)*3-1),0);
  }
  boost::shared_ptr<const bob::machine::GMMStats> sample_(new bob::machine::GMMStats(sample));
//  std::cout << sample.n << sample.sumPx;
  jfa_m.forward(sample_, score);
}

BOOST_AUTO_TEST_SUITE_END()
