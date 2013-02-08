/**
 * @file ip/cxx/GLCM.cc
 * @date Wed Jan 30 17:25:28 CET 2013
 * @author Ivana Chingovska <ivana.chingovska@idiap.ch> 
 *
 * @brief GLCMProp implementation
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

#include "bob/ip/GLCMProp.h"
#include "bob/core/array_copy.h"
#include "bob/core/array_assert.h"
#include <boost/make_shared.hpp>

static double sqr(const double x)
{
  return x*x;
}


bob::ip::GLCMProp::GLCMProp(){ }

bob::ip::GLCMProp::~GLCMProp() { }

const blitz::Array<double,3> bob::ip::GLCMProp::normalize_glcm(const blitz::Array<double,3>& glcm) const
{ 
   blitz::firstIndex i;
   blitz::secondIndex j;
   blitz::thirdIndex k;
   blitz::Array<double, 2> summations_temp(blitz::sum(glcm(i, k, j), k));
   blitz::Array<double, 1> summations(blitz::sum(summations_temp(j,i), j));
   blitz::Array<double,3> res(glcm / summations(k));
   return res;
}

const blitz::TinyVector<int,1> bob::ip::GLCMProp::get_prop_shape(const blitz::Array<double,3>& glcm) const
{
  blitz::TinyVector<int,1> res;
  res(0) = glcm.extent(2);
  return res;
}



void bob::ip::GLCMProp::angular_second_moment(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const
{
  // check if the size of the output matrix is as expected
  blitz::TinyVector<int,1> shape(get_prop_shape(glcm));
  bob::core::array::assertSameShape(prop, shape); 
  
  // normalize the input GLCM matrix
  blitz::Array<double,3> glcm_norm = normalize_glcm(glcm);
  
  blitz::Array<double,2> mat(glcm.extent(0), glcm.extent(1)); // auxiliary matrix that will be used for glcm matrix for one particular offset
  
  blitz::Range rall = blitz::Range::all();
  
  //do the computation of the feature
  for (int l=0; l < glcm_norm.extent(2); ++l)
  {
    mat = blitz::pow2(glcm_norm(rall, rall, l));
    prop(l) = blitz::sum(mat); // angular second moment
  } 
}

void bob::ip::GLCMProp::energy(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const 
{ 
  //do the computation of the feature
  angular_second_moment(glcm, prop);
  prop = blitz::sqrt(prop);
}

void bob::ip::GLCMProp::variance(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const
{
  // check if the size of the output matrix is as expected
  blitz::TinyVector<int,1> shape(get_prop_shape(glcm));
  bob::core::array::assertSameShape(prop, shape); 
  
  // normalize the input GLCM matrix
  blitz::Array<double,3> glcm_norm = normalize_glcm(glcm);
  
  blitz::Array<double,2> mat(glcm.extent(0), glcm.extent(1)); // auxiliary matrix that will be used for glcm matrix for one particular offset
  
  blitz::Range rall = blitz::Range::all();
  blitz::firstIndex i;
  blitz::secondIndex j;
  
  //do the computation of the feature
  for (int l=0; l < glcm_norm.extent(2); ++l)
  {
    mat = glcm_norm(rall, rall, l);
    prop(l) = blitz::sum(sqr(i-blitz::mean(mat))*mat);
  }  
}


void bob::ip::GLCMProp::contrast(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const
{
  // check if the size of the output matrix is as expected
  blitz::TinyVector<int,1> shape(get_prop_shape(glcm));
  bob::core::array::assertSameShape(prop, shape); 
  
  // normalize the input GLCM matrix
  blitz::Array<double,3> glcm_norm = normalize_glcm(glcm);
  
  blitz::Array<double,2> mat(glcm.extent(0), glcm.extent(1)); // auxiliary matrix that will be used for glcm matrix for one particular offset
  
  blitz::Range rall = blitz::Range::all();
  blitz::firstIndex i;
  blitz::secondIndex j;
  
  //do the computation of the feature
  for (int l=0; l < glcm_norm.extent(2); ++l)
  {
    mat = glcm_norm(rall, rall, l);
    prop(l) = blitz::sum((i-j)*(i-j)*mat);
  } 
  /*
  //as done in [1]
  for (int l=0; l < glcm_norm.extent(2); ++l)
  {
    mat = glcm_norm(rall, rall, l);
    double contrast = 0;
    for (int t=0; t < glcm_norm.extent(0) - 1; ++t) // iterate through all the levels
    {
      contrast += t*t*blitz::sum(blitz::where(abs(i-j)==t, mat, 0));
    } 
    prop(l) = contrast;
  }
  */
}

void bob::ip::GLCMProp::auto_correlation(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const
{
  // check if the size of the output matrix is as expected
  blitz::TinyVector<int,1> shape(get_prop_shape(glcm));
  bob::core::array::assertSameShape(prop, shape); 
  
  // normalize the input GLCM matrix
  blitz::Array<double,3> glcm_norm = normalize_glcm(glcm);
  
  blitz::Array<double,2> mat(glcm.extent(0), glcm.extent(1)); // auxiliary matrix that will be used for glcm matrix for one particular offset
  
  blitz::Range rall = blitz::Range::all();
  blitz::firstIndex i;
  blitz::secondIndex j;
  
  //do the computation of the feature
  for (int l=0; l < glcm_norm.extent(2); ++l)
  {
    mat = glcm_norm(rall, rall, l);
    prop(l) = blitz::sum(i*j*mat);
  }
}


void bob::ip::GLCMProp::correlation(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const
{
  // check if the size of the output matrix is as expected
  blitz::TinyVector<int,1> shape(get_prop_shape(glcm));
  bob::core::array::assertSameShape(prop, shape); 
  
  // normalize the input GLCM matrix
  blitz::Array<double,3> glcm_norm = normalize_glcm(glcm);
  
  blitz::Array<double,2> mat(glcm.extent(0), glcm.extent(1)); // auxiliary matrix that will be used for glcm matrix for one particular offset
  
  blitz::Range rall = blitz::Range::all();
  blitz::firstIndex i;
  blitz::secondIndex j;
  
  //do the computation of the feature
  for (int l=0; l < glcm_norm.extent(2); ++l)
  {
    mat = glcm_norm(rall, rall, l);
    double mean_x = blitz::sum(i*mat);
    double mean_y = blitz::sum(j*mat);
    double std_x = sqrt(blitz::sum(sqr(i-mean_x)*mat));
    double std_y = sqrt(blitz::sum(sqr(j-mean_y)*mat));
    prop(l) = (blitz::sum(i*j*mat) - mean_x*mean_y) / (std_x * std_y);
  }
}

void bob::ip::GLCMProp::correlation_m(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const
{
  // check if the size of the output matrix is as expected
  blitz::TinyVector<int,1> shape(get_prop_shape(glcm));
  bob::core::array::assertSameShape(prop, shape); 
  
  // normalize the input GLCM matrix
  blitz::Array<double,3> glcm_norm = normalize_glcm(glcm);
  
  blitz::Array<double,2> mat(glcm.extent(0), glcm.extent(1)); // auxiliary matrix that will be used for glcm matrix for one particular offset
  
  blitz::Range rall = blitz::Range::all();
  blitz::firstIndex i;
  blitz::secondIndex j;
  
  //do the computation of the feature
  for (int l=0; l < glcm_norm.extent(2); ++l)
  {
    mat = glcm_norm(rall, rall, l);
    double mean_x = blitz::sum(i*mat);
    double mean_y = blitz::sum(j*mat);
    double std_x = sqrt(blitz::sum(sqr(i-mean_x)*mat));
    double std_y = sqrt(blitz::sum(sqr(j-mean_y)*mat));
    prop(l) = blitz::sum(((i-mean_x) * (j-mean_x) * mat) / (std_x * std_y));
  }
}

void bob::ip::GLCMProp::inv_diff_mom(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const
{ 
  // check if the size of the output matrix is as expected
  blitz::TinyVector<int,1> shape(get_prop_shape(glcm));
  bob::core::array::assertSameShape(prop, shape); 
  
  // normalize the input GLCM matrix
  blitz::Array<double,3> glcm_norm = normalize_glcm(glcm);
  
  blitz::Array<double,2> mat(glcm.extent(0), glcm.extent(1)); // auxiliary matrix that will be used for glcm matrix for one particular offset
  
  blitz::Range rall = blitz::Range::all();
  blitz::firstIndex i;
  blitz::secondIndex j;
  
  //do the computation of the feature
  for (int l=0; l < glcm_norm.extent(2); ++l)
  {
    mat = glcm_norm(rall, rall, l);
    prop(l) = blitz::sum(mat / (1 + sqr(i-j)));
  }
}


void bob::ip::GLCMProp::sum_avg(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const
{
  // check if the size of the output matrix is as expected
  blitz::TinyVector<int,1> shape(get_prop_shape(glcm));
  bob::core::array::assertSameShape(prop, shape); 
  
  // normalize the input GLCM matrix
  blitz::Array<double,3> glcm_norm = normalize_glcm(glcm);
  
  blitz::Array<double,2> mat(glcm.extent(0), glcm.extent(1)); // auxiliary matrix that will be used for glcm matrix for one particular offset
  
  blitz::Range rall = blitz::Range::all();
  blitz::firstIndex i;
  blitz::secondIndex j;
  
  //do the computation of the feature
  for (int l=0; l < glcm_norm.extent(2); ++l)
  {
    mat = glcm_norm(rall, rall, l);
    double sum_avg = 0;
    for (int t = 0; t < 2 * glcm_norm.extent(0) - 1; t++) // iterate through all the levels
    {
      sum_avg += t * blitz::sum(blitz::where(i+j==t, mat, 0));
    }
    prop(l) = sum_avg;
  }  
}


void bob::ip::GLCMProp::sum_var(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const
{
  // check if the size of the output matrix is as expected
  blitz::TinyVector<int,1> shape(get_prop_shape(glcm));
  bob::core::array::assertSameShape(prop, shape); 
  
  // normalize the input GLCM matrix
  blitz::Array<double,3> glcm_norm = normalize_glcm(glcm);
  
  blitz::Array<double,2> mat(glcm.extent(0), glcm.extent(1)); // auxiliary matrix that will be used for glcm matrix for one particular offset
  
  blitz::Range rall = blitz::Range::all();
  blitz::firstIndex i;
  blitz::secondIndex j;
  
  //do the computation of the feature
  blitz::Array<double,1>& prop_sum_entropy(prop);
  sum_entropy(glcm, prop_sum_entropy);
  for (int l=0; l < glcm_norm.extent(2); ++l)
  {
    mat = glcm_norm(rall, rall, l);
    double sum_var = 0;
    for (int t = 0; t < 2 * glcm_norm.extent(0) -1; t++) // iterate through all the levels
    {
      sum_var += sqr(t-prop_sum_entropy(l)) * blitz::sum(blitz::where(i+j==t, mat, 0));
    }
    prop(l) = sum_var;
  }
}

void bob::ip::GLCMProp::sum_entropy(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const
{
  // check if the size of the output matrix is as expected
  blitz::TinyVector<int,1> shape(get_prop_shape(glcm));
  bob::core::array::assertSameShape(prop, shape); 
  
  // normalize the input GLCM matrix
  blitz::Array<double,3> glcm_norm = normalize_glcm(glcm);
  
  blitz::Array<double,2> mat(glcm.extent(0), glcm.extent(1)); // auxiliary matrix that will be used for glcm matrix for one particular offset
  
  blitz::Range rall = blitz::Range::all();
  blitz::firstIndex i;
  blitz::secondIndex j;
  
  //do the computation of the feature
  for (int l=0; l < glcm_norm.extent(2); ++l)
  {
    mat = glcm_norm(rall, rall, l);
    double sum_entropy = 0;
    for (int t = 0; t < 2 * glcm_norm.extent(0) - 1; t++) // iterate through all grey levels
    {
      sum_entropy += blitz::sum(blitz::where(i+j==t, mat, 0)) * log(blitz::sum(blitz::where(i+j==t, mat, 0)) + std::numeric_limits<double>::min());
    }
    prop(l) = -sum_entropy; 
  }
}
   
   
void bob::ip::GLCMProp::entropy(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const 
{
  // check if the size of the output matrix is as expected
  blitz::TinyVector<int,1> shape(get_prop_shape(glcm));
  bob::core::array::assertSameShape(prop, shape); 
  
  // normalize the input GLCM matrix
  blitz::Array<double,3> glcm_norm = normalize_glcm(glcm);
  
  blitz::Array<double,2> mat(glcm.extent(0), glcm.extent(1)); // auxiliary matrix that will be used for glcm matrix for one particular offset
  
  blitz::Range rall = blitz::Range::all();
  blitz::firstIndex i;
  blitz::secondIndex j;
  
  //do the computation of the feature
  for (int l=0; l < glcm_norm.extent(2); ++l)
  {
    mat = glcm_norm(rall, rall, l);
    prop(l) = -blitz::sum(mat * blitz::log(mat + std::numeric_limits<double>::min())); // small numeric value is added to avoid 0 as an argument to the logarithm
  }
}

      
void bob::ip::GLCMProp::diff_var(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const
{
  // check if the size of the output matrix is as expected
  blitz::TinyVector<int,1> shape(get_prop_shape(glcm));
  bob::core::array::assertSameShape(prop, shape); 
  
  // normalize the input GLCM matrix
  blitz::Array<double,3> glcm_norm = normalize_glcm(glcm);
  
  blitz::Array<double,2> mat(glcm.extent(0), glcm.extent(1)); // auxiliary matrix that will be used for glcm matrix for one particular offset
  
  blitz::Range rall = blitz::Range::all();
  blitz::firstIndex i;
  blitz::secondIndex j;
  
  //do the computation of the feature
  for (int l=0; l < glcm_norm.extent(2); ++l)
  {
    mat = glcm_norm(rall, rall, l);
    double diff_var = 0;
    for (int t = 0; t < glcm_norm.extent(0); t++) // iterate through all grey levels
    {
      diff_var +=  t * t * blitz::sum(blitz::where(abs(i-j)==t, mat, 0));
    }
    prop(l) = diff_var; 
  }
}


void bob::ip::GLCMProp::diff_entropy(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const
{
  // check if the size of the output matrix is as expected
  blitz::TinyVector<int,1> shape(get_prop_shape(glcm));
  bob::core::array::assertSameShape(prop, shape); 
  
  // normalize the input GLCM matrix
  blitz::Array<double,3> glcm_norm = normalize_glcm(glcm);
  
  blitz::Array<double,2> mat(glcm.extent(0), glcm.extent(1)); // auxiliary matrix that will be used for glcm matrix for one particular offset
  
  blitz::Range rall = blitz::Range::all();
  blitz::firstIndex i;
  blitz::secondIndex j;
  
  //do the computation of the feature
  for (int l=0; l < glcm_norm.extent(2); ++l)
  {
    mat = glcm_norm(rall, rall, l);
    double diff_entropy = 0;
    for (int t = 0; t < glcm_norm.extent(0); t++) // iterate through all grey levels
    {
      diff_entropy += blitz::sum(blitz::where(abs(i-j)==t, mat, 0)) * log(blitz::sum(blitz::where(abs(i-j)==t, mat, 0)) + std::numeric_limits<double>::min());
    }
    prop(l) = -diff_entropy; 
  }
}


void bob::ip::GLCMProp::dissimilarity(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const
{ 
  // check if the size of the output matrix is as expected
  blitz::TinyVector<int,1> shape(get_prop_shape(glcm));
  bob::core::array::assertSameShape(prop, shape); 
  
  // normalize the input GLCM matrix
  blitz::Array<double,3> glcm_norm = normalize_glcm(glcm);
  
  blitz::Array<double,2> mat(glcm.extent(0), glcm.extent(1)); // auxiliary matrix that will be used for glcm matrix for one particular offset
  
  blitz::Range rall = blitz::Range::all();
  blitz::firstIndex i;
  blitz::secondIndex j;
  
  //do the computation of the feature
  for (int l=0; l < glcm_norm.extent(2); ++l)
  {
    mat = glcm_norm(rall, rall, l);
    prop(l) = blitz::sum(abs(i-j)*mat);
  }
}


void bob::ip::GLCMProp::homogeneity(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const
{
  // check if the size of the output matrix is as expected
  blitz::TinyVector<int,1> shape(get_prop_shape(glcm));
  bob::core::array::assertSameShape(prop, shape); 
  
  // normalize the input GLCM matrix
  blitz::Array<double,3> glcm_norm = normalize_glcm(glcm);
  
  blitz::Array<double,2> mat(glcm.extent(0), glcm.extent(1)); // auxiliary matrix that will be used for glcm matrix for one particular offset
  
  blitz::Range rall = blitz::Range::all();
  blitz::firstIndex i;
  blitz::secondIndex j;
  
  //do the computation of the feature
  for (int l=0; l < glcm_norm.extent(2); ++l)
  {
    mat = glcm_norm(rall, rall, l);
    prop(l) = blitz::sum(mat / (1 + abs(i-j)));
  }  
}


void bob::ip::GLCMProp::cluster_prom(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const
{
  // check if the size of the output matrix is as expected
  blitz::TinyVector<int,1> shape(get_prop_shape(glcm));
  bob::core::array::assertSameShape(prop, shape); 
  
  // normalize the input GLCM matrix
  blitz::Array<double,3> glcm_norm = normalize_glcm(glcm);
  
  blitz::Array<double,2> mat(glcm.extent(0), glcm.extent(1)); // auxiliary matrix that will be used for glcm matrix for one particular offset
  
  blitz::Range rall = blitz::Range::all();
  blitz::firstIndex i;
  blitz::secondIndex j;
  
  //do the computation of the feature
  for (int l=0; l < glcm_norm.extent(2); ++l)
  {
    mat = glcm_norm(rall, rall, l);
    double mean_x = blitz::sum(i*mat);
    double mean_y = blitz::sum(j*mat);
    prop(l) = blitz::sum(pow(i + j - mean_x - mean_y, 4) * mat);
  }
}

void bob::ip::GLCMProp::cluster_shade(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const
{
  // check if the size of the output matrix is as expected
  blitz::TinyVector<int,1> shape(get_prop_shape(glcm));
  bob::core::array::assertSameShape(prop, shape); 
  
  // normalize the input GLCM matrix
  blitz::Array<double,3> glcm_norm = normalize_glcm(glcm);
  
  blitz::Array<double,2> mat(glcm.extent(0), glcm.extent(1)); // auxiliary matrix that will be used for glcm matrix for one particular offset
  
  blitz::Range rall = blitz::Range::all();
  blitz::firstIndex i;
  blitz::secondIndex j;
  
  //do the computation of the feature
  for (int l=0; l < glcm_norm.extent(2); ++l)
  {
    mat = glcm_norm(rall, rall, l);
    double mean_x = blitz::sum(i*mat);
    double mean_y = blitz::sum(j*mat);
    prop(l) = blitz::sum(pow(i + j - mean_x - mean_y, 3) * mat);
  }
}

void bob::ip::GLCMProp::max_prob(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const
{
  // check if the size of the output matrix is as expected
  blitz::TinyVector<int,1> shape(get_prop_shape(glcm));
  bob::core::array::assertSameShape(prop, shape); 
  
  // normalize the input GLCM matrix
  blitz::Array<double,3> glcm_norm = normalize_glcm(glcm);
  
  blitz::Array<double,2> mat(glcm.extent(0), glcm.extent(1)); // auxiliary matrix that will be used for glcm matrix for one particular offset
  
  blitz::Range rall = blitz::Range::all();
  blitz::firstIndex i;
  blitz::secondIndex j;
  
  //do the computation of the feature
  for (int l=0; l < glcm_norm.extent(2); ++l)
  {
    mat = glcm_norm(rall, rall, l);
    prop(l) = blitz::max(mat);
  }
}

void bob::ip::GLCMProp::inf_meas_corr1(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const
{
  // check if the size of the output matrix is as expected
  blitz::TinyVector<int,1> shape(get_prop_shape(glcm));
  bob::core::array::assertSameShape(prop, shape); 
  
  // normalize the input GLCM matrix
  blitz::Array<double,3> glcm_norm = normalize_glcm(glcm);
  
  blitz::Array<double,2> mat(glcm.extent(0), glcm.extent(1)); // auxiliary matrix that will be used for glcm matrix for one particular offset
  
  blitz::Range rall = blitz::Range::all();
  blitz::firstIndex i;
  blitz::secondIndex j;
  
  //do the computation of the feature
  blitz::Array<double,1>& prop_entropy(prop);
  entropy(glcm, prop_entropy); //calculate the entropy
  for (int l=0; l < glcm_norm.extent(2); ++l)
  {
    mat = glcm_norm(rall, rall, l);
    
    blitz::Array<double,1> marg_prob_i(blitz::sum(mat,j)); // marginal probability of first dimension (i.e. row-wise sum)
    blitz::Array<double,1> marg_prob_j(blitz::sum(mat(j,i),j)); // marginal probability of second dimension (i.e. column-wise sum)

    double hxy1 = -blitz::sum(mat * blitz::log(marg_prob_i(i) * marg_prob_j(j) + std::numeric_limits<double>::min())); // small numeric value is added to avoid 0 as an argument to the logarithm
    double px_entropy = -blitz::sum(marg_prob_i * blitz::log(marg_prob_i + std::numeric_limits<double>::min()));
    double py_entropy = -blitz::sum(marg_prob_j * blitz::log(marg_prob_j + std::numeric_limits<double>::min()));
    prop(l) = (prop_entropy(l) - hxy1) / std::max(px_entropy, py_entropy);  
  }
}

void bob::ip::GLCMProp::inf_meas_corr2(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const
{
  
  // check if the size of the output matrix is as expected
  blitz::TinyVector<int,1> shape(get_prop_shape(glcm));
  bob::core::array::assertSameShape(prop, shape); 
  
  // normalize the input GLCM matrix
  blitz::Array<double,3> glcm_norm = normalize_glcm(glcm);
  
  blitz::Array<double,2> mat(glcm.extent(0), glcm.extent(1)); // auxiliary matrix that will be used for glcm matrix for one particular offset
  
  blitz::Range rall = blitz::Range::all();
  blitz::firstIndex i;
  blitz::secondIndex j;
  
  //do the computation of the feature
  blitz::Array<double,1>& prop_entropy(prop);
  entropy(glcm, prop_entropy); //calculate the entropy
  for (int l=0; l < glcm_norm.extent(2); ++l)
  {
    mat = glcm_norm(rall, rall, l);
    blitz::Array<double,1> marg_prob_i(blitz::sum(mat,j)); // marginal probability of first dimension (i.e. row-wise sum)
    blitz::Array<double,1> marg_prob_j(blitz::sum(mat(j,i),j)); // marginal probability of second dimension (i.e. column-wise sum)
    
    double hxy2 = -blitz::sum(marg_prob_i(i) * marg_prob_j(j) * blitz::log(marg_prob_i(i) * marg_prob_j(j) + std::numeric_limits<double>::min())); // small numeric value is added to avoid 0 as an argument to the logarithm
    prop(l) = sqrt(1 - exp(-2 * (hxy2 - prop_entropy(l))));
  }
  
}

void bob::ip::GLCMProp::inv_diff(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const
{
  homogeneity(glcm, prop);
}

void bob::ip::GLCMProp::inv_diff_norm(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const
{
  // check if the size of the output matrix is as expected
  blitz::TinyVector<int,1> shape(get_prop_shape(glcm));
  bob::core::array::assertSameShape(prop, shape); 
  
  // normalize the input GLCM matrix
  blitz::Array<double,3> glcm_norm = normalize_glcm(glcm);
  
  blitz::Array<double,2> mat(glcm.extent(0), glcm.extent(1)); // auxiliary matrix that will be used for glcm matrix for one particular offset
  
  blitz::Range rall = blitz::Range::all();
  blitz::firstIndex i;
  blitz::secondIndex j;
  
  //do the computation of the feature
  for (int l=0; l < glcm_norm.extent(2); ++l)
  {
    mat = glcm_norm(rall, rall, l);
    prop(l) = blitz::sum(mat / (1 + (abs(i-j) / (double)mat.extent(0)) ));
  }  
}

void bob::ip::GLCMProp::inv_diff_mom_norm(const blitz::Array<double,3>& glcm, blitz::Array<double,1>& prop) const
{
  // check if the size of the output matrix is as expected
  blitz::TinyVector<int,1> shape(get_prop_shape(glcm));
  bob::core::array::assertSameShape(prop, shape); 
  
  // normalize the input GLCM matrix
  blitz::Array<double,3> glcm_norm = normalize_glcm(glcm);
  
  blitz::Array<double,2> mat(glcm.extent(0), glcm.extent(1)); // auxiliary matrix that will be used for glcm matrix for one particular offset
  
  blitz::Range rall = blitz::Range::all();
  blitz::firstIndex i;
  blitz::secondIndex j;
  
  //do the computation of the feature
  for (int l=0; l < glcm_norm.extent(2); ++l)
  {
    mat = glcm_norm(rall, rall, l);
    prop(l) = blitz::sum(mat / (1 + (sqr(i-j) / sqr(mat.extent(0)))));
  }
}










