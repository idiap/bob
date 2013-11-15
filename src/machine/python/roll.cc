/**
 * @file machine/python/roll.cc
 * @date Tue Jun 25 19:09:50 CEST 2013
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include <bob/python/ndarray.h>
#include <vector>
#include <bob/machine/roll.h>
#include <boost/python/stl_iterator.hpp>

using namespace boost::python;

static object unroll0(const bob::machine::MLP& m)
{ 
  bob::python::ndarray vec(bob::core::array::t_float64, 
    bob::machine::detail::getNbParameters(m));
  blitz::Array<double,1> vec_ = vec.bz<double,1>();
  bob::machine::unroll(m, vec_);
  return vec.self();
}

static void unroll1(const bob::machine::MLP& m, bob::python::ndarray vec)
{
  blitz::Array<double,1> vec_ = vec.bz<double,1>();
  bob::machine::unroll(m, vec_);
}

static void unroll2(object w, object b, bob::python::ndarray vec)
{
  stl_input_iterator<bob::python::const_ndarray> wbegin(w), wend;
  std::vector<bob::python::const_ndarray> wv(wbegin, wend);
  std::vector<blitz::Array<double,2> > w_;
  for(std::vector<bob::python::const_ndarray>::iterator it=wv.begin(); 
      it!=wv.end(); ++it)
    w_.push_back(it->bz<double,2>());

  stl_input_iterator<bob::python::const_ndarray> bbegin(b), bend;
  std::vector<bob::python::const_ndarray> bv(bbegin, bend);
  std::vector<blitz::Array<double,1> > b_;
  for(std::vector<bob::python::const_ndarray>::iterator it=bv.begin(); 
      it!=bv.end(); ++it)
    b_.push_back(it->bz<double,1>());

  blitz::Array<double,1> vec_ = vec.bz<double,1>();
  bob::machine::unroll(w_, b_, vec_);
}

static object unroll3(object w, object b)
{
  stl_input_iterator<bob::python::const_ndarray> wbegin(w), wend;
  std::vector<bob::python::const_ndarray> wv(wbegin, wend);
  std::vector<blitz::Array<double,2> > w_;
  for(std::vector<bob::python::const_ndarray>::iterator it=wv.begin(); 
      it!=wv.end(); ++it)
    w_.push_back(it->bz<double,2>());

  stl_input_iterator<bob::python::const_ndarray> bbegin(b), bend;
  std::vector<bob::python::const_ndarray> bv(bbegin, bend);
  std::vector<blitz::Array<double,1> > b_;
  for(std::vector<bob::python::const_ndarray>::iterator it=bv.begin(); 
      it!=bv.end(); ++it)
    b_.push_back(it->bz<double,1>());

  bob::python::ndarray vec(bob::core::array::t_float64, 
    bob::machine::detail::getNbParameters(w_, b_));
  blitz::Array<double,1> vec_ = vec.bz<double,1>();
  bob::machine::unroll(w_, b_, vec_);
  return vec.self();
}

static void roll1(bob::machine::MLP& m, bob::python::const_ndarray vec)
{
  bob::machine::roll(m, vec.bz<double,1>());
}

static void roll2(object w, object b, bob::python::const_ndarray vec)
{
  stl_input_iterator<bob::python::ndarray> wbegin(w), wend;
  std::vector<bob::python::ndarray> wv(wbegin, wend);
  std::vector<blitz::Array<double,2> > w_;
  for(std::vector<bob::python::ndarray>::iterator it=wv.begin(); 
      it!=wv.end(); ++it)
    w_.push_back(it->bz<double,2>());

  stl_input_iterator<bob::python::ndarray> bbegin(b), bend;
  std::vector<bob::python::ndarray> bv(bbegin, bend);
  std::vector<blitz::Array<double,1> > b_;
  for(std::vector<bob::python::ndarray>::iterator it=bv.begin(); 
      it!=bv.end(); ++it)
    b_.push_back(it->bz<double,1>());

  bob::machine::roll(w_, b_, vec.bz<double,1>());
}

void bind_machine_roll() {
  def("unroll", &unroll0, (arg("MLP")), "Unroll the parameters of an MLP into a single 1D numpy array");
  def("unroll", &unroll3, (arg("weights"), arg("biases")), "Unroll the parameters (weights and biases) into a single 1D numpy array.");
  def("unroll", &unroll1, (arg("MLP"), arg("vec")), "Unroll the parameters of an MLP into the 1D numpy array 'vec'. 'vec' should be allocated with the correct size.");
  def("unroll", &unroll2, (arg("weights"), arg("biases"), arg("vec")), "Unroll the parameters (weights and biases) into the 1D numpy array 'vec'. 'vec' should be allocated with the correct size.");

  def("roll", &roll1, (arg("MLP"), arg("vec")), "Roll the 1D numpy array 'vec' into the parameters (weights and biases) of the MLP.");
  def("roll", &roll2, (arg("weights"), arg("biases"), arg("vec")), "Roll the 1D numpy array 'vec' into the parameters (weights and biases)");
}
