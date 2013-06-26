/**
 * @file machine/python/roll.cc
 * @date Tue Jun 25 19:09:50 CEST 2013
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
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
  stl_input_iterator<blitz::Array<double,2> > wbegin(w), wend;
  std::vector<blitz::Array<double,2> > w_(wbegin, wend);
  stl_input_iterator<blitz::Array<double,1> > bbegin(b), bend;
  std::vector<blitz::Array<double,1> > b_(bbegin, bend);
  blitz::Array<double,1> vec_ = vec.bz<double,1>();
  bob::machine::unroll(w_, b_, vec_);
}

static object unroll3(object w, object b)
{
  stl_input_iterator<blitz::Array<double,2> > wbegin(w), wend;
  std::vector<blitz::Array<double,2> > w_(wbegin, wend);
  stl_input_iterator<blitz::Array<double,1> > bbegin(b), bend;
  std::vector<blitz::Array<double,1> > b_(bbegin, bend);
  bob::python::ndarray vec(bob::core::array::t_float64, 
    bob::machine::detail::getNbParameters(w_, b_));
  blitz::Array<double,1> vec_ = vec.bz<double,1>();
  bob::machine::unroll(w_, b_, vec_);
  return vec.self();
}

static void roll1(bob::machine::MLP& m, bob::python::const_ndarray vec)
{
  blitz::Array<double,1> vec_ = vec.bz<double,1>();
  bob::machine::roll(m, vec_);
}

static void roll2(list w, list b, bob::python::const_ndarray vec)
{
  stl_input_iterator<blitz::Array<double,2> > wbegin(w), wend;
  std::vector<blitz::Array<double,2> > w_(wbegin, wend);
  stl_input_iterator<blitz::Array<double,1> > bbegin(b), bend;
  std::vector<blitz::Array<double,1> > b_(bbegin, bend);
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
