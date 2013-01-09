/**
 * @file sp/python/ceps.cc
 * @date Wed Jan 11:10:40 2013 +0200
 * @author Elie El Khoury <Elie.Khoury@idiap.ch>
 *
 * @brief Binds the Cepstral Feature Extraction to python.
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

#include <boost/python.hpp>

#include "bob/sp/Ceps.h"
#include "bob/core/python/ndarray.h"

using namespace boost::python;

// documentation for classes
static const char* CEPS_DOC = "Objects of this class, after configuration, can extract Cepstral Features from a 1D array/signal.";
static const char* TESTCEPS_DOC = "Objects of this class, after configuration, can be used to test the private methods of bob.sp.Ceps.";

static object py_ceps_analysis(bob::sp::Ceps& ceps, int n_size)
{
	// Get the shape of the feature
	blitz::TinyVector<size_t,2> s = ceps.getCepsShape(n_size);

	// Allocate numpy array and define blitz wrapper
	bob::python::ndarray ceps_matrix(bob::core::array::t_float64, s(0), s(1));
	blitz::Array<double,2> ceps_matrix_ = ceps_matrix.bz<double,2>();

	// Extract the features
	ceps.CepsAnalysis(n_size, ceps_matrix_);
	return ceps_matrix.self();
}

static boost::python::tuple py_get_ceps_shape(bob::sp::Ceps& ceps, int n_size)
{
	blitz::TinyVector<int,2> size = ceps.getCepsShape(n_size);
	return boost::python::make_tuple(size[0], size[1]);
}

static double py_logEnergy(bob::sp::TestCeps& ceps, bob::python::ndarray data)
{
	blitz::Array<double,1> data_ = data.bz<double,1>();

	// Get the logEnergy
	return ceps.logEnergy(data_);
}

static void py_emphasis(bob::sp::TestCeps& ceps, bob::python::ndarray data, int n, double a)
{
	blitz::Array<double,1> data_ = data.bz<double,1>();

	//Compute the Pre-Emphasis
	ceps.emphasis(data_, n, a);
}

static void py_hammingWindow(bob::sp::TestCeps& ceps, bob::python::ndarray data)
{
	blitz::Array<double,1> data_ = data.bz<double,1>();

	//Compute the Hamming Wrapping
	ceps.hammingWindow(data_);
}

static object py_logFilterBank(bob::sp::TestCeps& ceps, bob::python::ndarray data, int m_win_size, int n_filters)
{

	blitz::Array<double,1> data_ = data.bz<double,1>();
	ceps.logFilterBank(data_, m_win_size);
	bob::python::ndarray filter(bob::core::array::t_float64, n_filters);
	blitz::Array<double,1> filter_ = filter.bz<double,1>();

	// Get the filter Bank
	filter_ = ceps.getFilter();
	return filter.self();
}

static object py_DCTransform(bob::sp::TestCeps& ceps, int n_ceps)
{
	ceps.DCTransform();
	bob::python::ndarray features(bob::core::array::t_float64, n_ceps);
	blitz::Array<double,1> features_ = features.bz<double,1>();

	// Get the Cepstral features
	features_ = ceps.getFeatures();
	return features.self();
}


void bind_sp_ceps()
{
	class_<bob::sp::Ceps, boost::shared_ptr<bob::sp::Ceps> >("Ceps", CEPS_DOC, init<double, int, int, int, int, double, double, double, const blitz::Array<double,1>& >
	((arg("sf"), arg("win_length_ms"), arg("win_shift_ms"), arg("n_filters"), arg("n_ceps"), arg("f_min"), arg("f_max"), arg("delta_win"), arg("data_array"))))
				.def("ceps_analysis", &py_ceps_analysis, (arg("m_dct_norm"),  arg("n_size")), "Compute the features")
				.def("get_ceps_shape", &py_get_ceps_shape, (arg("n_size")), "Compute the shape of the output features")
				.def("reinit", &bob::sp::Ceps::reinit, (arg("m_dct_norm"), arg("fb_linear"), arg("withEnergy"), arg("withDelta"),
										arg("withDeltaDelta"), arg("withDeltaEnergy"), arg("withDeltaDeltaEnergy")),"reinitialisation")
				;

	class_<bob::sp::TestCeps, boost::shared_ptr<bob::sp::TestCeps> >("TestCeps", TESTCEPS_DOC, init<bob::sp::Ceps&>((arg("ceps"))))
				.def("mel", &bob::sp::TestCeps::mel, (arg("f")), "Compute a mel scale.")
				.def("mel_inv", &bob::sp::TestCeps::MelInv, (arg("f")), "Compute an inverse mel scale.")
				.def("log_energy", &py_logEnergy, (arg("data")), "compute the gain")
				.def("pre_emphasis", &py_emphasis, (arg("data"), arg("n"),arg("a")), "compute pre-emphasis")
				.def("hamming_window", &py_hammingWindow, (arg("data")), "compute the wraped signal on a hamming Window")
				.def("log_filter_bank", &py_logFilterBank, (arg("data"), arg("m_win_size"), arg("n_filters")), "compute log Filter Bank")
				.def("dct_transform", &py_DCTransform, (arg("n_ceps")), "DCT Transform")
			;
}

