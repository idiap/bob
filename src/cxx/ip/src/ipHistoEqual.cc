/**
 * @file cxx/ip/src/ipHistoEqual.cc
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
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
#include "ip/ipHistoEqual.h"

/////////////////////////////////////////////////////////////////////////////////////////

namespace Torch {

/////////////////////////////////////////////////////////////////////////////////////////
// Constructor
ipHistoEqual::ipHistoEqual() 
	    : 	ipCore()
{
}

/////////////////////////////////////////////////////////////////////////////////////////
// Destructor
ipHistoEqual::~ipHistoEqual()
{
}

/////////////////////////////////////////////////////////////////////////////////////////
// Check if the input tensor has the right dimensions and type
bool ipHistoEqual::checkInput(const Tensor& input) const
{
	// Accept only 3D tensors of Torch::Image type
	if (    input.nDimension() != 3 ||
	        input.getDatatype() != Tensor::Short
		)
	{
		warning("ipHistoEqual::checkInput(): Incorrect Tensor type and dimension.");
		return false;
	}
	// Accept only grayscale images
	if (	input.size(2) !=1 )
	{
		warning("ipHistoEqual::checkInput(): Non grayscale image (multiple channels).");
		return false;
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////////////////////
// Allocate (if needed) the output tensors given the input tensor dimensions
bool ipHistoEqual::allocateOutput(const Tensor& input)
{
	if (	m_output == 0 ||
		m_output[0]->nDimension() != 3 ||
		m_output[0]->size(0) != input.size(0) ||
		m_output[0]->size(0) != input.size(1) ||
		m_output[0]->size(0) != input.size(2) )
	{
		cleanup();

		// Need allocation
		m_n_outputs = 1;
		m_output = new Tensor*[m_n_outputs];
		m_output[0] = new ShortTensor(input.size(0), input.size(1), input.size(2));
		return true;
	}

	return true;
}

/////////////////////////////////////////////////////////////////////////////////////////
// Process some input tensor (the input is checked, the outputs are allocated)
bool ipHistoEqual::processInput(const Tensor& input)
{
	//Process single channel images

	//Prepare pointers to access pixels
	const ShortTensor* t_input = (ShortTensor*)&input;
	ShortTensor* t_output = (ShortTensor*)m_output[0];

	DoubleTensor histo(256);
  histo.fill(0);
  DoubleTensor chisto(256);

  //Computes histogram and cumulative histogram: remember, only gray-scale
  //accepted!
  const int height = input.size(0);
  const int width = input.size(1);
  //First we calculate a simple histogram, just the bin count
	for (int x=0; x<width; ++x)
		for (int y = 0; y<height; ++y) ++histo((*t_input)(y, x, 0));

  //Now we normalize the histogram w.r.t to the image dimensions and calculate
  //the cumulative histograms. For that, a little iteration trick
  histo(0) /= (height * width);
  chisto(0) = histo(0);
  for (int k=1; k<256; ++k) {
    histo(k) /= (height * width);
    chisto(k) = histo(k) + chisto(k-1);
  }

  //Now we normalize the cumulative histogram in the regular way
  for (int k=0; k<256; ++k)
    chisto(k) = 255 * (chisto(k)-chisto(0)) / chisto(255);

	//Fills in the output, normalizing the image
	for (int x=0; x<width; ++x)
		for (int y = 0; y<height; ++y) 
      (*t_output)(y, x, 0) = FixI(chisto((*t_input)(y, x, 0)));

	return true;
}

} 

