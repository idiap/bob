/**
 * @file cxx/ip/test/Gabor.cc
 * @date Wed Apr 13 20:12:03 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Test the Gabor filtering function for 2D arrays/images
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
#define BOOST_TEST_MODULE IP-Gabor Tests
#define BOOST_TEST_MAIN
#include "core/logging.h"
#include "core/cast.h"
#include "core/convert.h"
#include <io/Array.h>
#include <blitz/array.h>
#include "ip/GaborWaveletTransform.h"

#include <boost/filesystem.hpp>
#include <sstream>

int main(int argc, char** argv)
{
#if 1
#if 0
  // Get path to the XML Schema definition
  char *testdata_cpath = getenv("TORCH_IP_TESTDATA_DIR");
  if( !testdata_cpath || !strcmp( testdata_cpath, "") ) {
    Torch::core::error << "Environment variable $TORCH_IP_TESTDATA_DIR " <<
      "is not set. " << "Have you setup your working environment " <<
      "correctly?" << std::endl;
    throw Torch::core::Exception();
  }
#endif
  // Load original image
  std::string image_name = "/idiap/home/mguenther/Source/torch5spro/src/cxx/ip/test/data/image.pgm";
  bob::io::Array read_image(image_name);
  blitz::Array<double,2> input_image = bob::core::fastCast<double>(read_image.get<unsigned char,2>());

  // perform GWT
  bob::ip::GaborWaveletTransform gwt;
  blitz::Array<std::complex<double>,3> trafo_image;
  gwt.gwt(trafo_image,input_image);

  // write layers to file
  for (int j = 0; j < trafo_image.extent(0); ++j){
    // get layer
    blitz::Array<std::complex<double>,2> layer(trafo_image(j, blitz::Range::all(), blitz::Range::all()));
    // compute absolute value
    blitz::Array<double,2> abs_layer(layer.shape());
    bob::core::getPart(abs_layer, layer, bob::core::ABS_PART);

    double min = *std::min_element(abs_layer.begin(), abs_layer.end()),
           max = *std::max_element(abs_layer.begin(), abs_layer.end());
    abs_layer =  255. * (abs_layer - min)/ (max - min);

    bob::io::Array writeable(bob::core::fastCast<unsigned char>(abs_layer));
    std::ostringstream fn;
    fn << "/scratch/mguenther/temp/unnormalized_" << j << ".png";
    writeable.save(fn.str());
  }

  // generate jet image
  blitz::Array<double,4> jet_image;
  gwt.jetImage(jet_image,input_image);

  // write one jet to console
  std::cout << jet_image(0,10,20,blitz::Range::all());
  std::cout << jet_image(1,10,20,blitz::Range::all());

  // generate images from the jets
  for (int j = 0; j < trafo_image.extent(0); ++j){
    // get absolute layer
    blitz::Array<double,2> abs_layer(jet_image(0, blitz::Range::all(), blitz::Range::all(),j));

    double min = *std::min_element(abs_layer.begin(), abs_layer.end()),
           max = *std::max_element(abs_layer.begin(), abs_layer.end());
    abs_layer =  255. * (abs_layer - min)/ (max - min);

    bob::io::Array writeable(bob::core::cast<unsigned char>(abs_layer));
    std::ostringstream fn;
    fn << "/scratch/mguenther/temp/normalized_" << j << ".png";
    writeable.save(fn.str());
  }

#else
  blitz::Array<std::complex<double>,2> x(3,2);
  x = 1, 2, 3, 4, 5, 6;

  blitz::Array<double,2> y(3,2);
  bob::core::getPart(y,x,bob::core::REAL_PART);

  std::cout << x << y;

#endif
  return 0;
}


