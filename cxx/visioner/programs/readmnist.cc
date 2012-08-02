/**
 * @file visioner/programs/readmnist.cc
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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

#include <fstream>
#include <boost/lexical_cast.hpp>
#include <QImage>

#include "core/logging.h"

#include "visioner/vision/image.h"

bool read(const std::string& str_ifimage, const std::string& str_iflabel, int n_images, const std::string& str_ofimage, const std::string& str_oflabel)
{
  // Image and label data streams
  std::ifstream fimage(str_ifimage.c_str(), std::ios::in | std::ios::binary);
  std::ifstream flabel(str_iflabel.c_str(), std::ios::in | std::ios::binary);

  if (!fimage.is_open() || !flabel.is_open())
  {
    bob::core::error << "Cannot open MNIST files!" << std::endl;
    return false;
  }

  // Create buffers for image data and correct labels
  static const int BUF_SIZE = 2048;
  char buffer[BUF_SIZE];
  char label[2];

  // Read headers
  fimage.read(buffer, 16);
  flabel.read(buffer, 8);

  bob::visioner::greyimage_t image(28, 28);

  // Now cycle over all images in MNIST dataset
  for (int i = 0; i < n_images; i ++)
  {
    flabel.read(label, 1);		
    fimage.read(buffer, image.size());

    std::copy(buffer, buffer + image.size(), image[0]);		
    const QImage qimage = bob::visioner::convert(image);
    qimage.save((str_ofimage + boost::lexical_cast<std::string>((int)label[0]) + "/"
          + boost::lexical_cast<std::string>(i + 1) + ".png").c_str());		

    std::ofstream oflabel;
    oflabel.open((str_oflabel + boost::lexical_cast<std::string>((int)label[0]) + "/" 
          + boost::lexical_cast<std::string>(i + 1) + ".gt").c_str());
    oflabel << 1 << "\n" << (int)(label[0]) << " 0 0 28 28\n";
  }

  // OK
  return true;
}

int main(int argc, char* argv[]) {

  return	argc == 2 &&
    read(std::string(argv[1]) + "/t10k-images-idx3-ubyte",
        std::string(argv[1]) + "/t10k-labels-idx1-ubyte", 10000,
        std::string(argv[1]) + "/test", std::string(argv[1]) + "/test") == true &&
    read(std::string(argv[1]) + "/train-images-idx3-ubyte",
        std::string(argv[1]) + "/train-labels-idx1-ubyte", 60000,
        std::string(argv[1]) + "/train", std::string(argv[1]) + "/train") == true ?

    EXIT_SUCCESS : EXIT_FAILURE;

}
