/**
 * @file visioner/src/image.cc
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

#include "visioner/vision/image.h"
#include "visioner/util/util.h"

namespace bob { namespace visioner {	

  // Loads an image
  bool load(const QImage& qimage, greyimage_t& grays)
  {
    return convert(qimage, grays);
  }

  bool load(const string_t& filename, greyimage_t& grays)
  {
    QImage qimage;
    return	qimage.load(filename.c_str()) &&
      load(qimage, grays);
  }

  // Scale the image to a specific <scale> of the <src> source image
  bool scale(const greyimage_t& src, scalar_t scale, greyimage_t& dst)
  {
    scale = range(scale, 0.01, 1.00);
    const int new_w = (int)(0.5 + scale * src.cols());
    const int new_h = (int)(0.5 + scale * src.rows());

    QImage qimage = convert(src);
    return load(qimage.scaled(new_w, new_h, Qt::KeepAspectRatio, Qt::SmoothTransformation), dst);
  }

  // Convert from <greyimage_t> to <QImage>
  QImage convert(const greyimage_t& grays)
  {
    QImage qimage(grays.cols(), grays.rows(), QImage::Format_RGB32);
    const int w = qimage.width(), h = qimage.height();

    const grey_t* ptr = &grays(0, 0);
    for (int y = 0; y < h; y ++)
    {
      QRgb* line = (QRgb*)qimage.scanLine(y);
      for (int x = 0; x < w; x ++)
      {
        const grey_t gray = *(ptr ++);
        *(line ++) = qRgb(gray, gray, gray);
      }
    }

    return qimage;
  }

  // Convert from <QImage> to <greyimage_t>
  bool convert(const QImage& qimage, greyimage_t& grays)
  {
    const int w = qimage.width(), h = qimage.height();
    grays.resize(h, w);

    switch (qimage.depth())
    {
      case 8:
        {
          const QVector<QRgb> colors = qimage.colorTable();
          grey_t* ptr = &grays(0, 0);
          for (int y = 0; y < h; y ++)
          {
            const unsigned char* line = (const unsigned char*)qimage.scanLine(y);				
            for (int x = 0; x < w; x ++)
            {
              *(ptr ++) = qGray(colors[*(line ++)]);
            }
          }
        }
        break;

      case 32:
        {
          grey_t* ptr = &grays(0, 0);
          for (int y = 0; y < h; y ++)
          {
            const QRgb* line = (const QRgb*)qimage.scanLine(y);
            for (int x = 0; x < w; x ++)
            {
              *(ptr ++) = qGray(*(line ++));
            }
          }
        }
        break;

      default:
        return false;
    }

    return true;
  }		

}}
