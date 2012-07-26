#ifndef BOB_VISIONER_IMAGE_H
#define BOB_VISIONER_IMAGE_H

#include <QImage>

#include "visioner/vision/vision.h"

namespace bob { namespace visioner {

  /////////////////////////////////////////////////////////////////////////////////////////
  // Image: loads a grayscale image from disk.
  /////////////////////////////////////////////////////////////////////////////////////////

  // Loads an image
  bool load(const QImage& qimage, greyimage_t& grays);
  bool load(const string_t& filename, greyimage_t& grays);

  // Scale the image to a specific <scale> of the <src> source image
  bool scale(const greyimage_t& src, scalar_t scale, greyimage_t& dst);

  // Convert from <greyimage_t> to <QImage>
  QImage convert(const greyimage_t& grays);
  bool convert(const QImage& qimage, greyimage_t& grays);

}}

#endif // BOB_VISIONER_IMAGE_H
