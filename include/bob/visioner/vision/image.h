/**
 * @file bob/visioner/vision/image.h
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_VISIONER_IMAGE_H
#define BOB_VISIONER_IMAGE_H

#include <QImage>

#include "bob/visioner/vision/vision.h"
#include "bob/visioner/util/matrix.h"

namespace bob { namespace visioner {

  /////////////////////////////////////////////////////////////////////////////////////////
  // Image: loads a grayscale image from disk.
  /////////////////////////////////////////////////////////////////////////////////////////

  // Loads an image
  bool load(const QImage& qimage, Matrix<uint8_t>& grays);
  bool load(const std::string& filename, Matrix<uint8_t>& grays);

  // Scale the image to a specific <scale> of the <src> source image
  bool scale(const Matrix<uint8_t>& src, double scale, Matrix<uint8_t>& dst);

  // Convert from <Matrix<uint8_t>> to <QImage>
  QImage convert(const Matrix<uint8_t>& grays);
  bool convert(const QImage& qimage, Matrix<uint8_t>& grays);

}}

#endif // BOB_VISIONER_IMAGE_H
