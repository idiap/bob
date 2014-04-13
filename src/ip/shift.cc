/**
 * @file ip/cxx/shift.cc
 * @date Sun Apr 17 23:11:51 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include <stdexcept>
#include <boost/format.hpp>
#include "bob/ip/shift.h"

void bob::ip::detail::shiftParameterCheck( const int shift_y, const int shift_x,
  const size_t src_height, const size_t src_width)
{
  // Check parameters and throw exception if required
  if (shift_y <= -(int)src_height || shift_y >= (int)src_height) {
    boost::format m("setting `shift_y' to %d is outside the expected range [%d, %d]");
    m % shift_y % -((int)src_height+1) % ((int)src_height-1);
    throw std::runtime_error(m.str());
  }
  if (shift_x <= -(int)src_width || shift_x >= (int)src_width) {
    boost::format m("setting `shift_x' to %d is outside the expected range [%d, %d]");
    m % shift_x % -((int)src_width+1) % ((int)src_width-1);
    throw std::runtime_error(m.str());
  }
}

