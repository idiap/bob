/**
 * @file sp/python/conv.cc
 * @date Mon Aug 27 18:00:00 2012 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds convolution options
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
#include "bob/sp/conv.h"

using namespace boost::python;

void bind_sp_convolution() 
{
  enum_<bob::sp::Conv::SizeOption>("SizeOption")
    .value("Full", bob::sp::Conv::Full)
    .value("Same", bob::sp::Conv::Same)
    .value("Valid", bob::sp::Conv::Valid)
    ; 
}
