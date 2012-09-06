/**
 * @file visioner/python/version.cc
 * @date Tue Nov 29 14:11:41 2011 +0100
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Describes ways to retrieve version information about all dependent
 * packages.
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

#include "bob/config.h"

#include <boost/python.hpp>
#ifdef HAVE_QT4
#include <qglobal.h>
#endif

using namespace boost::python;

/**
 * Qt4 version, if available
 */
static tuple qt4_version() {
#ifdef HAVE_QT4
  return make_tuple(str(QT_VERSION_STR), str(QT_PACKAGEDATE_STR));
#else
  return make_tuple(str("unavailable"), str("unknown"));
#endif
}

void bind_visioner_version() {
  dict vdict;
  vdict["Qt4"] = qt4_version();
  scope().attr("version") = vdict;
}
