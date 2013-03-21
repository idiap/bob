/**
 * @file bob/core/blitz_compat.h
 * @date Mon Apr 11 10:29:29 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines blitz-related types for compatibility purpose
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

#ifndef BOB_CORE_BLITZ_COMPAT_H
#define BOB_CORE_BLITZ_COMPAT_H

#include <bob/config.h>

/**
 * Defines the diffType and sizeType in case blitz (old) don't have it defined
 * already.
 */
#if !defined(HAVE_BLITZ_SPECIAL_TYPES)
namespace blitz { 
  typedef int sizeType; 
  typedef int diffType;
}
#endif

#endif /* BOB_CORE_BLITZ_COMPAT_H */
