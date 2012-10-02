/**
 * @file bob/io/Array.h
 * @date Wed Jun 22 17:50:08 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * The Array is the basic unit containing data in a Dataset
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

#ifndef BOB_IO_ARRAY_H 
#define BOB_IO_ARRAY_H

#include <boost/shared_ptr.hpp>
#include <blitz/array.h>
#include "bob/core/blitz_array.h"
#include "bob/io/File.h"

namespace bob {

  namespace io {

    /**
     * The array class for a dataset. The Array class acts like a manager for
     * the underlying data (blitz::Array<> in memory or serialized in file).
     */
    class Array {

      public:

        /**
         * Builds an Array that contains data from a file, specific (indexed)
         * data from the file is loaded using this constructor.
         *
         * Note: Not all file readers implement this cherry-picking mechanism
         * (e.g. image decoders). In these cases, the index parameter should be
         * set to the (default value of) zero.
         */
        Array(boost::shared_ptr<File> file, size_t index=0);

        /**
         * Refers to the Array data from another array.
         */
        Array(const Array& other);

        /**
         * Destroys this array. 
         */
        virtual ~Array();

        /**
         * Copies data from another array.
         */
        Array& operator= (const Array& other);

        /******************************************************************
         * Blitz Array specific manipulations
         ******************************************************************/

      private: //useful methods

      private: //representation
        boost::shared_ptr<File> m_external;
        ptrdiff_t m_index; ///< position on the file.
    };

  } //closes namespace io

} //closes namespace bob

#endif /* BOB_IO_ARRAY_H */
