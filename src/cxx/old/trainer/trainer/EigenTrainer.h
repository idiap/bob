/**
 * @file cxx/old/trainer/trainer/EigenTrainer.h
 * @date Wed Apr 6 14:49:40 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
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
#ifndef _TORCH5SPRO_TRAINER_EIGEN_TRAINER_H_
#define _TORCH5SPRO_TRAINER_EIGEN_TRAINER_H_

namespace Torch {

/**
 * \ingroup libtrainer_api 
 * @{
 *
 */

/** This structure is designed to sort data (double)
    #the_int# contains generally the index of a structured object,
    while #the_real# contains the value by which we wish to sort the objects.
    @author Sebastien Marcel (marcel@idiap.ch)
*/
	struct int_double 
	{
		int the_int;
		double the_double;
	};

/** Additional functions to sort eigenvalues in double precision
    @author Sebastien Marcel (marcel@idiap.ch)
*/
	extern "C"
	{
		/// sort an int_double in increasing order
		int sort_inc_int_double(const void *p1, const void *p2);

		/// sort an int_double in decreasing order
		int sort_dec_int_double(const void *p1, const void *p2);
	}

/**
 * @}
 */
}

#endif

