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

