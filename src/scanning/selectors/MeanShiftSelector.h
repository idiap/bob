#ifndef _TORCHVISION_SCANNING_MEAN_SHIFT_SELECTOR_H_
#define _TORCHVISION_SCANNING_MEAN_SHIFT_SELECTOR_H_

#include "Selector.h"		// <MeanShiftSelector> is a <Selector>

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::Scanning::MeanShiftSelector
	//	- merges and selects the final pattern sub-windows using the Mean Shift
	//		clustering algorithm
	//
        // TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class MeanShiftSelector : public Selector
	{
	public:

		// Constructor
		MeanShiftSelector();

		// Destructor
		virtual ~MeanShiftSelector();

		// Delete all stored patterns
		virtual void		clear();

		// Process the list of candidate sub-windows and select the best ones
		// (this will accumulate them to the pattern list)
		virtual bool		process(const PatternSpace& candidates);

	private:

		// Deallocate the buffers
		void			deallocate();

		// Initialize the LSH structures
		void			initLSH(const PatternList& lpatterns, int image_w, int image_h);

		// Get the closest points to the given one (using LSH tables)
		// Returns the number of found points (less or equal than <max_size>)
		int			getClosest(const Pattern& pattern, int* iclosest, int* iclosestBuf, int max_size);

		// LSH hash table: <key, list of data (SWs) indexes>
		struct LSH_HashTable
		{
			// Constructor
			LSH_HashTable(int K = 0);

			// Destructor
			~LSH_HashTable();

			// Resize the number of inequalities
			void		resize(int K);

			// Reset the hash table and generate inequalities
			void		reset(int image_w, int image_h);

			// Add a new pattern
			void		add(const Pattern& pattern, int index_pattern);

			/////////////////////////////////////////////////////////
			// Attributes

			static const int HashSize = 769;//1024;

			struct Value
			{
				// Constructor
				Value();

				// Destructor
				~Value();

				// Resize the indexes already stored to accomodate new ones
				void	resize();

				// Attributes
				int*	m_indexes;
				int	m_size;
				int	m_capacity;
			};

			Value		m_values[HashSize];
			unsigned char*	m_d;	// Dimension index: (cx, cy, w, h) of the SW [K]
			short*		m_v;	// Value to test for <d> dimension [K]
			int		m_K;	// Number of inequalities for each partition
		};

		/////////////////////////////////////////////////////////////////
		// Attributes

		// LSH structures - fast computing the closest points in high dimensional space!
		// 	(Locality Sensitive Hashing)
		int			m_LSH_L;	// Number of partitions
		LSH_HashTable*		m_LSH_hashTables;// Hash table for each partition [L]
	};
}

#endif
