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
		virtual bool		process(const PatternList& candidates);

	private:

		/////////////////////////////////////////////////////////////////

		// Get the closest points to the given one (using LSH table)
		// Returns the number of found points
		int			getClosest(const Pattern& pattern);
		int			getClosest(float cx, float cy, float w, float h);

		// Get the distance between two points (cx, cy, w, h)
		static float		getDistance(const Pattern& pattern, float cx, float cy, float w, float h);
		static float		getDistance(	float cx1, float cy1, float w1, float h1,
							float cx2, float cy2, float w2, float h2);

		// Compute the kernel function
		static float		getKernel(float distance, float bandwidth)
		{
			return (bandwidth - distance) / bandwidth;
		}

		// Fast (linear time) median search (kth-element, more general)
		// http://valis.cs.uiuc.edu/~sariel/research/CG/applets/linear_prog/median.html
		// (std::nth_element function from STL does the same thing!!!)
		float			kth_element(float* data, int size, int nth, float* buffer);

		/////////////////////////////////////////////////////////////////
		// Coarse grid structure to fast retrieve the nearest neighbours to some point

		static const int 	GridSize = 32;
		static const int 	MaxNoClosestPoints = 64;

		struct Grid
		{
			// <Cell>
			struct Cell
			{
				// Constructor
				Cell();

				// Destructor
				~Cell();

				// Add a new index
				void	add(int index_pattern);

				// Resize the indexes already stored to accomodate new ones
				void	resize();

				// Attributes
				int*	m_indexes;
				int	m_size;
				int	m_capacity;
			};

			// Initialize the grid structure with the given points
			void		init(const PatternList& lpatterns);

			// Print the grid structure
			void		print() const;

			// Get the associated value for some point
			Cell&		getCell(const Pattern& pattern);
			Cell&		getCell(float cx, float cy, float w, float h);

			/////////////////////////////////////////////////////////
			// Attributes

			Cell		m_cells[GridSize][GridSize];	// Indexes for each grid cell
			int 		m_x, m_y;			// Grid to-left coordinates
			int		m_dx, m_dy;			// Grid cell size
			float		m_inv_dx, m_inv_dy;
		};

		// Add a cell (with SW indexes) to the buffer of the closest points
		int			addClosest(const Grid::Cell& cell, int isize);

		/////////////////////////////////////////////////////////////////
		// Attributes

		// Grid structure for fast retrieval of the closest points
		Grid			m_grid;

		// Buffer to copy the indexes of the closest points
		int 			m_iclosest[MaxNoClosestPoints];
	};
}

#endif
