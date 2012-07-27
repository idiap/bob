/**
 * @file visioner/visioner/util/histogram.h
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
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

#ifndef BOB_VISIONER_HISTOGRAM_H
#define BOB_VISIONER_HISTOGRAM_H

#include "visioner/util/util.h"

namespace bob { namespace visioner {

	/////////////////////////////////////////////////////////////////////////////////////////
	// Histogram
	/////////////////////////////////////////////////////////////////////////////////////////
	class Histogram
	{
	public:
		
		// Constructor
		Histogram(index_t n_bins = 20, scalar_t min_value = 0.0, scalar_t max_value = 1.0);
				
		// Reset to the new bins
		void reset(index_t n_bins, scalar_t min_value, scalar_t max_value);
		
		// Delete all stored values
		void clear() { std::fill(m_bins.begin(), m_bins.end(), 0.0); }
                
                // Add a new value
		void add(scalar_t value);
		
		template <typename TIterator>
		void add(TIterator begin, TIterator end)
		{
			for ( ; begin != end; ++ begin)
				add(*begin);
		}
		
		// Save to file
		bool save(const string_t& path) const;
		
		// Compute and normalize the cumulated histogram
		void cumulate();
                
                // Normalize the histogram
                void norm();
		
		// Access functions
		index_t n_bins() const { return m_n_bins; }
		scalar_t delta() const { return m_delta; }
		scalar_t bin_value(index_t bin) const { return m_min + bin * m_delta; }
		const scalars_t& bins() const { return m_bins; }
		
	private:
		
		// Attributes
		index_t		m_n_bins;
		scalar_t	m_min, m_delta, m_max, m_inv_delta;
		scalars_t	m_bins;
	};

}}

#endif // BOB_VISIONER_HISTOGRAM_H
