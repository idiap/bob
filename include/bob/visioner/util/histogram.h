/**
 * @file bob/visioner/util/histogram.h
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
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

#ifndef BOB_VISIONER_HISTOGRAM_H
#define BOB_VISIONER_HISTOGRAM_H

#include "bob/visioner/util/util.h"

namespace bob { namespace visioner {

	/////////////////////////////////////////////////////////////////////////////////////////
	// Histogram
	/////////////////////////////////////////////////////////////////////////////////////////
	class Histogram
	{
	public:
		
		// Constructor
		Histogram(uint64_t n_bins = 20, double min_value = 0.0, double max_value = 1.0);
				
		// Reset to the new bins
		void reset(uint64_t n_bins, double min_value, double max_value);
		
		// Delete all stored values
		void clear() { std::fill(m_bins.begin(), m_bins.end(), 0.0); }
                
                // Add a new value
		void add(double value);
		
		template <typename TIterator>
		void add(TIterator begin, TIterator end)
		{
			for ( ; begin != end; ++ begin)
				add(*begin);
		}
		
		// Save to file
		bool save(const std::string& path) const;
		
		// Compute and normalize the cumulated histogram
		void cumulate();
                
                // Normalize the histogram
                void norm();
		
		// Access functions
		uint64_t n_bins() const { return m_n_bins; }
		double delta() const { return m_delta; }
		double bin_value(uint64_t bin) const { return m_min + bin * m_delta; }
		const std::vector<double>& bins() const { return m_bins; }
		
	private:
		
		// Attributes
		uint64_t		m_n_bins;
		double	m_min, m_delta, m_max, m_inv_delta;
		std::vector<double>	m_bins;
	};

}}

#endif // BOB_VISIONER_HISTOGRAM_H
