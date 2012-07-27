/**
 * @file visioner/visioner/model/lut.h
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

#ifndef BOB_VISIONER_LUT_H
#define BOB_VISIONER_LUT_H

#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>

#include "visioner/model/ml.h"

namespace bob { namespace visioner {	

  /////////////////////////////////////////////////////////////////////////////////////////	
  // Look-up-table with discrete feature values.
  /////////////////////////////////////////////////////////////////////////////////////////	

  class LUT
  {
    public:

      // Constructor
      LUT(	index_t feature = 0, index_t n_fvalues = 512)
        :	m_feature(feature), m_entries(n_fvalues, 0.0)
      {		

      }

      // Serialize the object
      friend class boost::serialization::access;
      template <typename Archive>
        void serialize(Archive& ar, const unsigned int)
        {
          ar & m_feature;
          ar & m_entries;
        }

      // Scale its output by a given factor
      void scale(scalar_t factor) 
      {
        std::transform(m_entries.begin(), m_entries.end(), m_entries.begin(),
            std::bind1st(std::multiplies<scalar_t>(), factor));
      }

      // Access functions
      template <typename TFeatureValue>
        const scalar_t& operator[](TFeatureValue fv) const { return m_entries[fv]; }
      scalars_t::const_iterator begin() const { return m_entries.begin(); }
      scalars_t::const_iterator end() const { return m_entries.end(); }

      template <typename TFeatureValue>
        scalar_t& operator[](TFeatureValue fv) { return m_entries[fv]; }
      scalars_t::iterator begin() { return m_entries.begin(); }
      scalars_t::iterator end() { return m_entries.end(); }

      const index_t& feature() const { return m_feature; }
      index_t& feature() { return m_feature; }

      index_t n_fvalues() const { return m_entries.size(); }

    private:

      // Attributes
      index_t                 m_feature;
      scalars_t               m_entries;
  };

  typedef std::vector<LUT>	LUTs;
  typedef std::vector<LUTs>       MultiLUTs;

}}

#endif // BOB_VISIONER_LUT_H
