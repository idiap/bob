/**
 * @file visioner/visioner/model/param.h
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

#ifndef BOB_VISIONER_PARAM_H
#define BOB_VISIONER_PARAM_H

#include <boost/program_options.hpp>

#include "visioner/model/ml.h"

namespace bob { namespace visioner {

  //////////////////////////////////////////////////////////////////////////////////////
  // Parameters:
  //	- loss, trainer, tagger
  //	- sliding-windows sampling
  //	- features
  //////////////////////////////////////////////////////////////////////////////////////

  struct param_t
  {
    public:

      // Constructor
      param_t();

      // Compute the range of top-left sub-window coordinates for a given image
      int min_row(index_t image_rows, index_t image_cols) const;
      int max_row(index_t image_rows, index_t image_cols) const;
      int min_col(index_t image_rows, index_t image_cols) const;
      int max_col(index_t image_rows, index_t image_cols) const;

      // Return the index of the given label (negative if not found)
      int find(const string_t& label) const;

      // Command line processing
      void add_options(boost::program_options::options_description& po_desc) const;
      bool decode(	const boost::program_options::options_description& po_desc,
          boost::program_options::variables_map& po_vm);

      // Serialize the object
      friend class boost::serialization::access;
      template <typename Archive>
        void serialize(Archive& ar, const unsigned int)
        {
          string_t dummy_to_be_removed("dense");

          ar & m_rows;
          ar & m_cols;
          ar & m_seed;
          ar & m_labels;
          ar & m_loss;
          ar & m_loss_param;
          ar & m_optimization;
          ar & m_trainer;
          ar & m_rounds;
          ar & m_bootstraps;
          ar & m_train_data;
          ar & m_valid_data;
          ar & m_train_samples;
          ar & m_valid_samples;
          ar & m_feature;
          ar & m_sharing;
          ar & dummy_to_be_removed;
          ar & m_projections;
          ar & m_min_gt_overlap;
          ar & m_ds;
          ar & m_tagger;
        }

    public:

      // Attributes
      index_t		m_rows, m_cols;		// Model size
      index_t         m_seed;                 // Random seed

      strings_t	m_labels;		// Object types/poses/IDs or Keypoint IDs of interest

      string_t	m_loss;			// Loss
      scalar_t        m_loss_param;           // Loss parameter
      string_t        m_optimization;         // Optimization type (expectation vs. variational)

      string_t	m_trainer;		// Training model

      index_t		m_rounds;		// Maximum boosting rounds
      index_t         m_bootstraps;           // Number of bootstrapping steps

      string_t	m_train_data;		// Training data
      string_t	m_valid_data;		// Validation data
      index_t         m_train_samples;        // #training samples
      index_t         m_valid_samples;        // #validation samples

      string_t	m_feature;		// Feature type
      string_t        m_sharing;              // Feature sharing
      index_t         m_projections;          // Coarse-to-fine feature projection

      scalar_t	m_min_gt_overlap;	// Minimum overlapping with ground truth for positive samples

      index_t		m_ds;			// Sliding windows
      string_t	m_tagger;		// Labelling sub-windows		
  };

  //////////////////////////////////////////////////////////////////////////////////////
  // Parametrizable objects depend on the <param_t> values.
  //////////////////////////////////////////////////////////////////////////////////////

  class Parametrizable
  {
    public:

      // Constructor
      Parametrizable(const param_t& param = param_t())	
        :	m_param(param)
      {			
      }

      // Destructor
      virtual ~Parametrizable() {}

      // Reset to new parameters
      virtual void reset(const param_t& param) = 0;

      // Access functions
      const param_t& param() const { return m_param; }

    protected:

      // Attributes
      param_t		m_param;
  };		

}}

#endif // BOB_VISIONER_PARAM_H
