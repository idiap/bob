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
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include "bob/visioner/model/ml.h"

namespace bob { namespace visioner {

  /**
   * Parameters:
   *	- loss, trainer, tagger
   *	- sliding-windows sampling
   *	- features
   */
  struct param_t {

    public:

      /**
       * Default Constructor
       *
       * @note Other parameters are hard-coded within the constructor
       * initialization. Have a look there to find all default parameters.
       */
      param_t(
          uint64_t rows=24,
          uint64_t cols=20,
          const std::string& loss="diag_log",
          double loss_parameter=0.0,
          const std::string& optimization_type="ept",
          const std::string& training_model="gboost",
          uint64_t num_of_bootstraps=3,
          const std::string& feature_type="elbp",
          const std::string& feature_sharing="shared",
          uint64_t feature_projections=0,
          double min_gt_overlap=0.8,
          uint64_t sliding_windows=2,
          const std::string& subwindow_labelling="object_type"
          );

      // Compute the range of top-left sub-window coordinates for a given image
      int min_row(uint64_t image_rows, uint64_t image_cols) const;
      int max_row(uint64_t image_rows, uint64_t image_cols) const;
      int min_col(uint64_t image_rows, uint64_t image_cols) const;
      int max_col(uint64_t image_rows, uint64_t image_cols) const;

      // Return the index of the given label (negative if not found)
      int find(const std::string& label) const;

      // Command line processing
      void add_options(boost::program_options::options_description& po_desc) const;
      bool decode(	const boost::program_options::options_description& po_desc,
          boost::program_options::variables_map& po_vm);

      // Serialize the object
      friend class boost::serialization::access;
      template <typename Archive>
        void serialize(Archive& ar, const unsigned int)
        {
          std::string dummy_to_be_removed("dense");

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

    public: //representation

      // Attributes
      uint64_t		m_rows, m_cols;		// Model size
      uint64_t         m_seed;                 // Random seed

      std::vector<std::string>	m_labels;		// Object types/poses/IDs or Keypoint IDs of interest

      std::string	m_loss;			// Loss
      double        m_loss_param;           // Loss parameter
      std::string        m_optimization;         // Optimization type (expectation vs. variational)

      std::string	m_trainer;		// Training model

      uint64_t		m_rounds;		// Maximum boosting rounds
      uint64_t         m_bootstraps;           // Number of bootstrapping steps

      std::string	m_train_data;		// Training data
      std::string	m_valid_data;		// Validation data
      uint64_t         m_train_samples;        // #training samples
      uint64_t         m_valid_samples;        // #validation samples

      std::string	m_feature;		// Feature type
      std::string        m_sharing;              // Feature sharing
      uint64_t         m_projections;          // Coarse-to-fine feature projection

      double	m_min_gt_overlap;	// Minimum overlapping with ground truth for positive samples

      uint64_t		m_ds;			// Sliding windows
      std::string	m_tagger;		// Labelling sub-windows		
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
