/**
 * @file visioner/src/param.cc
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

#include "core/logging.h"

#include "visioner/model/param.h"
#include "visioner/model/mdecoder.h"

namespace bob { namespace visioner {

  // Constructor
  param_t::param_t(
      index_t rows,
      index_t cols,
      const string_t& loss,
      scalar_t loss_parameter,
      const string_t& optimization_type,
      const string_t& training_model,
      index_t num_of_bootstraps,
      const string_t& feature_type,
      const string_t& feature_sharing,
      index_t feature_projections,
      scalar_t min_gt_overlap,
      index_t sliding_windows,
      const string_t& subwindow_labelling):
    m_rows(rows), 
    m_cols(cols), 
    m_seed(0), //hard-coded default
    m_labels(),
    m_loss(loss),
    m_loss_param(loss_parameter),
    m_optimization(optimization_type),
    m_trainer(training_model),
    m_rounds(1024), //hard-coded default
    m_bootstraps(num_of_bootstraps), 
    m_train_data(),
    m_valid_data(),
    m_train_samples(4096), //hard-coded default
    m_valid_samples(1024), //hard-coded default
    m_feature(feature_type),
    m_sharing(feature_sharing),
    m_projections(feature_projections),
    m_min_gt_overlap(min_gt_overlap),
    m_ds(sliding_windows),
    m_tagger(subwindow_labelling)
  {
  }

  // Return the index of the given object label (negative if not found)
  int param_t::find(const string_t& label) const
  {
    strings_t::const_iterator it = std::find(m_labels.begin(), m_labels.end(), label);
    return it == m_labels.end() ? -1 : (it - m_labels.begin());
  }

  // Compute the range of top-left sub-window coordinates for a given image
  int param_t::min_row(index_t, index_t) const
  {
    return 0; 
  }
  int param_t::max_row(index_t image_rows, index_t) const
  {
    return (int)image_rows - (int)m_rows + 1;
  }
  int param_t::min_col(index_t, index_t) const
  {
    return 0; 
  }
  int param_t::max_col(index_t, index_t image_cols) const
  {
    return (int)image_cols - (int)m_cols + 1;
  }

  // Command line processing
  template <typename T>
    void decode_var(const boost::program_options::options_description& po_desc,
        boost::program_options::variables_map& po_vm,				
        const char* var_name, T& var)
    {
      if (!po_vm.count(var_name))
      {
        bob::core::error << po_desc << std::endl;
        exit(EXIT_FAILURE);
      }

      var = po_vm[var_name].as<T>();
    }			    

  template <typename T>
    void decode_vars(const boost::program_options::options_description& po_desc,
        boost::program_options::variables_map& po_vm,				
        const char* var_name, std::vector<T>& var)
    {
      if (!po_vm.count(var_name))
      {
        bob::core::error << po_desc << std::endl;
        exit(EXIT_FAILURE);
      }

      var = split2values<T>(po_vm[var_name].as<string_t>(), ":");;
    }

  template <typename TInt>
    void decode_values_bundle(
        const boost::program_options::options_description& po_desc,
        boost::program_options::variables_map& po_vm,
        const char* var_name, std::vector<TInt>& values)
    {
      if (!po_vm.count(var_name))
      {
        bob::core::error << po_desc << std::endl;
        exit(EXIT_FAILURE);
      }

      values.clear();

      // Split to tokens ...
      const strings_t tokens = split2values<string_t>(po_vm[var_name].as<string_t>(), ":");
      for (strings_t::const_iterator it = tokens.begin(); it != tokens.end(); it ++)
      {
        const strings_t rtokens = split2values<string_t>(*it, "-");

        // Single value
        if (rtokens.size() == 1)
        {
          values.push_back(boost::lexical_cast<TInt>(rtokens[0]));
        }

        // Value interval
        else if (rtokens.size() == 2)
        {
          const index_t rmin = boost::lexical_cast<TInt>(rtokens[0]);
          const index_t rmax = boost::lexical_cast<TInt>(rtokens[1]);
          for (index_t r = rmin; r <= rmax; r ++)
          {
            values.push_back(r);
          }
        }

        // Error
        else
        {
          bob::core::error << po_desc << std::endl;
          exit(EXIT_FAILURE);
        }
      }

      // Error
      if (values.empty())
      {
        bob::core::error << po_desc << std::endl;
        exit(EXIT_FAILURE);
      }
    }

  bool param_t::decode(	const boost::program_options::options_description& po_desc,
      boost::program_options::variables_map& po_vm)
  {
    decode_var(po_desc, po_vm, "model_rows", m_rows);
    decode_var(po_desc, po_vm, "model_cols", m_cols);		
    decode_var(po_desc, po_vm, "model_seed", m_seed);	
    decode_vars(po_desc, po_vm, "model_labels", m_labels);
    decode_var(po_desc, po_vm, "model_loss", m_loss);
    decode_var(po_desc, po_vm, "model_loss_param", m_loss_param);
    decode_var(po_desc, po_vm, "model_optimization", m_optimization);                
    decode_var(po_desc, po_vm, "model_trainer", m_trainer);
    decode_var(po_desc, po_vm, "model_rounds", m_rounds);
    decode_var(po_desc, po_vm, "model_bootstraps", m_bootstraps);
    decode_var(po_desc, po_vm, "model_train_data", m_train_data);
    decode_var(po_desc, po_vm, "model_valid_data", m_valid_data);
    decode_var(po_desc, po_vm, "model_train_samples", m_train_samples);
    decode_var(po_desc, po_vm, "model_valid_samples", m_valid_samples);
    decode_var(po_desc, po_vm, "model_feature", m_feature);
    decode_var(po_desc, po_vm, "model_sharing", m_sharing);
    decode_var(po_desc, po_vm, "model_projections", m_projections);
    decode_var(po_desc, po_vm, "model_min_gt_overlap", m_min_gt_overlap);
    decode_var(po_desc, po_vm, "model_ds", m_ds);
    decode_var(po_desc, po_vm, "model_tagger", m_tagger);

    return true;
  }

  void param_t::add_options(boost::program_options::options_description& po_desc) const
  {
    po_desc.add_options()

      ("model_rows", boost::program_options::value<index_t>()->default_value(m_rows),
       "model: #rows in pixels")		
      ("model_cols", boost::program_options::value<index_t>()->default_value(m_cols),
       "model: #columns in pixels")
      ("model_seed", boost::program_options::value<index_t>()->default_value(m_seed),
       "model: sampling seed") 
      ("model_labels", boost::program_options::value<string_t>()->default_value(""),
       "model: object types, poses or IDs or keypoint IDs of interest (e.g. car:dog:cat)")
      ("model_loss", boost::program_options::value<string_t>()->default_value(m_loss),
       (string_t("model: loss function (") + available_losses() + ")").c_str())
      ("model_loss_param", boost::program_options::value<scalar_t>()->default_value(m_loss_param),
       "model: loss parameter")			
      ("model_optimization", boost::program_options::value<string_t>()->default_value(m_optimization),
       (string_t("model: optimization type (") + available_optimizations() + ")").c_str())
      ("model_trainer", boost::program_options::value<string_t>()->default_value(m_trainer),
       (string_t("model: trainer (") + available_trainers() + ")").c_str())
      ("model_rounds", boost::program_options::value<index_t>()->default_value(m_rounds),
       "model: maximum number of boosting rounds")	
      ("model_bootstraps", boost::program_options::value<index_t>()->default_value(m_bootstraps),
       "model: number of bootstrapping steps")
      ("model_train_data", boost::program_options::value<string_t>()->default_value(m_train_data),
       "model: training data (list files, e.g. list1:list2:...)")
      ("model_valid_data", boost::program_options::value<string_t>()->default_value(m_valid_data),
       "model: validation data (list files, e.g. list1:list2:...)")
      ("model_train_samples", boost::program_options::value<index_t>()->default_value(m_train_samples),
       "model: number of training samples")
      ("model_valid_samples", boost::program_options::value<index_t>()->default_value(m_valid_samples),
       "model: number of validation samples")
      ("model_feature", boost::program_options::value<string_t>()->default_value(m_feature),
       (string_t("model: feature type (") + available_models() + ")").c_str())
      ("model_sharing", boost::program_options::value<string_t>()->default_value(m_sharing),
       (string_t("model: feature sharing type (") + available_sharings() + ")").c_str())
      ("model_projections", boost::program_options::value<index_t>()->default_value(m_projections),
       "model: number of feature projection steps")
      ("model_min_gt_overlap", boost::program_options::value<scalar_t>()->default_value(m_min_gt_overlap),
       "model: minimum overlap of the positive sample with the ground truth")		
      ("model_ds", boost::program_options::value<index_t>()->default_value(m_ds),
       "model: scale variation to generate training samples")
      ("model_tagger", boost::program_options::value<string_t>()->default_value(m_tagger),
       (string_t("model: sample tagger type (") + available_taggers() + ")").c_str());
  }	

}}
