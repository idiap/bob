/**
 * @file visioner/visioner/model/sampler.h
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

#ifndef BOB_VISIONER_SAMPLER_H
#define BOB_VISIONER_SAMPLER_H

#include <map>
#include <boost/random.hpp>
#include <boost/random/uniform_real.hpp>

#include "visioner/model/model.h"
#include "visioner/model/loss.h"
#include "visioner/model/dataset.h"
#include "visioner/model/tagger.h"

namespace bob { namespace visioner {

  /////////////////////////////////////////////////////////////////////////////////////////
  // Object used for sampling uniformly,
  //	such that the same number of samples are obtained for distinct target values.
  /////////////////////////////////////////////////////////////////////////////////////////

  class Sampler
  {
    public:

      enum SamplerType
      {
        TrainSampler,
        ValidSampler
      };

      // Constructor
      Sampler(const param_t& param, SamplerType type);

      // Sample the given number of samples (uniformly)
      void sample(uint64_t n_samples, std::vector<uint64_t>& samples) const;

      // Sample the given number of samples (error based)
      void sample(uint64_t n_samples, const Model& model, std::vector<uint64_t>& samples) const;

      // Map the selected samples to a dataset
      void map(const std::vector<uint64_t>& samples, const Model& model, DataSet& data) const;

      // Access functions
      uint64_t n_images() const { return m_ipscales.size(); }
      uint64_t n_samples() const { return m_n_samples; }
      uint64_t n_outputs() const { return m_n_outputs; }
      uint64_t n_types() const { return m_n_types; }
      const std::vector<ipscale_t>& images() const { return m_ipscales; }

      // Resets to a list of images and (matching) ground truth files
      void load(const std::vector<std::string>& ifiles, const std::vector<std::string>& gfiles);

      inline SamplerType getType() const { return m_type; }

    private:

      // Cumulate a set of statistics indexed by an integer
      template <typename TData>
        std::vector<TData> stat_cumulate(const std::vector<std::vector<TData> >& stats) const
        {
          std::vector<TData> result(n_types(), (TData)0);
          for (uint64_t i = 0; i < stats.size(); i ++)
          {
            const std::vector<TData>& stat = stats[i];
            for (uint64_t iti = 0; iti < n_types(); iti ++)
            {
              result[iti] += stat[iti];
            }
          }

          return result;
        }

      // Initialize statistics indexed by an integer
      template <typename TData>
        void stat_init(std::vector<TData>& stats) const
        {
          stats.resize(n_types());
          std::fill(stats.begin(), stats.end(), (TData)0);
        }

      // Type to string
      std::string type2str() const
      {
        return m_type == TrainSampler ? "train" : "valid";
      }

      // Map the given sample to image
      uint64_t sample2image(uint64_t s) const;

      // Compute the error of the given sample
      double error(uint64_t x, uint64_t y, const std::vector<double>& targets, const Model& model, std::vector<double>& scores) const;

      // Cost-based sampling
      void sample(uint64_t s, double cost, boost::mt19937& gen, boost::uniform_01<>& die, std::vector<uint64_t>& samples) const;

    private:

      // Reset to a set of listfiles
      void load(const std::vector<std::string>& listfiles);

      // Uniform sampling thread
      void th_usample(uint64_t ith, std::pair<uint64_t, uint64_t> srange, std::vector<uint64_t>& samples) const;

      // Error-based sampling thread
      void th_esample(uint64_t ith, std::pair<uint64_t, uint64_t> srange, const Model& model, std::vector<uint64_t>& samples) const;

      // Evaluation thread
      void th_errors(std::pair<uint64_t, uint64_t> srange, const Model& model, std::vector<double>& terrors) const;

      // Mapping thread (samples to dataset)
      void th_map(std::pair<uint64_t, uint64_t> srange, const std::vector<uint64_t>& samples, const Model& model, std::vector<uint64_t>& types, DataSet& data) const;

    private:

      // Attributes
      param_t	m_param; // Model parameters
      SamplerType	m_type; // Training or Validation mode

      boost::shared_ptr<Tagger> m_tagger; // Sample labelling
      boost::shared_ptr<Loss> m_loss; // Loss

      uint64_t m_n_outputs; //
      uint64_t m_n_samples; //
      uint64_t m_n_types; // #distinct target types

      std::vector<ipscale_t> m_ipscales; // Input: image + annotations @ all scales
      std::vector<uint64_t> m_ipsbegins; // Sample interval [begin, end)
      std::vector<uint64_t> m_ipsends;   // for each scaled image

      std::vector<uint64_t> m_tcounts; // #times / distinct target type
      mutable std::vector<double> m_sprobs; // base sampling probability / distinct target type
      mutable std::vector<boost::mt19937> m_rgens; // Random number generators (per thread)
  };

}}

#endif // BOB_VISIONER_SAMPLER_H
