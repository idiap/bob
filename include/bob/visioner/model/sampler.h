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

#include "bob/visioner/model/model.h"
#include "bob/visioner/model/loss.h"
#include "bob/visioner/model/dataset.h"
#include "bob/visioner/model/tagger.h"

namespace bob { namespace visioner {

  /**
   * Object used for sampling uniformly, such that the same number of samples
   * are obtained for distinct target values.
   */
  class Sampler {

    public: //api

      enum SamplerType {
        TrainSampler,
        ValidSampler
      };

      /**
       * Builds a new sampler for either training or validation. Also specify
       * the maximum number of threads this sampler should be able to work with.
       * This parameter will be used to initialize N random generators (one for
       * each thread). A value of zero will initialize a single random number
       * generator.
       */
      Sampler(const param_t& param, SamplerType type, size_t max_threads=0);

      /**
       * Samples, approximately, the given number of samples (uniformly). The
       * sampling and mapping methods can be executed either in multiple
       * threads or single-threaded (the default). Pass the number of threads
       * you would like to have using the 3rd parameter. Zero will make the
       * sampler work in the current thread. One or more will make the sampler
       * spawn a number of threads to execute its job.
       *
       * This sampling method will sample within the available sampling pool
       * (all available samples - remember this is different then the number of
       * input images available as it also accounts for scaled versions of each
       * image) uniformily by dividing the available sampling set in T disjunct
       * subsets (T = number of threads to use) and sampling randomly, with
       * replacement each of the subsets. The subsets are then merged before
       * this method returns.
       *
       * Because of the way it is coded, this method cannot guarantee (even if
       * the seed is set on the constructor), that it responds the same way
       * independently of the number of threads that it uses to run. The reason
       * being that number of threads used to sample will determine the number
       * of sampling subsets and, therefore, how samples will be picked. If you
       * call this method with the same number of threads though, given the
       * same initial conditions (e.g., first time in the program), it should
       * reply in the same manner.
       *
       * Also note that, because of the way this method is implemented, it
       * cannot guarantee that the exact number of samples requested will be
       * returned.
       */
      void sample(uint64_t n_samples, std::vector<uint64_t>& samples, size_t
          threads) const;

      /**
       * Samples, approximately, the given number of samples (error based). The
       * comments for the uniform sample() method above for details.
       */
      void sample(uint64_t n_samples, const Model& model, 
          std::vector<uint64_t>& samples, size_t threads) const;

      /**
       * Maps the selected samples to a dataset. This method is only mapping
       * the selected samples into the provided Dataset. It can also be
       * parallelized if you provide the number of threads. Because this is
       * just a copy operation and no randomness is involved, calling this
       * method with different number of threads yields always to the same
       * results.
       */
      void map(const std::vector<uint64_t>& samples, const Model& model, 
          DataSet& data, size_t threads) const;

      /**
       * The total number of images loaded
       */
      uint64_t n_images() const { return m_ipscales.size(); }

      /**
       * The total number of samples loaded, taking into consideration all
       * possible scales for each image. The number of scales per image is
       * calculated based on the model size (number of rows and columns).
       */
      uint64_t n_samples() const { return m_n_samples; }

      uint64_t n_outputs() const { return m_n_outputs; }

      /**
       * Number of types in the model. This number is attributed by the tagger.
       */
      uint64_t n_types() const { return m_n_types; }

      /**
       * Grabs all images.
       */
      const std::vector<ipscale_t>& images() const { return m_ipscales; }

      /**
       * Resets to a list of images and (matching) ground truth files. Note:
       * This method is buggy and will only work after object initialization.
       */
      void load(const std::vector<std::string>& ifiles, 
          const std::vector<std::string>& gfiles);

      /**
       * Returns the current sampler type.
       */
      inline SamplerType getType() const { return m_type; }

    private:

      /**
       * Variations of the sampling and mapping sub-routines in single and 
       * multi-threaded implementations. Which is used will depend on the 
       * Sampler 'threads' construction parameter.
       */
      void sample_uniformily_single(uint64_t n_samples,
          std::vector<uint64_t>& samples) const;
      void sample_uniformily_multi(uint64_t n_samples,
          std::vector<uint64_t>& samples, size_t threads) const;
      void sample_based_on_error_single(uint64_t n_samples,
          const Model& model, std::vector<uint64_t>& samples) const;
      void sample_based_on_error_multi(uint64_t n_samples,
          const Model& model, std::vector<uint64_t>& samples,
          size_t threads) const;
      void map_single(const std::vector<uint64_t>& samples,
          const Model& model, DataSet& data) const;
      void map_multi(const std::vector<uint64_t>& samples,
          const Model& model, DataSet& data, size_t threads) const;

      // Type to string
      std::string type2str() const {
        return m_type == TrainSampler ? "training" : "validation";
      }

      // Map the given sample to image
      uint64_t sample2image(uint64_t s) const;

      // Compute the error of the given sample
      double error(uint64_t x, uint64_t y, const std::vector<double>& targets, 
          const Model& model, std::vector<double>& scores) const;

      // Cost-based sampling
      void sample(uint64_t s, double cost, boost::mt19937& gen, 
          boost::uniform_01<>& die, std::vector<uint64_t>& samples) const;

    private:

      // Reset to a set of listfiles
      void load(const std::vector<std::string>& listfiles);

      /**
       * Uniform sampling worker thread
       */
      void th_usample(uint64_t ith, std::pair<uint64_t, uint64_t> srange, 
          std::vector<uint64_t>& samples) const;

      /**
       * Error-based sampling worker thread
       */
      void th_esample(uint64_t ith, std::pair<uint64_t, uint64_t> srange, 
          const Model& model, std::vector<uint64_t>& samples) const;

      /**
       * Evaluation thread
       */
      void th_errors(std::pair<uint64_t, uint64_t> srange, const Model& model,
          std::vector<double>& terrors) const;

      /**
       * Mapping thread (samples to dataset)
       */
      void th_map(std::pair<uint64_t, uint64_t> srange, 
          const std::vector<uint64_t>& samples, const Model& model,
          std::vector<uint64_t>& types, DataSet& data) const;

    private: //representation

      param_t	m_param; ///< Model parameters
      SamplerType	m_type; ///< Training or Validation mode

      boost::shared_ptr<Tagger> m_tagger; ///< Sample labelling
      boost::shared_ptr<Loss> m_loss; ///< Loss

      uint64_t m_n_outputs; //
      uint64_t m_n_samples; //
      uint64_t m_n_types; ///< # of distinct target types

      std::vector<ipscale_t> m_ipscales; ///< Input: image + annotations @ all scales
      std::vector<uint64_t> m_ipsbegins; ///< Sample interval [begin, end)
      std::vector<uint64_t> m_ipsends;   ///< for each scaled image

      std::vector<uint64_t> m_tcounts; ///< # of times / distinct target type
      mutable std::vector<double> m_sprobs; ///< base sampling probability / distinct target type
      mutable std::vector<boost::mt19937> m_rgens; ///< Random number generators (per thread)

  };

}}

#endif // BOB_VISIONER_SAMPLER_H
