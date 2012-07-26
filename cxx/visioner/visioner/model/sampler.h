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
      void sample(index_t n_samples, indices_t& samples) const;        

      // Sample the given number of samples (error based)
      void sample(index_t n_samples, const Model& model, indices_t& samples) const;

      // Map the selected samples to a dataset
      void map(const indices_t& samples, const Model& model, DataSet& data) const;

      // Access functions
      index_t n_images() const { return m_ipscales.size(); }
      index_t n_samples() const { return m_n_samples; }
      index_t n_outputs() const { return m_n_outputs; }
      index_t n_types() const { return m_n_types; }
      const ipscales_t& images() const { return m_ipscales; }

    private:

      // <target type, #times it appears>
      typedef indices_t               	tcounts_t;
      typedef tcounts_t::const_iterator	tcounts_const_it;

      // <target type, sampling power>
      typedef scalars_t               	tpowers_t;
      typedef tpowers_t::const_iterator	tpowers_const_it; 

      // Random number generator in [0, 1)
      typedef boost::mt19937                  rgen_t;               
      typedef std::vector<rgen_t>             rgens_t;

      //boost::random::uniform_real_distribution<> die(0.0, 1.0);
      typedef boost::uniform_01<>             rdie_t;

      // Cumulate a set of statistics indexed by an integer
      template <typename TData>
        std::vector<TData> stat_cumulate(const std::vector<std::vector<TData> >& stats) const
        {
          typedef std::vector<TData>              stat_t;
          typedef typename stat_t::const_iterator stat_cit_t;

          stat_t result(n_types(), (TData)0);
          for (index_t i = 0; i < stats.size(); i ++)
          {
            const stat_t& stat = stats[i];
            for (index_t iti = 0; iti < n_types(); iti ++)
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
          typedef std::vector<TData>              stat_t;

          stats.resize(n_types());
          std::fill(stats.begin(), stats.end(), (TData)0);
        }

      // Type to string
      string_t type2str() const
      {
        return m_type == TrainSampler ? "train" : "valid";
      }

      // Map the given sample to image
      index_t sample2image(index_t s) const;

      // Compute the error of the given sample
      scalar_t error(
          index_t x, index_t y, 
          const scalars_t& targets, const Model& model, scalars_t& scores) const;

      // Cost-based sampling
      void sample(
          index_t s, scalar_t cost, rgen_t& gen, rdie_t& die, indices_t& samples) const;

    private:

      // Reset to a set of listfiles
      void load(const strings_t& listfiles);
      void load(const strings_t& ifiles, const strings_t& gfiles);

      // Uniform sampling thread
      void th_usample(
          index_t ith, index_pair_t srange, 
          indices_t& samples) const;

      // Error-based sampling thread
      void th_esample(
          index_t ith, index_pair_t srange, const Model& model, 
          indices_t& samples) const;

      // Evaluation thread
      void th_errors(
          index_pair_t srange, const Model& model, 
          tpowers_t& terrors) const; 

      // Mapping thread (samples to dataset)
      void th_map(
          index_pair_t srange, const indices_t& samples, const Model& model, 
          indices_t& types, DataSet& data) const;

    private:

      // Attributes
      param_t			m_param;	// Model parameters	
      SamplerType		m_type;		// Training or Validation mode

      rtagger_t               m_tagger;       // Sample labelling
      rloss_t                 m_loss;         // Loss

      index_t                 m_n_outputs;    //
      index_t                 m_n_samples;    //  
      index_t                 m_n_types;      // #distinct target types

      ipscales_t              m_ipscales;	// Input: image + annotations @ all scales
      indices_t               m_ipsbegins;    // Sample interval [begin, end)
      indices_t               m_ipsends;      //      for each scaled image

      tcounts_t               m_tcounts;      // #times / distinct target type
      mutable tpowers_t       m_sprobs;       // base sampling probability / distinct target type

      mutable rgens_t         m_rgens;        // Random number generators (per thread)
  };

}}

#endif // BOB_VISIONER_SAMPLER_H
