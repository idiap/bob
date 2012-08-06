/**
 * @file visioner/visioner/model/model.h
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

#ifndef BOB_VISIONER_MODEL_H
#define BOB_VISIONER_MODEL_H

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include "visioner/model/lut.h"
#include "visioner/model/param.h"
#include "visioner/model/ipyramid.h"

namespace bob { namespace visioner {	

  /**
   * Multivariate model as a linear combination of std::vector<LUT>.
   * NB: The ::preprocess() must be called before ::get() and ::score()
   * functions.
   */
  class Model : public Parametrizable {

    public: //api

      // Constructor
      Model(const param_t& param = param_t());

      // Destructor
      virtual ~Model() {}

      // Clone the object
      virtual boost::shared_ptr<Model> clone() const = 0;

      // Reset to new parameters
      virtual void reset(const param_t& param);

      // Reset to new std::vector<LUT> (lut.size() == model.n_outputs()!)
      virtual bool set(const std::vector<std::vector<LUT> >& mluts);

      // Project the selected features to a higher resolution
      virtual bool project() = 0;

      // Save/load to/from file
      bool save(const std::string& filename) const;
      bool load(const std::string& filename);
      static bool load(const std::string& filename, boost::shared_ptr<Model>& model);

      // Preprocess the current image
      virtual void preprocess(const ipscale_t& ipscale) = 0;

      // Compute the model score at the (x, y) position for the output <o>
      double score(uint64_t o, int x, int y) const;
      double score(uint64_t o, uint64_t rbegin, uint64_t rend, int x, int y) const;

      // Compute the value of the feature <f> at the (x, y) position
      virtual uint64_t get(uint64_t f, int x, int y) const = 0;

      // Access functions
      virtual uint64_t n_features() const = 0;
      virtual uint64_t n_fvalues() const = 0;
      uint64_t n_outputs() const { return m_mluts.size(); }
      uint64_t n_luts(uint64_t o) const { return m_mluts[o].size(); }
      const std::vector<std::vector<LUT> >& luts() const { return m_mluts; }
      virtual std::vector<uint64_t> features() const;

      // Describe a feature
      virtual std::string describe(uint64_t f) const = 0;   

      // Save/load specific (feature) information
      virtual void save(boost::archive::text_oarchive& oa) const = 0;
      virtual void save(boost::archive::binary_oarchive& oa) const = 0;
      virtual void load(boost::archive::text_iarchive& ia) = 0;
      virtual void load(boost::archive::binary_iarchive& ia) = 0;

    private: //representation

      // Attributes
      std::vector<std::vector<LUT> >               m_mluts;        // Multivariate std::vector<LUT>
  };

}}

#endif // BOB_VISIONER_MODEL_H
