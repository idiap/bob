/**
 * @file bob/visioner/model/models/model_pool.h
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

#ifndef BOB_VISIONER_MODEL_POOL_H
#define BOB_VISIONER_MODEL_POOL_H

#include "bob/visioner/model/model.h"

namespace bob { namespace visioner {

  /////////////////////////////////////////////////////////////////////////////////////////
  // Collection of two models that have the same number of feature values.
  /////////////////////////////////////////////////////////////////////////////////////////

  template <typename TModel1, typename TModel2>
    class ModelPool : public Model
  {	
    public:

      // Constructor
      ModelPool(const param_t& param = param_t())
        :	Model(param), 
        m_fpool1(param), m_fpool2(param)
    {
      reset(param);
    }

      // Destructor
      virtual ~ModelPool() {}

      // Clone the object
      virtual boost::shared_ptr<Model> clone() const { return boost::shared_ptr<Model>(new ModelPool(*this)); }

      // Reset to new parameters
      virtual void reset(const param_t& param)
      {
        Model::reset(param);
        m_fpool1.reset(param);
        m_fpool2.reset(param);
      } 

      // Reset to new std::vector<LUT> (lut.size() == model.n_outputs()!)
      virtual bool set(const std::vector<std::vector<LUT> >& mluts)
      {
        return _set(mluts);
      }

      // Project the selected features to a higher resolution
      virtual bool project() 
      {
        return m_fpool1.project() && m_fpool2.project();
      }

      // Preprocess the current image
      virtual void preprocess(const ipscale_t& ipscale)
      {
        m_fpool1.preprocess(ipscale);
        m_fpool2.preprocess(ipscale);
      }

      // Compute the value of the feature <f> at the (x, y) position
      virtual uint64_t get(uint64_t f, int x, int y) const
      {
        if (f < n_features1())
        {
          return m_fpool1.get(f, x, y);
        }      
        else
        {
          return m_fpool2.get(f - n_features1(), x, y);
        }
      }

      // Access functions
      virtual uint64_t n_fvalues() const { return m_fpool1.n_fvalues(); }
      virtual uint64_t n_features() const { return n_features1() + n_features2(); }
      uint64_t n_features1() const { return m_fpool1.n_features(); }
      uint64_t n_features2() const { return m_fpool2.n_features(); }

      // Describe a feature
      virtual std::string describe(uint64_t f) const
      {
        if (f < n_features1())
        {
          return m_fpool1.describe(f);
        }                     
        else
        {
          return m_fpool2.describe(f - n_features1());
        }
      }

      // Save/load specific (feature) information
      virtual void save(boost::archive::text_oarchive& oa) const
      {
        m_fpool1.save(oa);
        m_fpool2.save(oa);
      }
      virtual void save(boost::archive::binary_oarchive& oa) const
      {
        m_fpool1.save(oa);
        m_fpool2.save(oa);
      }
      virtual void load(boost::archive::text_iarchive& ia)
      {
        m_fpool1.load(ia);
        m_fpool2.load(ia);
      }
      virtual void load(boost::archive::binary_iarchive& ia)
      {
        m_fpool1.load(ia);
        m_fpool2.load(ia);
      }

    private:

      // Reset to new std::vector<LUT> (lut.size() == model.n_outputs()!)
      bool _set(const std::vector<std::vector<LUT> >& mluts)
      {
        if (Model::set(mluts) == false)
        {
          return false;
        }

        // Split the std::vector<LUT> to the associated feature pool ...
        std::vector<std::vector<LUT> > mluts1(n_outputs()), mluts2(n_outputs());
        for (uint64_t o = 0; o < n_outputs(); o ++)
        {
          std::vector<LUT>& luts1 = mluts1[o];
          std::vector<LUT>& luts2 = mluts2[o];

          for (uint64_t r = 0; r < n_luts(o); r ++)
          {
            LUT lut = luts()[o][r];                                        
            if (lut.feature() < n_features1())
            {
              luts1.push_back(lut);
            }
            else
            {
              lut.feature() -= n_features1();
              luts2.push_back(lut);
            }
          }                          
        }

        return  m_fpool1.set(mluts1) && 
          m_fpool2.set(mluts2);
      }

    private:

      // Attributes
      TModel1    m_fpool1;     
      TModel2    m_fpool2;
  };

}}

#endif // BOB_VISIONER_MODEL_POOL_H
