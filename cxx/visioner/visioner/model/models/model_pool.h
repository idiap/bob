#ifndef BOB_VISIONER_MODEL_POOL_H
#define BOB_VISIONER_MODEL_POOL_H

#include "visioner/model/model.h"

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
      virtual rmodel_t clone() const { return rmodel_t(new ModelPool(*this)); }

      // Reset to new parameters
      virtual void reset(const param_t& param)
      {
        Model::reset(param);
        m_fpool1.reset(param);
        m_fpool2.reset(param);
      } 

      // Reset to new LUTs (lut.size() == model.n_outputs()!)
      virtual bool set(const MultiLUTs& mluts)
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
      virtual index_t get(index_t f, int x, int y) const
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
      virtual index_t n_fvalues() const { return m_fpool1.n_fvalues(); }
      virtual index_t n_features() const { return n_features1() + n_features2(); }
      index_t n_features1() const { return m_fpool1.n_features(); }
      index_t n_features2() const { return m_fpool2.n_features(); }

      // Describe a feature
      virtual string_t describe(index_t f) const
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

      // Reset to new LUTs (lut.size() == model.n_outputs()!)
      bool _set(const MultiLUTs& mluts)
      {
        if (Model::set(mluts) == false)
        {
          return false;
        }

        // Split the LUTs to the associated feature pool ...
        MultiLUTs mluts1(n_outputs()), mluts2(n_outputs());
        for (index_t o = 0; o < n_outputs(); o ++)
        {
          LUTs& luts1 = mluts1[o];
          LUTs& luts2 = mluts2[o];

          for (index_t r = 0; r < n_luts(o); r ++)
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
