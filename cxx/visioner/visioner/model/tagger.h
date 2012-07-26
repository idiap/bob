#ifndef BOB_VISIONER_TAGGER_H
#define BOB_VISIONER_TAGGER_H

#include "visioner/model/ipyramid.h"
#include "visioner/model/param.h"

namespace bob { namespace visioner {

  class Tagger;
  typedef boost::shared_ptr<Tagger>	rtagger_t;

  /////////////////////////////////////////////////////////////////////////////////////////
  // Sub-window labelling for either classification or regression.
  /////////////////////////////////////////////////////////////////////////////////////////

  class Tagger : public Parametrizable
  {
    public:

      // Constructor
      Tagger(const param_t& param = param_t())
        :	Parametrizable(param)
      {
      }

      // Destructor
      virtual ~Tagger() {}

      // Clone the object
      virtual rtagger_t clone() const = 0;

      // Reset to new parameters
      virtual void reset(const param_t& param) { m_param = param; }

      // Number of outputs 
      virtual index_t n_outputs() const = 0;

      // Number of types
      virtual index_t n_types() const = 0;

      // Label a sub-window
      virtual bool check(const ipscale_t& ipscale, int x, int y, 
          scalars_t& targets, index_t& type) const = 0;
  };

}}

#endif // BOB_VISIONER_TAGGER_H
