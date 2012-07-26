#ifndef BOB_VISIONER_II_MODEL_H
#define BOB_VISIONER_II_MODEL_H

#include "visioner/model/model.h"
#include "visioner/vision/integral.h"

namespace bob { namespace visioner {

  /////////////////////////////////////////////////////////////////////////////////////////
  // Model that uses integral images.
  /////////////////////////////////////////////////////////////////////////////////////////
  class IIModel : public Model
  {
    public:

      // Constructor
      IIModel(const param_t& param = param_t())
        :	Model(param)
      {
      }

      // Destructor
      virtual ~IIModel() {}

      // Preprocess the current image
      void preprocess(const ipscale_t& ipscale)
      {
        integral(ipscale.m_image, m_iimage);
      }

    protected:    

      // Attributes
      igreyimage_t            m_iimage;       // Integral image
  };

}}

#endif // BOB_VISIONER_II_MODEL_H
