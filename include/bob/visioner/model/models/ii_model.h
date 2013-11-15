/**
 * @file bob/visioner/model/models/ii_model.h
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_VISIONER_II_MODEL_H
#define BOB_VISIONER_II_MODEL_H

#include "bob/visioner/model/model.h"
#include "bob/visioner/vision/integral.h"

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
      Matrix<uint32_t>            m_iimage;       // Integral image
  };

}}

#endif // BOB_VISIONER_II_MODEL_H
