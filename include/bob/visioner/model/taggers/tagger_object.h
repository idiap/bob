/**
 * @file bob/visioner/model/taggers/tagger_object.h
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_VISIONER_TAGGER_OBJECT_ID_H
#define BOB_VISIONER_TAGGER_OBJECT_ID_H

#include "bob/visioner/model/tagger.h"

namespace bob { namespace visioner {

  class ObjectTagger : public Tagger
  {
    public:

      enum Type
      {
        IDTagger,
        TypeTagger,
        PoseTagger
      };

      // Constructor
      ObjectTagger(Type type, const param_t& param = param_t());

      // Destructor
      virtual ~ObjectTagger() {}

      // Clone the object
      virtual boost::shared_ptr<Tagger> clone() const { return boost::shared_ptr<Tagger>(new ObjectTagger(*this)); }

      // Number of outputs
      virtual uint64_t n_outputs() const;

      // Number of types
      virtual uint64_t n_types() const { return m_param.m_labels.size() + 1; }

      // Label a sub-window
      virtual bool check(const ipscale_t& ipscale, int x, int y, 
          std::vector<double>& targets, uint64_t& type) const;

      // Return the label (ID or type) of a given object
      const std::string& label(const Object& object) const
      {
        return m_type == IDTagger ? object.id() : 
          (m_type == PoseTagger ? object.pose() : object.type());
      }

      // Return the label index of a given object
      int find(const Object& object) const
      {
        return m_param.find(label(object));
      }

    private:

      // Attributes
      Type            m_type;
  };

}}

#endif // BOB_VISIONER_TAGGER_OBJECT_ID_H
