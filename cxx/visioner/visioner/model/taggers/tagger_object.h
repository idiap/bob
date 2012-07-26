#ifndef BOB_VISIONER_TAGGER_OBJECT_ID_H
#define BOB_VISIONER_TAGGER_OBJECT_ID_H

#include "visioner/model/tagger.h"

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
      virtual rtagger_t clone() const { return rtagger_t(new ObjectTagger(*this)); }

      // Number of outputs
      virtual index_t n_outputs() const;

      // Number of types
      virtual index_t n_types() const { return m_param.m_labels.size() + 1; }

      // Label a sub-window
      virtual bool check(const ipscale_t& ipscale, int x, int y, 
          scalars_t& targets, index_t& type) const;

      // Return the label (ID or type) of a given object
      const string_t& label(const Object& object) const
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
