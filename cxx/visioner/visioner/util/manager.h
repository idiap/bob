#ifndef BOB_VISIONER_MANAGER_H
#define BOB_VISIONER_MANAGER_H

#include <map>

#include <boost/serialization/singleton.hpp>
#include <boost/shared_ptr.hpp>

#include "visioner/util/util.h"

namespace bob { namespace visioner {

  /////////////////////////////////////////////////////////////////////////////////////////
  // Manager: used to retrieve an object type based on a given ID.
  //	The objects need to posses a ::clone() function.
  /////////////////////////////////////////////////////////////////////////////////////////

  template <typename TObject>
    class Manager : public boost::serialization::singleton<Manager<TObject> >
  {
    protected:	

      typedef string_t			ID;		
      typedef boost::shared_ptr<TObject>	robject_t;

      // Constructor
      Manager() 
      {
      }

    public:

      // Register a new classifier
      void add(ID id, const TObject& proto)
      {
        m_prototypes[id] = proto.clone();
      }

      // Retrieve a classifier prototype by its ID
      robject_t get(ID id) const
      {
        const typename std::map<ID, robject_t>::const_iterator it = m_prototypes.find(id);
        if (it == m_prototypes.end())
        {
          log_error("Manager", "get") 
            << "The manager cannot find the object <" << id << ">!\n";
          exit(EXIT_FAILURE);
        }
        return it->second->clone();
      }

      // Retrieve the registered IDs as a list
      string_t describe() const
      {
        string_t desc;
        for (typename std::map<ID, robject_t>::const_iterator it = m_prototypes.begin();
            it != m_prototypes.end(); ++ it)
        {
          desc += it->first;

          typename std::map<ID, robject_t>::const_iterator it2 = it;
          ++ it2;
          if (it2 != m_prototypes.end())
            desc += ", ";
        }

        return desc;
      }

    private:

      // Attributes
      std::map<ID, robject_t>	m_prototypes;
  };

  template <typename TObject>
    class Manageable
    {
      public:

        typedef boost::shared_ptr<Manageable<TObject> >	robject_t;

        // Constructor
        Manageable(const TObject& data)      
          :       m_data(data)
        {                        
        }

        // Clone the object
        robject_t clone() const { return robject_t(new Manageable<TObject>(m_data)); }

        // Access function
        TObject operator*() const { return m_data; }

      private:

        // Attributes
        TObject         m_data;
    };

}}

#endif // BOB_VISIONER_MANAGER_H
