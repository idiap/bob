/**
 * @file visioner/visioner/util/manager.h
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

#ifndef BOB_VISIONER_MANAGER_H
#define BOB_VISIONER_MANAGER_H

#include <map>

#include <boost/serialization/singleton.hpp>
#include <boost/shared_ptr.hpp>

#include "core/logging.h"

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
          bob::core::error << "The manager cannot find the object <" << id << ">!" << std::endl;
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
