/**
 * @file database/Relation.h
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Declares the Relation class for the Torch Dataset system.
 */

#ifndef TORCH_DATABASE_RELATION_H 
#define TORCH_DATABASE_RELATION_H

#include <map>
#include <string>
#include <cstdlib>

namespace Torch { namespace database {

  /**
   * The relation class for a dataset combines Members (array/arrayset
   * pointers) to indicate relationship between database arrays and arraysets.
   */
  class Relation {

    //I promise this exists
    //class Relationset;

    public:
      /**
       * Constructor, builds an empty Relation. 
       */
      Relation();

      /**
       * Copy constructor
       */
      Relation (const Relation& other);

      /**
       * Destructor
       */
      virtual ~Relation();

      /**
       * Assignment operation
       */
      Relation& operator= (const Relation& other);

      /**
       * Adds a member to the Relation. If a member with a given role already
       * exists, it is overwritten.
       */
      void add (const std::string& role, size_t arraysetid);
      void add (const std::string& role, size_t arraysetid, size_t arrayid);

      /**
       * Removes a member, given the role. If the member does not exist, this
       * is a noop.
       */
      void remove (const std::string& role);

      /**
       * Gets the id for this relation
       */
      inline size_t getId() const { return m_id; }

      /**
       * Given the role, returns a std::pair<size_t, size_t> where 'first' is
       * the arrayset id and 'second' is the array id. If the array id is set
       * to zero, it means this member points to an arrayset instead of a
       * single array. This will throw an exception if the role was not
       * registered in this Relation.
       */
      const std::pair<size_t, size_t>& operator[] (const std::string& role);

      /**
       * How to get a handle to all my roles. You must provide a container that
       * accepts push_back() and has std::string elements (e.g.
       * std::vector<std::string> or std::list<std::string>)
       */
      template <typename T> void index(T& container) const {
        for (std::map<std::string, std::pair<size_t,size_t> >::const_iterator it=m_member.begin(); it!=m_member.end(); ++it) container.push_back(it->first);
      }

      /**
       * A handle to all my members
       */
      const std::map<std::string, std::pair<size_t,size_t> >& members() const {
        return m_member;
      }

      /**
       * Gets the parent arrayset of this array
       */
      /**
      inline boost::shared_ptr<const Relationset> getParent() const { 
        return m_parent.lock(); 
      }
      **/

      //The next methods are sort of semi-private: Only to be used by the
      //Database loading system. You can adventure yourself, but really not
      //recommended to set the id or the parent of an array. Be sure to
      //understand the consequences.
 
      /**
       * Sets the parent arrayset of this array. Please note this is a simple
       * assignment that has to be done by the Dataset parent of the Arrayset
       * as it is the only entity in the system that holds a
       * boost::shared_ptr<> to an Arrayset.
       *
       * It is meant to be used in the context of the database creation. So,
       * not for us, mortal users ;-)
       */
      /**
      inline void setParent (boost::shared_ptr<Relationset> parent) { 
        m_parent = parent;
      }
      **/

      /**
       * Sets the id for this relation. This is some sort of semi-private
       * method and is intended only for database parsers. Use it with care.
       */
      inline void setId(const size_t id) { m_id = id; }

    private:
      //boost::weak_ptr<Relationset> m_parent; ///< my parent relation set
      size_t m_id; ///< my identifier
      std::map<std::string, std::pair<size_t, size_t> > m_member; ///< my members

  };

} }

#endif /* TORCH_DATABASE_RELATION_H */

