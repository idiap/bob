/**
 * @file database/Relation.h
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Declares the Relation class for the Torch Dataset system.
 */

#ifndef TORCH_DATABASE_RELATION_H 
#define TORCH_DATABASE_RELATION_H

#include <list>
#include <string>
#include <cstdlib>
#include <boost/shared_ptr.hpp>

namespace Torch { namespace database {

  /**
   * The relation class for a dataset combines Members (array/arrayset
   * pointers) to indicate relationship between database arrays and arraysets.
   */
  class Relation {

    //I promise this exists
    class Rule;

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
       * Appends a member to the Relation. No further checks are performed.
       */
      inline void add (size_t arraysetid) {
        m_member.push_back(std::make_pair(arraysetid, 0));
      }

      inline void add (size_t arraysetid, size_t arrayid) {
        m_member.push_back(std::make_pair(arraysetid, arrayid));
      }

      /**
       * Adds by index or resets a position in this Relation. If the position
       * indicated exists, I replace this position by the value given. If not,
       * I raise an exception.
       */
      void set (size_t index, size_t arrayset_id);
      void set (size_t index, size_t arrayset_id, size_t array_id);

      /**
       * Removes a member, given its index. If the member does not exist, I
       * raise an exception.
       * 
       */
      void erase (size_t index);

      /**
       * Removes a member, given its arrayset id. It is not an error to
       * specify arraysets that don't exist in the relation. Please note this
       * method will only remove full arrayset entries from the relation.
       */
      void remove (size_t arrayset_id);

      /**
       * Removes a member, given its arrayset id and array id. It is not an
       * error to specify arrays that don't exist in the relation. 
       */
      void remove (size_t arrayset_id, size_t array_id);

      /**
       * Given the index, returns a std::pair<size_t, size_t> where 'first' is
       * the arrayset id and 'second' is the array id. If the array id is set
       * to zero, it means this member points to an arrayset instead of a
       * single array. This will throw an exception if the id was not
       * registered in this Relation.
       */
      const std::pair<size_t, size_t>& operator[] (size_t index) const;

      /**
       * How to get a handle to all my roles. You must provide a container that
       * accepts push_back() and has std::string elements (e.g.
       * std::vector<std::string> or std::list<std::string>)
       */
      template <typename T> inline void index(T& container) const {
        for (std::list<std::pair<size_t,size_t> >::const_iterator it=m_member.begin(); it!=m_member.end(); ++it) container.push_back(it->first);
      }

      /**
       * A handle to all my members
       */
      inline const std::list<std::pair<size_t,size_t> >& members() const {
        return m_member;
      }

    private:
      std::list<std::pair<size_t, size_t> > m_member; ///< my members

  };

} }

#endif /* TORCH_DATABASE_RELATION_H */
