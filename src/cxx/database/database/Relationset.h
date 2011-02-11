/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Thu 10 Feb 18:58:40 2011 
 *
 * @brief The Relationset describes relations between the arrays and arraysets
 * in a Dataset.
 */

#ifndef TORCH_DATABASE_RELATIONSET_H 
#define TORCH_DATABASE_RELATIONSET_H

#include <string>
#include <cstdlib>
#include <map>
#include <boost/shared_ptr.hpp>

#include "database/Relation.h"
#include "database/Rule.h"

namespace Torch { namespace database {

  /**
   * The Relationset class describes relations between Arraysets and Arrays in
   * a database, binding them to compose groups of identities or pattern-target
   * relationships for example.
   */
  class Relationset {

    public:

      /**
       * Constructor
       */
      Relationset ();

      /**
       * Copy constructor
       */
      Relationset (const Relationset& other);

      /**
       * Destructor virtualization
       */
      virtual ~Relationset();

      /**
       * Assignment operator
       */
      Relationset& operator= (const Relationset& other);

      /**
       * Adds rules. Please note that if you add a rule with a role that
       * already exists internally, that rule is overwritten. Returns the
       * amount of rules so far.
       */
      size_t add (const Rule& rule);
      size_t add (boost::shared_ptr<const Rule> rule);

      /**
       * Removes rules. Please note that if you remove rules that are obeyed by
       * Relations in this Relationset, their refered members will be also
       * removed.
       */
      void remove(const std::string& rulerole);

      /**
       * Adds a Relation, returns its allocated id, if it passes the rule check.
       * If the Relation id is set, I'll overwrite any currently occupied
       * position inside my map.
       *
       * Please note you cannot add relations while you have not established any
       * rules using add(rule) (and remove(rule)).
       *
       * Returns the assign id.
       */
      size_t add (const Relation& relation);
      size_t add (boost::shared_ptr<const Relation> relation);

      /**
       * Removes a relation with a certain id. If the Relation with the given id
       * does not exist, this is a noop.
       */
      void remove (size_t id);

      /**
       * Returns a pointer to my internal map of relations
       */
      inline const std::map<size_t, boost::shared_ptr<Relation> >& relations () const { return m_relation; }

      /**
       * Returns a pointer to my internal map of rules
       */
      inline const std::map<std::string, boost::shared_ptr<Rule> >& rules () { return m_rule; } 

      /**
       * Sets my name
       */
      inline void setName (const std::string& name) { m_name = name; }

      /**
       * Gets my name
       */
      inline const std::string& getName () const { return m_name; }

      /**
       * Returns a reference to an existing relation. Raises an exception if the
       * given relation is not there.
       */
      const Relation& operator[] (size_t id) const;

      /**
       * Returns a reference to an existing rule.
       */
      const Rule& operator[] (const std::string& role) const;

      /**
       * Returns a boost shared_ptr to the relation instead
       */
      boost::shared_ptr<const Relation> ptr (size_t id) const;

      /**
       * Returns a boost shared_ptr to the rule insted
       */
      boost::shared_ptr<const Rule> ptr(const std::string& role) const;

      /**
       * Returns the list of ids I have registered. You have to pass an STL
       * conforming container that has size_t elements and accepts push_back().
       */
      template <typename T> inline void index (T& container) {
        for (std::map<size_t, boost::shared_ptr<Relation> >::const_iterator it=m_relation.begin(); it != m_relation.end(); ++it) container.push_back(it->first);
      }

      /**
       * Returns the list of roles I have registered. You have to pass an STL
       * conforming container that has std::string elements and accepts
       * push_back().
       */
      template <typename T> inline void roles (T& container) {
        for (std::map<std::string, boost::shared_ptr<Rule> >::const_iterator it=m_rule.begin(); it != m_rule.end(); ++it) container.push_back(it->first);
      }

      /**
       * Gets the next free id for a Relation
       */
      size_t getNextFreeId() const;

      /**
       * Consolidates the relation ids by resetting the first array to have id =
       * 1, the second id = 2 and so on.
       */
      void consolidateIds();

    private: //representation
      std::string m_name; ///< My name
      std::map<size_t, boost::shared_ptr<Relation> > m_relation; ///< My declared relations
      std::map<std::string, boost::shared_ptr<Rule> > m_rule; ///< My currently set rules

  };

}}

#endif /* TORCH_DATABASE_RELATIONSET_H */
