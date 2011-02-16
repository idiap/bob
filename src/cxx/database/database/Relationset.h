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
#include <vector>
#include <boost/shared_ptr.hpp>

#include "database/Relation.h"
#include "database/Rule.h"

namespace Torch { namespace database {

  class Dataset;

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
       * Copy constructor. If a Dataset parent was set, this will also be
       * copied, but I'll not trigger a add() of myself on that Dataset. It
       * will only be used for checking purposes. You can attach this
       * Relationset to another dataset if you like using the Dataset::add()
       * method.
       */
      Relationset (const Relationset& other);

      /**
       * Destructor virtualization
       */
      virtual ~Relationset();

      /**
       * Assignment operator. If a Dataset parent was set, this will also be
       * copied, but I'll not trigger a add() of myself on that Dataset. It
       * will only be used for checking purposes. You can attach this
       * Relationset to another dataset if you like using the Dataset::add()
       * method.
       */
      Relationset& operator= (const Relationset& other);

      /**
       * Adds rules. Will work if no other rule for the given role has been set
       * yet.
       */
      void add (const std::string& role, const Rule& rule);
      void add (const std::string& role, boost::shared_ptr<const Rule> rule);

      /**
       * Overwrites rules. Will work if the rule has already been set.
       */
      void set (const std::string& role, const Rule& rule);
      void set (const std::string& role, boost::shared_ptr<const Rule> rule);

      /**
       * Removes rules that exist, it is an error to remove an unexisting rule.
       * To get existing rule either use exists() or get the rule index with
       * rules().
       */
      void remove(const std::string& role);

      /**
       * Adds a Relation, returns its allocated id, if it passes the rule check.
       *
       * Please note you cannot add relations while you have not established any
       * rules using add(rule) (and remove(rule)).
       *
       * Returns the assign id.
       */
      size_t add (const Relation& relation);
      size_t add (boost::shared_ptr<const Relation> relation);

      /**
       * Adds a Relation, will work if id has not been set yet
       *
       * Please note you cannot add relations while you have not established any
       * rules using add(rule) (and remove(rule)).
       *
       * Returns the assign id.
       */
      void add (size_t id, const Relation& relation);
      void add (size_t id, boost::shared_ptr<const Relation> relation);

      /**
       * Overwrites a Relation, will work if id has already been set
       *
       * Please note you cannot add relations while you have not established any
       * rules using add(rule) (and remove(rule)).
       *
       * Returns the assign id.
       */
      void set (size_t id, const Relation& relation);
      void set (size_t id, boost::shared_ptr<const Relation> relation);

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
      inline const std::map<std::string, boost::shared_ptr<Rule> >& rules () const { return m_rule; } 

      /**
       * Returns a reference to an existing relation. Raises an exception if the
       * given relation is not there.
       */
      Relation& operator[] (size_t id);
      const Relation& operator[] (size_t id) const;

      /**
       * Returns a reference to an existing rule.
       */
      Rule& operator[] (const std::string& role);
      const Rule& operator[] (const std::string& role) const;

      /**
       * Returns a boost shared_ptr to the relation instead
       */
      boost::shared_ptr<Relation> ptr (size_t id);
      boost::shared_ptr<const Relation> ptr (size_t id) const;

      /**
       * Returns a boost shared_ptr to the rule insted
       */
      boost::shared_ptr<Rule> ptr(const std::string& role);
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
       * Consolidates the relation ids by resetting the first array to have id =
       * 1, the second id = 2 and so on.
       */
      void consolidateIds();

      /**
       * Use the next two methods with care as they influence on how you can
       * add/remove relations.
       */
      void setParent (const Dataset* parent) { m_parent = parent; }
      const Dataset* getParent () const { return m_parent; }

      /**
       * Clear relations
       */
      inline void clearRelations () { m_relation.clear(); }

      /**
       * Clear rules in a safe way
       */
      void clearRules ();

      /**
       * Checks existence of Relation or Rule
       */
      bool exists(size_t relation_id) const;
      bool exists(const std::string& rule_role) const;

      /**
       * Given a certain internal relation checks, I can fill a map for you
       * that maps roles to pairs (arrayset-id, array-id), so you get a
       * segmented representation of a Relation. You must pass a std::map in
       * which the keys are the roles and the values are vectors of pairs
       * (arrayset-id, array-id).
       */
      void fillMemberMap(size_t relation_id, std::map<std::string, std::vector<std::pair<size_t, size_t> > >& dictionary) const;

    private: //a few helpers for the work

      /**
       * Gets the next free id for a Relation
       */
      size_t getNextFreeId() const;

      /**
       * Checks if a given relation respects all my rules
       */
      void checkRelation(const Relation& relation) const;

    private: //representation
      const Dataset* m_parent; ///< My parent dataset
      std::map<size_t, boost::shared_ptr<Relation> > m_relation; ///< My declared relations
      std::map<std::string, boost::shared_ptr<Rule> > m_rule; ///< My currently set rules

  };

}}

#endif /* TORCH_DATABASE_RELATIONSET_H */
