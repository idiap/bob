/**
 * @file src/cxx/database/database/Relationset.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief A torch representation of a Relationset for a Dataset.
 */

#ifndef TORCH5SPRO_DATABASE_RELATIONSET_H
#define TORCH5SPRO_DATABASE_RELATIONSET_H 1

#include "database/Arrayset.h"

namespace Torch {   
  /**
   * \ingroup libdatabase_api
   * @{
   *
   */
  namespace database {
    
    /**
     * @brief The rule class for a dataset
     */
    class Rule {
      public:
        /**
         * @brief Constructor
         */
        Rule();

        /**
         * @brief Destructor
         */
        ~Rule();

        /**
         * @brief Set the arraysetrole for this rule
         */
        void setArraysetRole(const std::string& arraysetrole) { 
          m_arraysetrole.assign(arraysetrole); }
        /**
         * @brief Set the minimum number of occurences for this rule
         */
        void setMin(const size_t min) { m_min = min; }
        /**
         * @brief Set the maximum number of occurences for this rule
         */
        void setMax(const size_t max) { m_max = max; }
        /**
         * @brief Get the arrayset role for this rule
         */
        const std::string& getArraysetRole() const { return m_arraysetrole; }
        /**
         * @brief Get the minimum number of occurences for this rule
         */
        size_t getMin() const { return m_min; }
        /**
         * @brief Get the maximum number of occurences for this rule
         */
        size_t getMax() const { return m_max; }

      private:
        std::string m_arraysetrole;
        size_t m_min;
        size_t m_max;
    };


    /**
     * @brief The relationset class for a dataset
     */
    class Relationset {
      public:
        /**
         * @brief Constructor
         */
        Relationset();

        /**
         * @brief Destructor
         */
        ~Relationset();

        /**
         * @brief Add a Relation to the Relationset
         */
        void append( boost::shared_ptr<Relation> relation);

        /**
         * @brief Add a Rule to the Relationset
         */
        void append( boost::shared_ptr<Rule> rule);

        /**
         * @brief Get the name of this Relationset
         */
        const std::string& getName() const { return m_name; }
        /**
         * @brief Set the name of this Relationset
         */
        void setName(const std::string& name) { m_name.assign(name); }

        /**
         * @brief const_iterator over the Relations of the Relationset
         */
        typedef std::map<size_t, boost::shared_ptr<Relation> >::const_iterator
          const_iterator;
        /**
         * @brief Return a const_iterator pointing at the first Relation of 
         * the Relationset
         */
        const_iterator begin() const { return m_relation.begin(); }
        /**
         * @brief Return a const_iterator pointing at the last Relation of 
         * the Relationset
         */
        const_iterator end() const { return m_relation.end(); }

        /**
         * @brief iterator over the Relations of the Relationset
         */
        typedef std::map<size_t, boost::shared_ptr<Relation> >::iterator 
          iterator;
        /**
         * @brief Return an iterator pointing at the first Relation of the
         * Relationset
         */
        iterator begin() { return m_relation.begin(); }
        /**
         * @brief Return an iterator pointing at the last Relation of the 
         * Relationset
         */
        iterator end() { return m_relation.end(); }

        /**
         * @brief const_iterator over the Rules of the Relationset
         */
        typedef std::map<std::string, boost::shared_ptr<Rule> >::const_iterator
          rule_const_iterator;
        /**
         * @brief Return a rule_const_iterator pointing at the first Rule of 
         * the Relationset
         */
        rule_const_iterator rule_begin() const { return m_rule.begin(); }
        /**
         * @brief Return a rule_const_iterator pointing at the last Rule of
         * the Relationet
         */
        rule_const_iterator rule_end() const { return m_rule.end(); }

        /**
         * @brief iterator over the Rules of the Relationset
         */
        typedef std::map<std::string, boost::shared_ptr<Rule> >::iterator 
          rule_iterator;
        /**
         * @brief Return an iterator pointing at the first Rule of the
         * Relationset
         */
        rule_iterator rule_begin() { return m_rule.begin(); }
        /**
         * @brief Return an iterator pointing at the last Rule of the 
         * Relationset
         */
        rule_iterator rule_end() { return m_rule.end(); }

        /**
         * @brief Return the relation of the given id
         * @warning Please note that if you use that method, scope matters,
         * because the dataset owns the relations.
         */
        const Relation& operator[](size_t id) const;
        /**
         * @brief Return a smart pointer to the relation of the given id
         */
        boost::shared_ptr<const Relation> getRelation(size_t id) const;
        boost::shared_ptr<Relation> getRelation(size_t id);

        /**
         * @brief Return the rule object referred by the given role 
         * @warning Please note that if you use that method, scope matters,
         * because the dataset owns the rules.
         */
        const Rule& operator[](const std::string& name) const;
        /**
         * @brief Return a smart pointer to the rule given the role
         */
        boost::shared_ptr<const Rule> getRule(const std::string& name) const;
        boost::shared_ptr<Rule> getRule(const std::string& name);

      private:
        std::string m_name;

        std::map<std::string, boost::shared_ptr<Rule> > m_rule;        
        std::map<size_t, boost::shared_ptr<Relation> > m_relation;
    };





  }
  /**
   * @}
   */
}

#endif /* TORCH5SPRO_DATABASE_RELATIONSET_H */

