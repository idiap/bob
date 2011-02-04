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
     * @brief The member class for a dataset
     */
    class Member {
      public:
        /**
         * @brief Constructor
         */
        Member();

        /**
         * @brief Destructor
         */
        ~Member();

        /**
         * @brief Set the array-id for this member
         */
        void setArrayId(const size_t array_id) { m_array_id = array_id; }
        /**
         * @brief Get the array-id for this member
         */
        size_t getArrayId() const { return m_array_id; }

        /**
         * @brief Set the array-id for this member
         */
        void setArraysetId(const size_t arrayset_id) { 
          m_arrayset_id = arrayset_id; }
        /**
         * @brief Get the array-id for this member
         */
        size_t getArraysetId() const { return m_arrayset_id; }

      private:
        size_t m_array_id;
        size_t m_arrayset_id;
    };


    /**
     * @brief The relation class for a dataset
     */
    class Relation {
      public:
        /**
         * @brief Constructor
         */
        Relation(boost::shared_ptr<std::map<size_t,std::string> > id_role);

        /**
         * @brief Destructor
         */
        ~Relation();

        /**
         * @brief Add a member to the Relation
         */
        void append( boost::shared_ptr<Member> member);

        /**
         * @brief Set the id for this relation
         */
        void setId(const size_t id) { m_id = id; }
        /**
         * @brief Get the id for this relation
         */
        size_t getId() const { return m_id; }

        typedef std::pair<size_t, size_t> size_t_pair;
        /**
         * @brief const_iterator over the Members of the Relation
         */
        typedef std::map<size_t_pair, boost::shared_ptr<Member> >::const_iterator
          const_iterator;
        /**
         * @brief Return a const_iterator pointing at the first Member of 
         * the Relation
         */
        const_iterator begin() const { return m_member.begin(); }
        /**
         * @brief Return a const_iterator pointing at the last Member of 
         * the Relation
         */
        const_iterator end() const { return m_member.end(); }

        /**
         * @brief iterator over the Members of the Relation
         */
        typedef std::map<size_t_pair, boost::shared_ptr<Member> >::iterator 
          iterator;
        /**
         * @brief Return an iterator pointing at the first Member of the
         * Relation
         */
        iterator begin() { return m_member.begin(); }
        /**
         * @brief Return an iterator pointing at the last Member of the 
         * Relation
         */
        iterator end() { return m_member.end(); }


        /**
         * @brief iterator over the Members of the Relation with a given 
         * arrayset-role
         */
        template <typename T, typename U, typename V> 
        class iterator_template {
          public:
            /**
             * @brief Constructor
             */
            iterator_template(): m_str(""), m_it(0), m_parent(0) { }
            iterator_template(const std::string& str, V it, U* parent):
              m_str(str), m_it(it), m_parent(parent) { }

            T& operator*() const;
            T* operator->() const;
            iterator_template<T,U,V>& operator++(); // prefix
            iterator_template<T,U,V> operator++(int); // postfix
            bool operator==(const iterator_template<T,U,V>& i) const;
            bool operator!=(const iterator_template<T,U,V>& i) const;

          private:
            std::string m_str;
            V m_it;
            const U* m_parent;
        };
       
        /**
         * @warning Looking at the STL implementation of a map, the keys are 
         * const:
         * "template <typename _Key, typename _Tp, 
         *    typename _Compare = std::less<_Key>,
         *    typename _Alloc = std::allocator<std::pair<const _Key, _Tp> > >"
         * The following iterator typedefs take this fact into consideration,
         * and use a const size_t_pair as Keys type.
         */
        typedef iterator_template<std::pair<const size_t_pair, 
          boost::shared_ptr<Member> >, Relation, Relation::iterator> 
          iterator_b;
        typedef iterator_template<const std::pair<const size_t_pair,
          boost::shared_ptr<Member> >, const Relation, 
          Relation::const_iterator> const_iterator_b;

        /**
         * @brief Return an iterator pointing at the first Member of the 
         * Relation with a given arrayset-role
         */
        iterator_b begin(const std::string& str) {
          iterator it=begin();
          while( it!=end() &&
            m_id_role->operator[]( it->second->getArraysetId()).compare(str) )
            ++it;
          return iterator_b( str, it, this);
        }

        /**
         * @brief Return an iterator pointing at the last Member of the 
         * Relation with a given arrayset-role
         */
        iterator_b end(const std::string& str) {
          return iterator_b( str, end(), this);
        }

        /**
         * @brief Return an iterator pointing at the first Member of the 
         * Relation with a given arrayset-role
         */
        const_iterator_b begin(const std::string& str) const {
          const_iterator it=begin();
          while( it!=end() &&
            m_id_role->operator[]( it->second->getArraysetId()).compare(str) )
            ++it;
          return const_iterator_b( str, it, this);
        }

        /**
         * @brief Return an iterator pointing at the last Member of the 
         * Relation with a given arrayset-role
         */
        const_iterator_b end(const std::string& str) const {
          return const_iterator_b( str, end(), this);
        }


        boost::shared_ptr<std::map<size_t,std::string> > getIdRole() const {
          return m_id_role;
        }

      private:
        std::map<size_t_pair, boost::shared_ptr<Member> > m_member;
        size_t m_id;
        /**
         * @brief Mapping from arrayset-id to role
         */
        boost::shared_ptr<std::map<size_t,std::string> > m_id_role;
    };

    template <typename T, typename U, typename V> 
    T& Relation::iterator_template<T,U,V>::operator*() const {
      return *m_it;
    }

    template <typename T, typename U, typename V> 
    T* Relation::iterator_template<T,U,V>::operator->() const {
      return m_it.operator->();
    }

    template <typename T, typename U, typename V> 
    Relation::iterator_template<T,U,V>& 
    Relation::iterator_template<T,U,V>::operator++() {
      ++m_it;
      while( m_it!=m_parent->end() && 
        m_parent->getIdRole()->operator[]( 
          m_it->second->getArraysetId()).compare(m_str) )
        ++m_it;
      return *this;
    }

    template <typename T, typename U, typename V> 
    Relation::iterator_template<T,U,V> 
    Relation::iterator_template<T,U,V>::operator++(int) {
      m_it++;
      while( m_it!=m_parent->end() && 
        m_parent->getIdRole()->operator[]( 
          m_it->second->getArraysetId()).compare(m_str) )
        ++m_it;
      return *this;
    }

    template <typename T, typename U, typename V> 
    bool Relation::iterator_template<T,U,V>::operator==(
      const iterator_template<T,U,V>& it) const 
    {
      return m_it == it.m_it;
    }

    template <typename T, typename U, typename V> 
    bool Relation::iterator_template<T,U,V>::operator!=(
    const iterator_template<T,U,V>& it) const 
    {
      return m_it != it.m_it;
    }


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

