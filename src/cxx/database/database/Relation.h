/**
 * @file database/Relation.h
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Declares the Relation class for the Torch Dataset system.
 */

#ifndef TORCH_DATABASE_RELATION_H 
#define TORCH_DATABASE_RELATION_H

namespace Torch { namespace database {

  /**
   * The relation class for a dataset combines Members (array/arrayset
   * pointers) to indicate relationship between database arrays and arraysets.
   */
  class Relation {

    public:
      /**
       * Constructor.
       */
      Relation(boost::shared_ptr<std::map<size_t,std::string> > id_role);

      /**
       * Destructor
       */
      ~Relation();

      /**
       * Add a member to the Relation
       */
      void append( boost::shared_ptr<Member> member);

      /**
       * Set the id for this relation
       */
      void setId(const size_t id) { m_id = id; }

      /**
       * Get the id for this relation
       */
      size_t getId() const { return m_id; }

      typedef std::pair<size_t, size_t> size_t_pair;

      /**
       * const_iterator over the Members of the Relation
       */
      typedef std::map<size_t_pair, boost::shared_ptr<Member> >::const_iterator
        const_iterator;

      /**
       * Return a const_iterator pointing at the first Member of 
       * the Relation
       */
      const_iterator begin() const { return m_member.begin(); }

      /**
       * Return a const_iterator pointing at the last Member of 
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

} }

#endif /* TORCH_DATABASE_RELATION_H */

