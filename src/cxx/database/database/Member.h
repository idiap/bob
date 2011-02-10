/**
 * @file database/Member.h
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Members are the atomic units of relations in a Dataset
 */

#ifndef TORCH_DATABASE_MEMBER_H 
#define TORCH_DATABASE_MEMBER_H

#include <cstdlib>

namespace Torch { namespace database {
    
  /**
   * The member class for a dataset represents relation pointers to arrays or
   * arraysets that are present in the database.
   */
  class Member {

    public:
      /**
       * Constructor. Initalizes a new Member by telling to which
       * arrayset/array to point to. The array_id may be zero, indicating this
       * Member points to a whole arrayset. The arrayset_id cannot be made
       * zero.
       */
      Member(size_t arrayset_id, size_t array_id=0);

      /**
       * Destructor virtualization
       */
      virtual ~Member();

      inline size_t getArraysetId () const { return m_arrayset_id; }
      inline size_t getArrayId () const { return m_array_id; }

    private:
      size_t m_arrayset_id;
      size_t m_array_id;
  };

}}

#endif /* TORCH_DATABASE_MEMBER_H */

