/**
 * @file database/Rule.h
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Rules define organizational requirements for members in a Relation.
 */

#ifndef TORCH_DATABASE_RULE_H 
#define TORCH_DATABASE_RULE_H

#include <cstdlib>
#include <string>

namespace Torch { namespace database {

  /**
   * The Rule class establishes and controls the insertion of Members in a
   * Relation.
   */
  class Rule {

    public:

      /**
       * Constructor, establishes a Rule for a certain arrayset role with a
       * minimum set of occurences an a maximum one.
       */
      Rule (size_t min=1, size_t max=1);

      /**
       * Destructor virtualization
       */
      virtual ~Rule();

      /**
       * Accessors
       */
      inline size_t getMin() const { return m_min; }
      inline size_t getMax() const { return m_max; }

    private:

      size_t m_min; ///< the minimum amount of members of this kind
      size_t m_max; ///< the maximum amount of members of this kind

  };

}}

#endif /* TORCH_DATABASE_RULE_H */

