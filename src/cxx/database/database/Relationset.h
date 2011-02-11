/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Thu 10 Feb 18:58:40 2011 
 *
 * @brief The Relationset describes relations between the arrays and arraysets
 * in a Dataset.
 */

#ifndef TORCH_DATABASE_RELATIONSET_H 
#define TORCH_DATABASE_RELATIONSET_H

#include <list>
#include <string>
#include <cstdlib>
#include <map>
#include <pair>

namespace Torch { namespace database {

  /**
   * The Relationset class describes relations between Arraysets and Arrays in
   * a database, binding them to compose groups of identities or pattern-target
   * relationships for example.
   */
  class Relationset {

    private: //representation

      std::map<size_t, std::list<std::pair<size_t, size_t> > m_relation; ///< My declared relations
      std::map<std::string, 

  };

}}

#endif /* TORCH_DATABASE_RELATIONSET_H */

