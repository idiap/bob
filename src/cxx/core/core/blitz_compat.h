/**
 * @file src/cxx/core/core/blitz_compat.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines blitz-related types for compatibility purpose
 *
 */

#ifndef TORCH_CORE_BLITZ_COMPAT_H
#define TORCH_CORE_BLITZ_COMPAT_H

/**
 * Defines the diffType and sizeType in case blitz (old) don't have it defined
 * already.
 */
#ifndef HAVE_BLITZ_SIZETYPE
namespace blitz { typedef int sizeType; }
#endif
#ifndef HAVE_BLITZ_DIFFTYPE
namespace blitz { typedef int diffType; }
#endif

#endif /* TORCH_CORE_BLITZ_COMPAT_H */
