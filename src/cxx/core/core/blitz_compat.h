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
#if !defined(HAVE_BLITZ_SPECIAL_TYPES)
namespace blitz { 
  typedef int sizeType; 
  typedef int diffType;
}
#endif

#endif /* TORCH_CORE_BLITZ_COMPAT_H */
