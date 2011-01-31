/**
 * @file src/cxx/core/core/Random.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a>
 *
 * @brief Random integer generators based on Boost
 */

#ifndef TORCH5SPRO_CORE_RANDOM_H
#define TORCH5SPRO_CORE_RANDOM_H 1

#include <boost/random.hpp>
#include <boost/scoped_ptr.hpp>

namespace Torch {

  namespace core {

    namespace random {

      class generator {
        public:
          virtual ~generator() {}
          static boost::mt19937& instance();

        private:
          static boost::scoped_ptr<boost::mt19937> s_generator;
      };


      template <typename T> struct uniform_int {
        T operator()(const T& min_arg, const T& max_arg) {
          boost::uniform_int<T> rng(min_arg, max_arg);
          return rng(Torch::core::random::generator::instance());
        }
      };

      template <typename T> struct uniform_uint {
        T operator()(const T& i) {
          boost::uniform_int<T> rng(0, i-1);
          return rng(Torch::core::random::generator::instance());
        }
      };

      template <typename T> struct uniform_real {
        T operator()(const T& min_arg, const T& max_arg) {
          boost::uniform_real<T> rng(min_arg, max_arg);
          return rng(Torch::core::random::generator::instance());
        }
      };

    }
  }
}

#endif
