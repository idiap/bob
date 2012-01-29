/**
 * @file cxx/core/core/Random.h
 * @date Mon Jan 31 20:02:21 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Random integer generators based on Boost
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef BOB_CORE_RANDOM_H
#define BOB_CORE_RANDOM_H

#include <boost/random.hpp>
#include <boost/scoped_ptr.hpp>

namespace bob {

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
          return rng(bob::core::random::generator::instance());
        }
      };

      template <typename T> struct uniform_uint {
        T operator()(const T& i) {
          boost::uniform_int<T> rng(0, i-1);
          return rng(bob::core::random::generator::instance());
        }
      };

      template <typename T> struct uniform_real {
        T operator()(const T& min_arg, const T& max_arg) {
          boost::uniform_real<T> rng(min_arg, max_arg);
          return rng(bob::core::random::generator::instance());
        }
      };

    }
  }
}

#endif
