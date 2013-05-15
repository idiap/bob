/**
 * @file trainer/python/overload/main.cc
 * @date Thu Jun 9 18:12:33 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Combines all modules to make up the complete bindings
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

#include <bob/core/python/ndarray.h>

void bind_trainer_kmeans_wrappers();
void bind_trainer_gmm_wrappers();
void bind_trainer_mlp_wrappers();

BOOST_PYTHON_MODULE(_trainer_overload) {

  bob::python::setup_python("bob classes and sub-classes for overloading trainers");
  
  bind_trainer_kmeans_wrappers();
  bind_trainer_gmm_wrappers();
  bind_trainer_mlp_wrappers();
}
