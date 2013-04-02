/**
 * @file trainer/python/main.cc
 * @date Tue May 10 11:35:58 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
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

#include "bob/config.h"
#include "bob/core/python/ndarray.h"

void bind_trainer_linear();
void bind_trainer_gmm();
void bind_trainer_kmeans();
void bind_trainer_rprop();
void bind_trainer_backprop();
void bind_trainer_jfa();
void bind_trainer_ivector();
void bind_trainer_plda();
void bind_trainer_wiener();
void bind_trainer_empca();
void bind_trainer_bic();
void bind_trainer_llr();

#if WITH_LIBSVM
void bind_trainer_svm();
#endif

BOOST_PYTHON_MODULE(_trainer) {

  bob::python::setup_python("bob classes and sub-classes for trainers");
  
  bind_trainer_linear();
  bind_trainer_gmm();
  bind_trainer_kmeans();
  bind_trainer_rprop();
  bind_trainer_backprop();
  bind_trainer_jfa();
  bind_trainer_ivector();
  bind_trainer_plda();
  bind_trainer_wiener();
  bind_trainer_empca();
  bind_trainer_bic();
  bind_trainer_llr();

# if WITH_LIBSVM
  bind_trainer_svm();
# endif
}
