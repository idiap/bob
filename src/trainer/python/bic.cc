/**
 * @file trainer/python/bic.cc
 * @date Wed Jun  6 10:29:09 CEST 2012
 * @author Manuel Guenther <Manuel.Guenther@idiap.ch>
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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

#include <boost/python.hpp>
#include "bob/trainer/BICTrainer.h"

void bind_trainer_bic(){

  boost::python::class_<bob::trainer::BICTrainer, boost::shared_ptr<bob::trainer::BICTrainer> > (
      "BICTrainer",
      "A Trainer for a BICMachine. It trains either a BIC model (including projection matrix and eigenvalues), "
          "or an IEC model (containing mean and variance only). See :py:class:`bob.machine.BICMachine` for more details.",
      boost::python::init<int,int>(
          (
              boost::python::arg("intra_dim"),
              boost::python::arg("extra_dim")
          ),
          "Initializes the BICTrainer to train a BIC model with the given resulting dimensions of the intraperonal and extrapersonal subspaces."
      )
    )

    .def(
      boost::python::init<>(
        "Initializes the BICTrainer to train a IEC model."
      )
    )

    .def(
      "train",
      &bob::trainer::BICTrainer::train,
      (
          boost::python::arg("machine"),
          boost::python::arg("intra_differences"),
          boost::python::arg("extra_differences")
      ),
      "Trains the given machine (should be of type :py:class:`bob.machine.BICMachine`) to classify intrapersonal image differences vs. extrapersonal ones. "
      "The given difference vectors might be the result of any image comparison function, e.g., the pixel difference of the images."
    );
}
