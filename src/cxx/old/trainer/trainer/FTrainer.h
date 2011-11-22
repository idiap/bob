/**
 * @file cxx/old/trainer/trainer/FTrainer.h
 * @date Wed Apr 6 14:49:40 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
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
#ifndef F_TRAINER_INC
#define F_TRAINER_INC

#include "trainer/Trainer.h"

namespace Torch
{
	class FTrainer : public Trainer
	{
	public:

		/// Constructor
		FTrainer();

		/// Destructor
		virtual ~FTrainer();

		/// Train the given machine on the given dataset
		virtual double forward(const Tensor *example_) = 0; //get the output score for a single stage
		virtual float forwardScan(const Tensor &example_,TensorRegion &tregion);
	};
}

#endif
