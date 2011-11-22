/**
 * @file cxx/old/trainer/src/FTrainer.cc
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
#include "trainer/FTrainer.h"

namespace Torch {

///////////////////////////////////////////////////////////////////////////
// Constructor

FTrainer::FTrainer() : Trainer()
{
}

FTrainer::~FTrainer()
{
}
float FTrainer::forwardScan(const Tensor &example_,TensorRegion &tregion)
{
	print("Ftrainer::Should not be here\n");
	return 1.0;
}
}

