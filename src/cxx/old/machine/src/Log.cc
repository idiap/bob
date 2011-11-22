/**
 * @file cxx/old/machine/src/Log.cc
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
#include "machine/Log.h"

namespace Torch {

//////////////////////////////////////////////////////////////////////////
// Constructor

Log::Log() : GradientMachine()
{
}

Log::Log(const int n_units_) : GradientMachine(n_units_, n_units_)
{
}

//////////////////////////////////////////////////////////////////////////
// Destructor

Log::~Log()
{
}


//////////////////////////////////////////////////////////////////////////
// Process the input tensor

bool Log::forward(const DoubleTensor *input)
{
  const int K=input->sizeAll();
  for (int k=0; k<K; ++k) m_output(k) = log((*input)(k));
	return true;
}

bool Log::backward(const DoubleTensor *input, const DoubleTensor *alpha)
{
   	warning("Log::backward() not implemented.");

	return false;
}

//////////////////////////////////////////////////////////////////////////
}
