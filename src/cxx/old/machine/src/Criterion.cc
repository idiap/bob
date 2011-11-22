/**
 * @file cxx/old/machine/src/Criterion.cc
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
#include "machine/Criterion.h"

namespace Torch {

//////////////////////////////////////////////////////////////////////////
// Constructor

Criterion::Criterion(const int target_size, const int error_size)
{
   	m_target_size = target_size;
   	m_error_size = error_size;
	m_error = NULL;
	m_beta = NULL;
	m_error = new DoubleTensor(m_error_size);
	m_beta = new DoubleTensor(m_target_size);
	m_target = new DoubleTensor(m_target_size);
}

//////////////////////////////////////////////////////////////////////////
// Destructor

Criterion::~Criterion()
{
        // Cleanup
	delete m_error;
	delete m_beta;
	delete m_target;
}

//////////////////////////////////////////////////////////////////////////
}
