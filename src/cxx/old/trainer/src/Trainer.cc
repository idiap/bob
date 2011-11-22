/**
 * @file cxx/old/trainer/src/Trainer.cc
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
#include "trainer/Trainer.h"

namespace Torch {

////////////////////////////////////////////////////////////////////////
/// Constructor

Trainer::Trainer(): Object()
{
	m_dataset = NULL;
	m_machine = NULL;
}

////////////////////////////////////////////////////////////////////////

bool Trainer::setData(DataSet* m_dataset_)
{
   	if(m_dataset_ == NULL) return false;
	m_dataset = m_dataset_;
	return true;
}
///////////////////////////////////////
bool Trainer::setMachine(Machine* m_machine_)
{
   	if(m_machine_ == NULL) return false;
	m_machine = m_machine_;
	return true;
}

/// Destructor
Trainer::~Trainer()
{
}

////////////////////////////////////////////////////////////////////////

}

