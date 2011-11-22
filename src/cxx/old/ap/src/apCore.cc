/**
 * @file cxx/old/ap/src/apCore.cc
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
#include "ap/apCore.h"

namespace Torch {

//////////////////////////////////////////////////////////////////////////
// Constructor

apCore::apCore()
	: 	spCore()
{
	m_audioSize = 0;
}

//////////////////////////////////////////////////////////////////////////
// Destructor

apCore::~apCore()
{
}

//////////////////////////////////////////////////////////////////////////
// Change the input audio size

bool apCore::setAudioSize(int new_length)
{
	if (new_length > 0)
	{
		m_audioSize = new_length;
		return true;
	}
	return false;
}

//////////////////////////////////////////////////////////////////////////
// Retrieve the input audio size

int apCore::getAudioSize() const
{
	return m_audioSize;
}

//////////////////////////////////////////////////////////////////////////
}

