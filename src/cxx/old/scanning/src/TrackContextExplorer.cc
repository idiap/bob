/**
 * @file cxx/old/scanning/src/TrackContextExplorer.cc
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
#include "scanning/TrackContextExplorer.h"

namespace Torch
{

/////////////////////////////////////////////////////////////////////////
// Constructor

TrackContextExplorer::TrackContextExplorer(Mode mode)
	:	ContextExplorer(mode)
{
}

/////////////////////////////////////////////////////////////////////////
// Destructor

TrackContextExplorer::~TrackContextExplorer()
{
}

/////////////////////////////////////////////////////////////////////////
// Change the sub-windows to process

void TrackContextExplorer::setSeedPatterns(const PatternList& patterns)
{
	m_seed_patterns.clear();
	m_seed_patterns.add(patterns);
}

/////////////////////////////////////////////////////////////////////////
// Initialize the sub-windows to process

bool TrackContextExplorer::initContext()
{
	if (m_seed_patterns.isEmpty())
	{
	      return ContextExplorer::initContext();
	}
	else
	{
	      m_data->clear();
	      for (int i = 0; i < m_seed_patterns.size(); i ++)
	      {
		      m_data->storePattern(m_seed_patterns.get(i));
	      }
	      return true;
	}
}

/////////////////////////////////////////////////////////////////////////

}
