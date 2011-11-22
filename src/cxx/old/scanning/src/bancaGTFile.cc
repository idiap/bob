/**
 * @file cxx/old/scanning/src/bancaGTFile.cc
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
#include "scanning/bancaGTFile.h"

namespace Torch {

bancaGTFile::bancaGTFile() : GTFile(10)
{
	CHECK_FATAL(setLabel(1, "leye_center") == true);
	CHECK_FATAL(setLabel(4, "reye_center") == true);
}

bool bancaGTFile::load(File* file)
{
	const bool verbose = getBOption("verbose");

	if(verbose) message("bancaGTFile::load() ...");

	// BANCA format
	float x, y;

	for(int i = 0 ; i < 10 ; i++)
	{
		file->scanf("%f", &x);
		file->scanf("%f", &y);
		m_points[i].x = x;
		m_points[i].y = y;
	}

	if(verbose)
	{
		message("%s", getName());
		for(int i = 0 ; i < m_n_points ; i++)
			print(" [%d] %s (%.1f-%.1f)\n", i, getLabel(i), m_points[i].x, m_points[i].y);
	}

	return true;
}

const char* bancaGTFile::getName()
{
	return "BANCA format";
}

bancaGTFile::~bancaGTFile()
{
}

}



