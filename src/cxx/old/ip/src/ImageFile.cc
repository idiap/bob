/**
 * @file cxx/old/ip/src/ImageFile.cc
 * @date Sat Apr 30 18:41:25 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
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
#include "ip/ImageFile.h"
#include "ip/Image.h"
#include "ip/Convert.h"

namespace Torch {

///////////////////////////////////////////////////////////////////////////////
// Constructor

ImageFile::ImageFile()
{
}

///////////////////////////////////////////////////////////////////////////////
// Destructor

ImageFile::~ImageFile()
{
	m_file.close();
}

///////////////////////////////////////////////////////////////////////////////
// Save an image

bool ImageFile::save(const Image& image, const char* filename)
{
	bool ret = false;

	if (m_file.open(filename, "w") && writeHeader(image) && writePixmap(image))
	{
		ret = true;
	}
	m_file.close();

	return ret;
}

///////////////////////////////////////////////////////////////////////////////
// Load an image

bool ImageFile::load(Image& image, const char* filename)
{
	bool ret = false;

	if (m_file.open(filename, "r") && readHeader(image) && readPixmap(image))
	{
		ret = true;
	}
	m_file.close();

	return ret;
}

}
