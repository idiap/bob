/**
 * @file visioner/programs/vgui/image_collection.cc
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

#include <QProgressDialog>

#include "image_collection.h"

ImageCollection::ImageCollection():	m_crt_index(0)
{	
}

void ImageCollection::clear()
{
  m_listfiles.clear();
  m_ifiles.clear();
  m_gfiles.clear();
  m_ipscales.clear();
  m_crt_index = 0;
}

bool ImageCollection::add(const std::string& list_file)
{
  std::vector<std::string> ifiles, gfiles;
  if (	bob::visioner::load_listfiles(list_file, ifiles, gfiles) == false ||
      ifiles.empty() || ifiles.size() != gfiles.size())
  {
    return false;
  }

  QProgressDialog progress("Loading files...", "Abort", 0, ifiles.size());
  progress.setWindowModality(Qt::WindowModal);

  bob::visioner::ipscale_t ip;
  for (uint64_t i = 0; i < ifiles.size(); i ++)
  {
    if (    bob::visioner::Object::load(gfiles[i], ip.m_objects) == true &&
        bob::visioner::load(ifiles[i], ip.m_image) == true)
    {
      m_ifiles.push_back(ifiles[i]);
      m_gfiles.push_back(gfiles[i]);
      m_ipscales.push_back(ip);
    }       

    progress.setValue(i);
    if (progress.wasCanceled() == true)
    {
      break;
    }
  }
  progress.setValue(ifiles.size());

  m_listfiles.push_back(list_file);

  return rewind();
}

bool ImageCollection::rewind()
{
  if (empty() == true)
  {
    return false;
  }

  m_crt_index = 0;
  return load();
}

bool ImageCollection::next()
{
  if (m_crt_index + 1 >= size())
  {
    return false;
  }

  m_crt_index ++;
  return load();
}

bool ImageCollection::previous()
{
  if (m_crt_index == 0)
  {
    return false;
  }

  m_crt_index --;
  return load();
}

bool ImageCollection::move(std::size_t index)
{
  if (index >= size())
  {
    return false;
  }

  m_crt_index = index;
  return load();
}

bool ImageCollection::load()
{
  m_crt_ipscale = m_ipscales[m_crt_index];
  return	! (m_crt_name = bob::visioner::basename(m_ifiles[m_crt_index])).empty();
}

bool ImageCollection::pixInsideObject(int x, int y) const
{
  bool inside = false;
  for (std::vector<bob::visioner::Object>::const_iterator it = m_crt_ipscale.m_objects.begin();
      it != m_crt_ipscale.m_objects.end() && inside == false; ++ it)
  {
    inside = it->bbx().contains(x, y);
  }
  return inside;
}
