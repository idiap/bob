/**
 * @file visioner/programs/vgui/image_collection.h
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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

#ifndef IMAGE_COLLECTION_H
#define IMAGE_COLLECTION_H

#include <boost/serialization/singleton.hpp>

#include "bob/visioner/model/ipyramid.h"

/**
 * Collection of images and ground truth, accessible to every object in the
 * project, when the data is loaded from .list files.
 */
class ImageCollection : public boost::serialization::singleton<ImageCollection>
{
  protected:

    // Constructor	
    ImageCollection();

  public:

    // Manage images
    void clear();
    bool add(const std::string& list_file);

    // Move the current index 
    bool rewind();
    bool next();
    bool previous();
    bool move(std::size_t index);

    // Checks if a pixel is inside of any object in the current image
    bool pixInsideObject(int x, int y) const;

    // Access functions
    bool empty() const { return m_ifiles.empty(); }
    std::size_t size() const { return m_ifiles.size(); }
    std::size_t index() const { return m_crt_index; }
    const bob::visioner::ipscale_t& ipscale() const { return m_crt_ipscale; }
    const std::string& name() const { return m_crt_name; }	
    const std::vector<std::string>& listfiles() const { return m_listfiles; }
    const std::vector<std::string>& ifiles() const { return m_ifiles; }
    const std::vector<std::string>& gfiles() const { return m_gfiles; }

  private:

    // Load the image and the ground truth at the specified index
    bool load();

    // Attributes
    std::vector<std::string>	m_listfiles;		// List files
    std::vector<std::string>	m_ifiles, m_gfiles;	// List of image and ground truth files to process
    std::vector<bob::visioner::ipscale_t>    m_ipscales; // Images

    std::size_t		m_crt_index;		// Current index in the image list
    bob::visioner::ipscale_t	m_crt_ipscale;		// Current image && ground truth (if available)
    std::string		m_crt_name;		// Current image name
};

#endif // IMAGE_COLLECTION_H
