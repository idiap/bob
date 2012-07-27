#ifndef IMAGE_COLLECTION_H
#define IMAGE_COLLECTION_H

#include <boost/serialization/singleton.hpp>

#include "visioner/model/ipyramid.h"

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
    const bob::visioner::strings_t& listfiles() const { return m_listfiles; }
    const bob::visioner::strings_t& ifiles() const { return m_ifiles; }
    const bob::visioner::strings_t& gfiles() const { return m_gfiles; }

  private:

    // Load the image and the ground truth at the specified index
    bool load();

    // Attributes
    bob::visioner::strings_t	m_listfiles;		// List files
    bob::visioner::strings_t	m_ifiles, m_gfiles;	// List of image and ground truth files to process
    bob::visioner::ipscales_t    m_ipscales;             // Images

    std::size_t		m_crt_index;		// Current index in the image list
    bob::visioner::ipscale_t	m_crt_ipscale;		// Current image && ground truth (if available)
    std::string		m_crt_name;		// Current image name
};

#endif // IMAGE_COLLECTION_H
