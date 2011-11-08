/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed 26 Oct 09:57:00 2011 CEST
 *
 * @brief Implements an image format reader/writer using ffmpeg.
 *
 * This codec will only be able to work with 4D input and output (color videos)
 */

#include <boost/filesystem.hpp>
#include <boost/make_shared.hpp>

#include "core/blitz_array.h"

#include "io/CodecRegistry.h"
#include "io/Video.h"

namespace fs = boost::filesystem;
namespace io = Torch::io;
namespace ca = Torch::core::array;

class VideoFile: public io::File {

  public: //api

    VideoFile(const std::string& path, char mode):
      m_filename(path),
      m_newfile(true) {

        if (mode == 'r') {
          m_reader = boost::make_shared<io::VideoReader>(m_filename);
          m_newfile = false;
        }
        else if (mode == 'a' && fs::exists(path)) {
          // to be able to append must load all data and save in VideoWriter
          m_reader = boost::make_shared<io::VideoReader>(m_filename);
          ca::blitz_array data(m_reader->video_type());
          m_reader->load(data);
          size_t height = m_reader->height();
          size_t width = m_reader->width();
          m_reader.reset(); ///< cleanup before truncating the file
          m_writer =
            boost::make_shared<io::VideoWriter>(m_filename, height, width);
          m_writer->append(data); ///< we are now ready to append
          m_newfile = false;
        }
        else { //mode is 'w'
          m_newfile = true;
        }

      }

    virtual ~VideoFile() { }

    virtual const std::string& filename() const {
      return m_filename;
    }

    virtual const ca::typeinfo& array_type() const {
      return (m_reader)? m_reader->video_type() : m_writer->video_type();
    }

    virtual const ca::typeinfo& arrayset_type() const {
      return (m_reader)? m_reader->video_type() : m_writer->video_type();
    }

    virtual size_t arrayset_size() const {
      return (m_reader)? 1:(!m_newfile);
    }

    virtual const std::string& name() const {
      return s_codecname;
    }

    virtual void array_read(ca::interface& buffer) {
      arrayset_read(buffer, 0); ///we only have 1 video in a video file anyways
    }

    virtual void arrayset_read(ca::interface& buffer, size_t index) {

      if (index != 0) 
        throw std::runtime_error("can only read all frames at once in video codecs");

      if (!m_reader)
        throw std::runtime_error("can only read if opened video in 'r' mode");

      if(!buffer.type().is_compatible(m_reader->video_type())) 
        buffer.set(m_reader->video_type());

      m_reader->load(buffer);
    }

    virtual size_t arrayset_append (const ca::interface& buffer) {

      const ca::typeinfo& type = buffer.type();
  
      if (type.nd != 3 and type.nd != 4)
        throw std::invalid_argument("input buffer for videos must have 3 or 4 dimensions");

      if(m_newfile) {
        size_t height = (type.nd==3)? type.shape[1]:type.shape[2];
        size_t width  = (type.nd==3)? type.shape[2]:type.shape[3];
        m_writer = boost::make_shared<io::VideoWriter>(m_filename, height, width);
      }

      if(!m_writer)
        throw std::runtime_error("can only read if open video in 'a' or 'w' modes");

      m_writer->append(buffer);
      return 1;
    }

    virtual void array_write (const ca::interface& buffer) {

      arrayset_append(buffer);

    }

  private: //representation
    std::string m_filename;
    bool m_newfile;
    boost::shared_ptr<io::VideoReader> m_reader;
    boost::shared_ptr<io::VideoWriter> m_writer;

    static std::string s_codecname;

};

std::string VideoFile::s_codecname = "torch.video";

/**
 * From this point onwards we have the registration procedure. If you are
 * looking at this file for a coding example, just follow the procedure bellow,
 * minus local modifications you may need to apply.
 */

/**
 * This defines the factory method F that can create codecs of this type.
 * 
 * Here are the meanings of the mode flag that should be respected by your
 * factory implementation:
 *
 * 'r': opens for reading only - no modifications can occur; it is an
 *      error to open a file that does not exist for read-only operations.
 * 'w': opens for reading and writing, but truncates the file if it
 *      exists; it is not an error to open files that do not exist with
 *      this flag. 
 * 'a': opens for reading and writing - any type of modification can 
 *      occur. If the file does not exist, this flag is effectively like
 *      'w'.
 *
 * Returns a newly allocated File object that can read and write data to the
 * file using a specific backend.
 *
 * @note: This method can be static.
 */
static boost::shared_ptr<io::File> 
make_file (const std::string& path, char mode) {

  return boost::make_shared<VideoFile>(path, mode);

}

/**
 * Takes care of codec registration per se.
 */
static bool register_codec() {

  boost::shared_ptr<io::CodecRegistry> instance =
    io::CodecRegistry::instance();
  
  instance->registerExtension(".avi", &make_file);
  instance->registerExtension(".dv", &make_file);
  instance->registerExtension(".filmstrip", &make_file);
  instance->registerExtension(".flv", &make_file);
  instance->registerExtension(".h261", &make_file);
  instance->registerExtension(".h263", &make_file);
  instance->registerExtension(".h264", &make_file);
  instance->registerExtension(".mov", &make_file);
  instance->registerExtension(".image2", &make_file);
  instance->registerExtension(".image2pipe", &make_file);
  instance->registerExtension(".m4v", &make_file);
  instance->registerExtension(".mjpeg", &make_file);
  instance->registerExtension(".mpeg", &make_file);
  instance->registerExtension(".mpegts", &make_file);
  instance->registerExtension(".ogg", &make_file);
  instance->registerExtension(".rawvideo", &make_file);
  instance->registerExtension(".rm", &make_file);
  instance->registerExtension(".rtsp", &make_file);
  instance->registerExtension(".yuv4mpegpipe", &make_file);

  return true;

}

static bool codec_registered = register_codec();
