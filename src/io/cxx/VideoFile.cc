/**
 * @file io/cxx/VideoFile.cc
 * @date Wed Oct 26 17:11:16 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Implements an image format reader/writer using ffmpeg.
 * This codec will only be able to work with 4D input and output (color videos)
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

#include <set>

#include <boost/filesystem.hpp>
#include <boost/make_shared.hpp>

#include "bob/core/blitz_array.h"

#include "bob/io/CodecRegistry.h"
#include "bob/io/VideoReader.h"
#include "bob/io/VideoWriter.h"
#include "bob/io/VideoUtilities.h"

extern "C" {
#include <libavformat/avformat.h>
}
namespace fs = boost::filesystem;
namespace io = bob::io;
namespace ca = bob::core::array;

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

    virtual const ca::typeinfo& type_all() const {
      return (m_reader)? m_reader->video_type() : m_writer->video_type();
    }

    virtual const ca::typeinfo& type() const {
      return (m_reader)? m_reader->video_type() : m_writer->video_type();
    }

    virtual size_t size() const {
      return (m_reader)? 1:(!m_newfile);
    }

    virtual const std::string& name() const {
      return s_codecname;
    }

    virtual void read_all(ca::interface& buffer) {
      read(buffer, 0); ///we only have 1 video in a video file anyways
    }

    virtual void read(ca::interface& buffer, size_t index) {

      if (index != 0) 
        throw std::runtime_error("can only read all frames at once in video codecs");

      if (!m_reader)
        throw std::runtime_error("can only read if opened video in 'r' mode");

      if(!buffer.type().is_compatible(m_reader->video_type())) 
        buffer.set(m_reader->video_type());

      m_reader->load(buffer);
    }

    virtual size_t append (const ca::interface& buffer) {

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

    virtual void write (const ca::interface& buffer) {

      append(buffer);

    }

  private: //representation
    std::string m_filename;
    bool m_newfile;
    boost::shared_ptr<io::VideoReader> m_reader;
    boost::shared_ptr<io::VideoWriter> m_writer;

    static std::string s_codecname;

};

std::string VideoFile::s_codecname = "bob.video";

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
 * Arranges a listing of input and output file formats
 */
static void list_formats(std::map<std::string, std::string>& formats) {
  std::map<std::string, AVInputFormat*> iformat;
  io::detail::ffmpeg::iformats_installed(iformat);
  std::map<std::string, AVOutputFormat*> oformat;
  io::detail::ffmpeg::oformats_installed(oformat);

  for (auto k=iformat.begin(); k!=iformat.end(); ++k) {
    auto o=oformat.find(k->first);
    if (o!=oformat.end()) {
      //format can be used for input and output
      std::vector<std::string> extensions;
      io::detail::ffmpeg::tokenize_csv(o->second->extensions, extensions);
      for (auto e=extensions.begin(); e!=extensions.end(); ++e) {
        std::string key = ".";
        key += *e;
        std::string value = k->second->long_name;
        value += " (video/ffmpeg)";
        formats[key] = value;
      }
    }
  }
}

/**
 * Takes care of codec registration per se.
 */
static bool register_codec() {
  static std::string tmp[] = {
    ".bmp",
    ".dpx",
    ".jpeg", 
    ".jpg", 
    ".jp2", 
    ".ljpg", 
    ".png", 
    ".pam",
    ".pcx",
    ".pbm",
    ".pnm",
    ".ppm",
    ".pgm",
    ".pgmyuv",
    ".sgi",
    ".tga", 
    ".tif", 
    ".tiff"
  };
  static std::set<std::string> avoid(tmp, tmp+sizeof(tmp) / sizeof(tmp[0]));

  /* Initialize libavcodec, and register all codecs and formats. */
  av_register_all();
  av_log_set_level(-1);

  boost::shared_ptr<io::CodecRegistry> instance =
    io::CodecRegistry::instance();

  std::map<std::string, std::string> formats;
  list_formats(formats);
  for (auto k=formats.begin(); k!=formats.end(); ++k) {
    if (!instance->isRegistered(k->first) && avoid.find(k->first) == avoid.end()) {
      instance->registerExtension(k->first, k->second, &make_file);
    }
  }

  return true;
}

static bool codec_registered = register_codec();
