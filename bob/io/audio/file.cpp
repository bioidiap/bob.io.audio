/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed 20 Jan 2016 13:43:56 CET
 *
 * @brief Implements an audio format reader for Bob
 *
 * Copyright (C) 2011-2016 Idiap Research Institute, Martigny, Switzerland
 */

#include <set>

#include <boost/filesystem.hpp>
#include <boost/make_shared.hpp>

#include <bob.io.base/blitz_array.h>

#include <bob.io.base/CodecRegistry.h>
#include <bob.io.base/File.h>

#include "cpp/reader.h"
#include "cpp/writer.h"

class AudioFile: public bob::io::base::File {

  public: //api

    AudioFile(const std::string& path, char mode):
      m_filename(path),
      m_newfile(true) {

        if (mode == 'r') {
          m_reader = boost::make_shared<bob::io::audio::Reader>(m_filename.c_str());
          m_newfile = false;
        }
        else if (mode == 'a' && boost::filesystem::exists(path)) {
          // to be able to append must load all data and save in audio::Writer
          m_reader = boost::make_shared<bob::io::audio::Reader>(m_filename.c_str());
          bob::io::base::array::blitz_array data(m_reader->type());
          m_reader->load(data);
          auto rate = m_reader->rate();
          auto encoding = m_reader->encoding();
          auto bps = m_reader->bitsPerSample();
          m_reader.reset(); ///< cleanup before truncating the file
          m_writer = boost::make_shared<bob::io::audio::Writer>(m_filename.c_str(), rate, encoding, bps);
          m_writer->append(data); ///< we are now ready to append
          m_newfile = false;
        }
        else { //mode is 'w'
          m_newfile = true;
        }

      }

    virtual ~AudioFile() { }

    virtual const char* filename() const {
      return m_filename.c_str();
    }

    virtual const bob::io::base::array::typeinfo& type_all() const {
      return (m_reader)? m_reader->type() : m_writer->type();
    }

    virtual const bob::io::base::array::typeinfo& type() const {
      return (m_reader)? m_reader->type() : m_writer->type();
    }

    virtual size_t size() const {
      return (m_reader)? 1:(!m_newfile);
    }

    virtual const char* name() const {
      return s_codecname.c_str();
    }

    virtual void read_all(bob::io::base::array::interface& buffer) {
      read(buffer, 0); ///we only have 1 audio in a audio file anyways
    }

    virtual void read(bob::io::base::array::interface& buffer, size_t index) {

      if (index != 0)
        throw std::runtime_error("can only read all samples at once in audio codecs");

      if (!m_reader)
        throw std::runtime_error("can only read if opened audio in 'r' mode");

      if(!buffer.type().is_compatible(m_reader->type()))
        buffer.set(m_reader->type());

      m_reader->load(buffer);
    }

    virtual size_t append (const bob::io::base::array::interface& buffer) {

      const bob::io::base::array::typeinfo& type = buffer.type();

      if (type.nd != 1 && type.nd != 2)
        throw std::runtime_error("input buffer for audio input must have either 1 (channel values for 1 sample) or 2 dimensions (channels, samples)");

      if(m_newfile) {
        m_writer = boost::make_shared<bob::io::audio::Writer>(m_filename.c_str());
      }

      if(!m_writer)
        throw std::runtime_error("can only read if open audio in 'a' or 'w' modes");

      m_writer->append(buffer);
      return 1;
    }

    virtual void write (const bob::io::base::array::interface& buffer) {

      append(buffer);

    }

  private: //representation
    std::string m_filename;
    bool m_newfile;
    boost::shared_ptr<bob::io::audio::Reader> m_reader;
    boost::shared_ptr<bob::io::audio::Writer> m_writer;

    static std::string s_codecname;

};

std::string AudioFile::s_codecname = "bob.audio";

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
boost::shared_ptr<bob::io::base::File> make_file (const char* path, char mode) {
  return boost::make_shared<AudioFile>(path, mode);
}
