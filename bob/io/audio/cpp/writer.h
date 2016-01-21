/**
 * @date Wed 20 Jan 2016 12:43:32 CET
 * @author Elie Khoury <elie.khoury@idiap.ch>
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_IO_AUDIO_WRITER_H
#define BOB_IO_AUDIO_WRITER_H

#include <map>
#include <string>
#include <blitz/array.h>

#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>

#include <bob.io.base/array.h>

extern "C" {
#include <sox.h>
}


namespace bob { namespace io { namespace audio {

  /* Until we can get a better handle (requires C++-11 initializers) */
  extern const std::map<std::string, std::string> SUPPORTED_FORMATS;

  /**
   * Use objects of this class to create and write audio files.
   */
  class Writer {

    public:

      /**
       * Default constructor, creates a new output file given the input
       * parameters. The codec to be used will be derived from the filename
       * extension.
       *
       * @param filename The name of the file that will contain the video
       * @param rate The number of samples per second
       * @param encoding The codec to use
       * @param bits_per_sample What-it-says...
       */
      Writer(const char* filename, double rate=8000,
          sox_encoding_t encoding=SOX_ENCODING_UNKNOWN,
          size_t bits_per_sample=16);

      /**
       * Destructor virtualization
       */
      virtual ~Writer();

      /**
       * Closes the current audio stream and forces writing the trailer. After
       * this point the audio becomes invalid.
       */
      void close();

      /**
       * Returns the name of the file I'm reading
       */
      inline const char* filename() const { return m_filename.c_str(); }

      /**
       * Returns the sampling rate of the audio stream.
       */
      inline double rate() const { return m_file->signal.rate; }

      /**
       * Returns the number of channels.
       */
      inline size_t numberOfChannels() const {
        return m_file->signal.channels;
      }

      /**
       * Returns the number of channels.
       */
      inline size_t bitsPerSample() const {
        return m_file->signal.precision;
      }

      /**
       * Returns the number of samples.
       */
      inline size_t numberOfSamples() const {
        return m_file->signal.length/this->numberOfChannels();
      }

      /**
       * Duration of the audio stream, in seconds
       */
      inline double duration() const {
        return this->numberOfSamples()/this->rate();
      }

      /**
       * Returns the encoding name
       */
      inline sox_encoding_t encoding() const {
        return m_file->encoding.encoding;
      }

      /**
       * Returns the compression factor
       */
      inline double compressionFactor() const {
        return m_file->encoding.compression;
      }

      /**
       * Returns the typing information for the audio file
       */
      inline const bob::io::base::array::typeinfo& type() const
      { return m_typeinfo; }

      /**
       * Returns if the video is currently opened for writing
       */
      inline bool is_opened() const { return m_opened; }

      /**
       * Writes a set of samples to the file. The sample set should be setup as
       * a blitz::Array<> with either 1 or 2 dimensions.
       *
       * \warning At present time we only support arrays that have C-style
       * storages (if you pass reversed arrays or arrays with Fortran-style
       * storage, the result is undefined).
       */
      void append(const blitz::Array<double,1>& data);
      void append(const blitz::Array<double,2>& data);

      /**
       * Writes a set of frames to the file.
       */
      void append(const bob::io::base::array::interface& b);

    private:

      /**
       * Not implemented
       */
      Writer(const Writer& other);

      /**
       * Not implemented
       */
      Writer& operator=(const Writer& other);


    private: //representation

      std::string m_filename; ///< the name of the file we are manipulating
      bob::io::base::array::typeinfo m_typeinfo; ///< read the audio type
      boost::shared_ptr<sox_format_t> m_file; ///< output file
      boost::shared_array<sox_sample_t> m_buffer; ///< buffer for writing
      bool m_opened; ///< is the file currently opened?

  };

}}}

#endif /* BOB_IO_AUDIO_WRITER_H */
