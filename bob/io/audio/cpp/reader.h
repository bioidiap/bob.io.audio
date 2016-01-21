/**
 * @date Wed 20 Jan 2016 11:23:10 CET
 * @author Elie Khoury <elie.khoury@idiap.ch>
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_IO_AUDIO_READER_H
#define BOB_IO_AUDIO_READER_H

#include <string>

#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>
#include <blitz/array.h>

#include <bob.io.base/array.h>

extern "C" {
#include <sox.h>
}


namespace bob { namespace io { namespace audio {

  /**
   * Reader objects can read data from audio files. The current implementation
   * uses SOX which is a stable freely available implementation for these
   * tasks. You can read an entire audio in memory by using the "load()"
   * method.
   */
  class Reader {

    public:

      /**
       * Opens a new SOX stream for reading. The audio will be loaded if the
       * combination of format and codec are known to work and have been
       * tested, otherwise an exception is raised. If you set 'check' to
       * 'false', though, we will ignore this check.
       */
      Reader(const char* filename);

      /**
       * Destructor virtualization
       */
      virtual ~Reader();

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
       * Returns the typing information for this audio file
       */
      inline const bob::io::base::array::typeinfo& type() const
      { return m_typeinfo; }

      /**
       * Loads all of the audio stream in a blitz array organized in this way:
       * (rate, samples). The 'data' parameter will be
       * resized if required.
       *
       * The check function is used to hook a signal handler and stop
       * processing if required by the user.
       */
      size_t load(blitz::Array<double,2>& data, void (*check)(void)=0);

      /**
       * Loads all of the audio stream in a buffer. Resizes the buffer if
       * the space and type are not good.
       */
      size_t load(bob::io::base::array::interface& b, void (*check)(void)=0);

      /**
      * Closes the file
      */
      void reset();


    private: //methods

      /**
       * Not implemented
       */
      Reader(const Reader& other);

      /**
       * Not implemented
       */
      Reader& operator=(const Reader& other);

      /**
       * Opens the previously set up SOX stream for the reader
       */
      void open(const char* filename);


    private: //our representation

      std::string m_filename; ///< the name of the file we are manipulating
      bob::io::base::array::typeinfo m_typeinfo; ///< read the audio type
      boost::shared_ptr<sox_format_t> m_file; ///< input file
      boost::shared_array<sox_sample_t> m_buffer; ///< buffer for readout
      uint64_t m_offset; ///< start of stream
  };

}}}

#endif /* BOB_IO_AUDIO_READER_H */
