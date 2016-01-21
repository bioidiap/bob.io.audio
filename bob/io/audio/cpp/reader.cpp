/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @author Elie Khoury <elie.khoury@idiap.ch>
 * @date Tue Nov 12 11:54:30 CET 2013
 *
 * @brief A class to help you read audio. This code is based on Sox
 */

#include "reader.h"
#include "utils.h"

#include <stdexcept>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>

#include <bob.io.base/blitz_array.h>


bob::io::audio::Reader::Reader(const char* filename) {
  open(filename);
}

void bob::io::audio::Reader::open(const char* filename) {

  m_filename = filename;

  if (!boost::filesystem::exists(filename)) {
    boost::format m("file `%s' does not exist or cannot be read");
    m % filename;
    throw std::runtime_error(m.str());
  }

  //reset previously open file
  m_file.reset();

  sox_format_t* f = sox_open_read(filename, 0, 0, 0);

  if (!f) {
    boost::format m("file `%s' is not readable by SoX (internal call to `sox_open_read()' failed)");
    m % filename;
    throw std::runtime_error(m.str());
  }

  //protects file pointer
  m_file = boost::shared_ptr<sox_format_t>(f, std::ptr_fun(bob::io::audio::close_sox_file));
  m_offset = m_file->tell_off; ///< start of stream

  // Set typeinfo variables
  m_typeinfo.dtype = bob::io::base::array::t_float64;
  m_typeinfo.nd = 2;
  m_typeinfo.shape[0] = this->numberOfChannels();
  m_typeinfo.shape[1] = this->numberOfSamples();
  m_typeinfo.update_strides();

  // Pointer to a single sample that is re-used for readouts
  m_buffer = boost::shared_array<sox_sample_t>(new sox_sample_t[m_typeinfo.shape[0]]);
}


bob::io::audio::Reader::~Reader() {
}


size_t bob::io::audio::Reader::load(blitz::Array<double,2>& data,
    void (*check)(void)) {
  bob::io::base::array::blitz_array tmp(data);
  return load(tmp, check);
}


void bob::io::audio::Reader::reset() {
  sox_seek(this->m_file.get(), this->m_offset, SOX_SEEK_SET);
  //force re-open if necessary
  if ((size_t)m_file->tell_off != m_offset) open(m_filename.c_str());
}


size_t bob::io::audio::Reader::load(bob::io::base::array::interface& b,
    void (*check)(void)) {

  //checks if the output array shape conforms to the audio specifications,
  //otherwise, throw.
  if (!m_typeinfo.is_compatible(b.type())) {
    boost::format s("input buffer (%s) does not conform to the audio stream size specifications (%s)");
    s % b.type().str() % m_typeinfo.str();
    throw std::runtime_error(s.str());
  }

  //now we copy from one container to the other, using our Blitz++ technique
  blitz::TinyVector<int,2> shape;
  blitz::TinyVector<int,2> stride;

  shape = m_typeinfo.shape[0], m_typeinfo.shape[1];
  stride = m_typeinfo.stride[0], m_typeinfo.stride[1];

  blitz::Array<double,2> dst(static_cast<double*>(b.ptr()), shape, stride, blitz::neverDeleteData);

  int nchan = this->numberOfChannels();
  int nsamp = this->numberOfSamples();
  for (int i=0; i<nsamp; ++i) {
    if (check) check(); ///< runs user check function before we start our work
    sox_read(m_file.get(), m_buffer.get(), nchan);
    for (int j=0; j<nchan; ++j) {
      dst(j,i) = m_buffer[j] / bob::io::audio::SOX_CONVERSION_COEF;
    }
  }

  this->reset();
  return nsamp;
}
