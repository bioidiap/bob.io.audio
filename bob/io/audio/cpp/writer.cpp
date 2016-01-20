/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @author Elie Khoury <elie.khoury@idiap.ch>
 * @date Wed 20 Jan 2016 17:18:05 CET
 *
 * @brief A class to help you write audio.
 */

#include "utils.h"
#include "writer.h"

#include <stdexcept>
#include <boost/format.hpp>

bob::io::audio::Writer::Writer(const char* filename, double rate,
    sox_encoding_t encoding, size_t bits_per_sample):
  m_filename(filename),
  m_opened(false)
{

  sox_signalinfo_t siginfo;
  siginfo.rate = rate;
  siginfo.precision = bits_per_sample;
  siginfo.channels = 0;
  siginfo.length = 0;

  sox_encodinginfo_t encoding_info;
  encoding_info.encoding = encoding;
  encoding_info.bits_per_sample = bits_per_sample;
  encoding_info.compression = HUGE_VAL;

#if SOX_LIB_VERSION_CODE >= SOX_LIB_VERSION(14,4,0)
  encoding_info.reverse_bytes = sox_option_default;
  encoding_info.reverse_nibbles = sox_option_default;
  encoding_info.reverse_bits = sox_option_default;
#else
  encoding_info.reverse_bytes = SOX_OPTION_DEFAULT;
  encoding_info.reverse_nibbles = SOX_OPTION_DEFAULT;
  encoding_info.reverse_bits = SOX_OPTION_DEFAULT;
#endif

  sox_format_t* f = 0;
  if (encoding == SOX_ENCODING_UNKNOWN) f = sox_open_write(filename, &siginfo, 0, lsx_find_file_extension(filename), 0, 0);
  else f = sox_open_write(filename, &siginfo, &encoding_info, 0, 0, 0);

  if (!f) {
    boost::format m("file '%s' is not writeable by SoX");
    m % filename;
    throw std::runtime_error(m.str());
  }

  m_file = boost::shared_ptr<sox_format_t>(f, std::ptr_fun(bob::io::audio::close_sox_file));

  m_typeinfo.dtype = bob::core::array::t_float64;
  m_typeinfo.nd = 2;
  m_typeinfo.shape[0] = 0;
  m_typeinfo.shape[1] = 0;
  m_typeinfo.update_strides();

  m_buffer = boost::shared_array<sox_sample_t>(new sox_sample_t[m_typeinfo.shape[0]]);

  m_opened = true; ///< file is now considered opened for business
}


void bob::io::audio::Writer::append(const blitz::Array<double,1>& data) {

  if (!m_opened) {
    boost::format m("audio writer for file `%s' is closed and cannot be written to");
    m % m_filename;
    throw std::runtime_error(m.str());
  }

  if (!m_typeinfo.shape[0]) /* set for the first time */ {
    m_file->signal.channels = data.extent(0);
    m_typeinfo.shape[0] = data.extent(0);
    m_typeinfo.update_strides();
  }

  //checks data specifications
  if (m_typeinfo.shape[0] != data.extent(0)) {
    boost::format m("input sample size for file `%s' should be (%d,)");
    m % m_filename % m_typeinfo.shape[0];
    throw std::runtime_error(m.str());
  }

  for (int j=0; j<data.extent(0); ++j)
    m_buffer[j] = (sox_sample_t)(data(j) * bob::io::audio::SOX_CONVERSION_COEF);
  sox_write(m_file.get(), m_buffer.get(), m_typeinfo.shape[0]);

  // updates internal counters
  m_signal_cache.length += m_file->signal.channels;
  m_typeinfo.shape[1] += 1;
  m_typeinfo.update_strides();
}


void bob::io::audio::Writer::append(const blitz::Array<double,2>& data) {

  if (!m_opened) {
    boost::format m("audio writer for file `%s' is closed and cannot be written to");
    m % m_filename;
    throw std::runtime_error(m.str());
  }

  if (!m_typeinfo.shape[0]) /* set for the first time */ {
    m_file->signal.channels = data.extent(0);
    m_typeinfo.shape[0] = data.extent(0);
    m_typeinfo.update_strides();
  }

  //checks data specifications
  if (m_typeinfo.shape[0] != data.extent(0)) {
    boost::format m("input sample size for file `%s' should have %d rows");
    m % m_filename % m_typeinfo.shape[0];
    throw std::runtime_error(m.str());
  }

  for (int i=0; i<data.extent(1); i++) {
    for (int j=0; j<data.extent(0); ++j)
      m_buffer[j] = (sox_sample_t)(data(j, i) * bob::io::audio::SOX_CONVERSION_COEF);
    sox_write(m_file.get(), m_buffer.get(), m_typeinfo.shape[0]);
  }

  // updates internal counters
  m_signal_cache.length += data.extent(1) * m_file->signal.channels;
  m_typeinfo.shape[1] += data.extent(1);
  m_typeinfo.update_strides();
}

void bob::io::audio::Writer::append(const bob::io::base::array::interface& data) {

  if (!m_opened) {
    boost::format m("audio writer for file `%s' is closed and cannot be written to");
    m % m_filename;
    throw std::runtime_error(m.str());
  }

  const bob::io::base::array::typeinfo& type = data.type();

  if ( type.dtype != bob::io::base::array::t_float64 ) {
    boost::format m("input data type = `%s' does not conform to the specified input specifications (1 or 2D array of type `%s'), while writing data to file `%s'");
    m % type.str() % m_typeinfo.item_str() % m_filename;
    throw std::runtime_error(m.str());
  }

  if ( type.nd == 1 ) { //appends single sample
    blitz::TinyVector<int,1> shape;
    shape = type.shape[0];
    blitz::Array<double,1> tmp(const_cast<double*>(static_cast<const double*>(data.ptr())), shape, blitz::neverDeleteData);
    this->append(tmp);
  }

  else if ( type.nd == 2 ) { //appends multiple frames
    blitz::TinyVector<int,2> shape;
    shape = type.shape[0], type.shape[1];
    blitz::Array<double,2> tmp(const_cast<double*>(static_cast<const double*>(data.ptr())), shape, blitz::neverDeleteData);
    this->append(tmp);
  }

  else {
    boost::format m("input data type information = `%s' does not conform to the specified input specifications (1 or 2D array of type = `%s'), while writing data to file `%s'");
    m % type.str() % m_typeinfo.item_str() % m_filename;
  }


}

bob::io::audio::Writer::~Writer() {
  close();
}

void bob::io::audio::Writer::close() {
  if (!m_opened) return;
  m_file.reset();
  m_opened = false; ///< file is now considered closed
}
