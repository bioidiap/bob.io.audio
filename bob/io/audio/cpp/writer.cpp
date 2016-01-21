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

/* Until we can get a better handle (requires C++-11 initializers) */
const std::map<std::string, std::string> bob::io::audio::SUPPORTED_FORMATS = {
  {".aifc", "AIFF-C (not compressed), defined in DAVIC 1.4 Part 9 Annex B"},
  {".aiff", "AIFF files used on Apple IIc/IIgs and SGI"},
  {".al", "Raw audio"},
  {".au", "PCM file format used widely on Sun systems"},
  {".avr", "Audio Visual Research format; used on the Mac"},
  {".cdda", "Red Book Compact Disc Digital Audio"},
  {".cvsd", "Headerless MIL Std 188 113 Continuously Variable Slope Delta modulation"},
  {".cvu", "Headerless Continuously Variable Slope Delta modulation (unfiltered)"},
  {".dat", "Textual representation of the sampled audio"},
  {".dvms", "MIL Std 188 113 Continuously Variable Slope Delta modulation with header"},
  {".f4", "Raw audio"},
  {".f8", "Raw audio"},
  {".gsrt", "Grandstream ring tone"},
  {".hcom", "Mac FSSD files with Huffman compression"},
  {".htk", "PCM format used for Hidden Markov Model speech processing"},  {".ima", "Raw IMA ADPCM"},
  {".la", "Raw audio"},
  {".lu", "Raw audio"},
  {".maud", "Used with the ‘Toccata’ sound-card on the Amiga"},
  {".prc", "Psion Record; used in EPOC devices (Series 5, Revo and similar)"},
  {".raw", "Raw PCM, mu-law, or A-law"},
  {".s1", "Raw audio"},
  {".s2", "Raw audio"},
  {".s3", "Raw audio"},
  {".s4", "Raw audio"},
  {".sf", "Institut de Recherche et Coordination Acoustique/Musique"},
  {".sln", "Asterisk PBX headerless format"},
  {".smp", "Turtle Beach SampleVision"},
  {".sndr", "8-bit linear audio as used by Aaron Wallace's `Sounder' of 1991"},
  {".sndt", "8-bit linear audio as used by Martin Hepperle's `SoundTool' of 1991/2"},
  {".sox", "SoX native intermediate format"},
  {".sph", "SPeech HEader Resources; defined by NIST"},
  {".8svx", "Amiga audio format (a subformat of the Interchange File Format)"},
  {".txw", "Yamaha TX-16W sampler"},
  {".u1", "Raw audio"},
  {".u2", "Raw audio"},
  {".u3", "Raw audio"},
  {".u4", "Raw audio"},
  {".ul", "Raw audio"},
  {".voc", "Creative Technology Sound Blaster format"},
  {".vox", "Raw OKI/Dialogic ADPCM"},
  {".wav", "Microsoft audio format"},
  {".wve", "Psion 3 audio format"},
  {".xa", "16-bit ADPCM audio files used by Maxis games"}
};

bob::io::audio::Writer::Writer(const char* filename, double rate,
    sox_encoding_t encoding, size_t bits_per_sample):
  m_filename(filename),
  m_opened(false)
{

  sox_signalinfo_t siginfo;
  siginfo.rate = rate;
  siginfo.precision = bits_per_sample;
  siginfo.channels = SOX_UNSPEC;
#ifdef SOX_UNKNOWN_LEN
  siginfo.length = SOX_UNKNOWN_LEN;
#else
  siginfo.length = -1;
#endif

  const char* extension = lsx_find_file_extension(filename);

  if (bob::io::audio::SUPPORTED_FORMATS.find(extension-1) ==
      bob::io::audio::SUPPORTED_FORMATS.end()) { //unsupported file format
    boost::format m("file `%s' cannot be written by SoX (file format `%d' is unsupported -- use `sox --help-format all' for a list of all supported formats");
    m % filename % extension;
    throw std::runtime_error(m.str());
  }

  sox_format_t* f = 0;
  if (encoding == SOX_ENCODING_UNKNOWN) {
    f = sox_open_write(filename, &siginfo, 0, extension, 0, 0);
  }
  else {
    sox_encodinginfo_t encoding_info;
    encoding_info.encoding = encoding;
    encoding_info.bits_per_sample = bits_per_sample;
    encoding_info.compression = HUGE_VAL;
    f = sox_open_write(filename, &siginfo, &encoding_info, 0, 0, 0);
  }

  if (!f) {
    boost::format m("file `%s' is not writeable by SoX (internal call to `sox_open_write()' failed) -- we suggest you check writing permissions and existence of leading paths");
    m % filename;
    throw std::runtime_error(m.str());
  }

  m_file = boost::shared_ptr<sox_format_t>(f, std::ptr_fun(bob::io::audio::close_sox_file));

  m_typeinfo.dtype = bob::io::base::array::t_float64;
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
  if (m_typeinfo.shape[0] != (size_t)data.extent(0)) {
    boost::format m("input sample size for file `%s' should be (%d,)");
    m % m_filename % m_typeinfo.shape[0];
    throw std::runtime_error(m.str());
  }

  for (int j=0; j<data.extent(0); ++j)
    m_buffer[j] = (sox_sample_t)(data(j) * bob::io::audio::SOX_CONVERSION_COEF);
  size_t written = sox_write(m_file.get(), m_buffer.get(), m_typeinfo.shape[0]);

  // updates internal counters
  m_file->signal.length += m_file->signal.channels;
  m_typeinfo.shape[1] += 1;
  m_typeinfo.update_strides();

  if (written != 1) {
    boost::format m("I was asked to append 1 sample to file `%s', but `sox_write()' failed miserably - this is not a definitive error, the stream is still sane");
    m % m_filename;
    throw std::runtime_error(m.str());
  }
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
  if (m_typeinfo.shape[0] != (size_t)data.extent(0)) {
    boost::format m("input sample size for file `%s' should have %d rows");
    m % m_filename % m_typeinfo.shape[0];
    throw std::runtime_error(m.str());
  }

  size_t written = 0;
  for (int i=0; i<data.extent(1); i++) {
    for (int j=0; j<data.extent(0); ++j) {
      m_buffer[j] = (sox_sample_t)(data(j, i) * bob::io::audio::SOX_CONVERSION_COEF);
    }
    written += sox_write(m_file.get(), m_buffer.get(), m_typeinfo.shape[0]);
  }

  // updates internal counters
  m_file->signal.length += written * m_file->signal.channels;
  m_typeinfo.shape[1] += written;
  m_typeinfo.update_strides();

  if (written != (size_t)data.extent(1)) {
    boost::format m("I was asked to append %d samples to file `%s', but `sox_write()' managed to append %d samples only - this is not a definitive error, the stream is still sane");
    m % data.extent(1) % m_filename % written;
    throw std::runtime_error(m.str());
  }
}

void bob::io::audio::Writer::append(const bob::io::base::array::interface& data) {

  if (!m_opened) {
    boost::format m("audio writer for file `%s' is closed and cannot be written to");
    m % m_filename;
    throw std::runtime_error(m.str());
  }

  const bob::io::base::array::typeinfo& type = data.type();

  if (type.dtype != bob::io::base::array::t_float64) {
    boost::format m("input data type = `%s' does not conform to the specified input specifications (1 or 2D array of type `%s'), while writing data to file `%s'");
    m % type.str() % m_typeinfo.item_str() % m_filename;
    throw std::runtime_error(m.str());
  }

  if (type.nd == 1) { //appends single sample
    blitz::TinyVector<int,1> shape;
    shape = type.shape[0];
    blitz::Array<double,1> tmp(const_cast<double*>(static_cast<const double*>(data.ptr())), shape, blitz::neverDeleteData);
    this->append(tmp);
  }

  else if (type.nd == 2) { //appends multiple frames
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
}

void bob::io::audio::Writer::close() {
  if (!m_opened) return;
  m_file.reset();
  m_opened = false; ///< file is now considered closed
}
