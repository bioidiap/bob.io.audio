/**
 * @author Elie Khoury <elie.khoury@idiap.ch>
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed 20 Jan 2016 11:19:53 CET
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_IO_AUDIO_UTILS_H
#define BOB_IO_AUDIO_UTILS_H

#include <string>

extern "C" {
#include <sox.h>
}

namespace bob { namespace io { namespace audio {

  extern const double SOX_CONVERSION_COEF;

  void close_sox_file(sox_format_t* f);

  const char* encoding2string(sox_encoding_t encoding);

  sox_encoding_t string2encoding(const char* encoding);

}}}

#endif /* BOB_IO_AUDIO_UTILS_H */
