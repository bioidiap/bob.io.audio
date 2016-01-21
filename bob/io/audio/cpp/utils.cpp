/**
 * @date Thu Nov 14 20:46:52 CET 2013
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief A class for some utilities using sox.
 */

#include "utils.h"
#include <map>

const double bob::io::audio::SOX_CONVERSION_COEF=2147483648.; /* 2^31 */

void bob::io::audio::close_sox_file(sox_format_t* f) {
  sox_close(f);
}

//requires c++11 for compiling
static const std::map<sox_encoding_t, std::string> ENC2STR = {
  {SOX_ENCODING_SIGN2, "SIGN2"},
  {SOX_ENCODING_UNSIGNED, "UNSIGNED"},
  {SOX_ENCODING_FLOAT, "FLOAT"},
  {SOX_ENCODING_FLOAT_TEXT, "FLOAT_TEXT"},
  {SOX_ENCODING_FLAC, "FLAC"},
  {SOX_ENCODING_HCOM, "HCOM"},
  {SOX_ENCODING_WAVPACK, "WAVPACK"},
  {SOX_ENCODING_WAVPACKF, "WAVPACKF"},
  {SOX_ENCODING_ULAW, "ULAW"},
  {SOX_ENCODING_ALAW, "ALAW"},
  {SOX_ENCODING_G721, "G721"},
  {SOX_ENCODING_G723, "G723"},
  {SOX_ENCODING_CL_ADPCM, "CL_ADPCM"},
  {SOX_ENCODING_CL_ADPCM16, "CL_ADPCM16"},
  {SOX_ENCODING_MS_ADPCM, "MS_ADPCM"},
  {SOX_ENCODING_IMA_ADPCM, "IMA_ADPCM"},
  {SOX_ENCODING_OKI_ADPCM, "OKI_ADPCM"},
  {SOX_ENCODING_DPCM, "DPCM"},
  {SOX_ENCODING_DWVW, "DWVW"},
  {SOX_ENCODING_DWVWN, "DWVWN"},
  {SOX_ENCODING_GSM, "GSM"},
  {SOX_ENCODING_MP3, "MP3"},
  {SOX_ENCODING_VORBIS, "VORBIS"},
  {SOX_ENCODING_AMR_WB, "AMR_WB"},
  {SOX_ENCODING_AMR_NB, "AMR_NB"},
  {SOX_ENCODING_CVSD, "CVSD"},
  {SOX_ENCODING_LPC10, "LPC10"},
  {SOX_ENCODING_UNKNOWN, "UNKNOWN"}
};

const char* bob::io::audio::encoding2string(sox_encoding_t e) {
  auto it = ENC2STR.find(e);
  if (it != ENC2STR.end()) return it->second.c_str();
  return ENC2STR.rbegin()->second.c_str(); //last entry: UNKNOWN
}

//requires c++11 for compiling
static const std::map<std::string, sox_encoding_t> STR2ENC = {
  {"SIGN2", SOX_ENCODING_SIGN2},
  {"UNSIGNED", SOX_ENCODING_UNSIGNED},
  {"FLOAT", SOX_ENCODING_FLOAT},
  {"FLOAT_TEXT", SOX_ENCODING_FLOAT_TEXT},
  {"FLAC", SOX_ENCODING_FLAC},
  {"HCOM", SOX_ENCODING_HCOM},
  {"WAVPACK", SOX_ENCODING_WAVPACK},
  {"WAVPACKF", SOX_ENCODING_WAVPACKF},
  {"ULAW", SOX_ENCODING_ULAW},
  {"ALAW", SOX_ENCODING_ALAW},
  {"G721", SOX_ENCODING_G721},
  {"G723", SOX_ENCODING_G723},
  {"CL_ADPCM", SOX_ENCODING_CL_ADPCM},
  {"CL_ADPCM16", SOX_ENCODING_CL_ADPCM16},
  {"MS_ADPCM", SOX_ENCODING_MS_ADPCM},
  {"IMA_ADPCM", SOX_ENCODING_IMA_ADPCM},
  {"OKI_ADPCM", SOX_ENCODING_OKI_ADPCM},
  {"DPCM", SOX_ENCODING_DPCM},
  {"DWVW", SOX_ENCODING_DWVW},
  {"DWVWN", SOX_ENCODING_DWVWN},
  {"GSM", SOX_ENCODING_GSM},
  {"MP3", SOX_ENCODING_MP3},
  {"VORBIS", SOX_ENCODING_VORBIS},
  {"AMR_WB", SOX_ENCODING_AMR_WB},
  {"AMR_NB", SOX_ENCODING_AMR_NB},
  {"CVSD", SOX_ENCODING_CVSD},
  {"LPC10", SOX_ENCODING_LPC10},
  {"UNKNOWN", SOX_ENCODING_UNKNOWN}
};

sox_encoding_t bob::io::audio::string2encoding(const char* s) {
  auto it = STR2ENC.find(s);
  if (it != STR2ENC.end()) return it->second;
  return STR2ENC.rbegin()->second; //last entry: UNKNOWN
}
