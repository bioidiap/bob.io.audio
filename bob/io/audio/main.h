/**
 * @author Manuel Guenther <siebenkopf@googlemail.com>
 * @date Wed Mar  2 18:35:11 MST 2016
 *
 * @brief Header file for bindings to bob::io::audio
 */


#ifndef BOB_IP_BASE_MAIN_H
#define BOB_IP_BASE_MAIN_H

#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>
#include <bob.core/api.h>
#include <bob.io.base/api.h>
#include <bob.extension/documentation.h>

#include "cpp/utils.h"
#include "cpp/reader.h"
#include "cpp/writer.h"

// Reader
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::io::audio::Reader> v;
} PyBobIoAudioReaderObject;

extern PyTypeObject PyBobIoAudioReader_Type;
bool init_BobIoAudioReader(PyObject* module);
int PyBobIoAudioReader_Check(PyObject* o);

#endif // BOB_IP_BASE_MAIN_H
