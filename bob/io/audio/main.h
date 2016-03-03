/**
 * @author Manuel Guenther <siebenkopf@googlemail.com>
 * @date Wed Mar  2 18:35:11 MST 2016
 *
 * @brief Header file for bindings to bob::io::audio
 */


#ifndef BOB_IO_AUDIO_MAIN_H
#define BOB_IO_AUDIO_MAIN_H

#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>
#include <bob.core/api.h>
#include <bob.io.base/api.h>
#include <bob.extension/documentation.h>

#include "cpp/utils.h"
#include "cpp/reader.h"
#include "cpp/writer.h"
#include "bobskin.h"

// Reader
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::io::audio::Reader> v;
} PyBobIoAudioReaderObject;

extern PyTypeObject PyBobIoAudioReader_Type;
bool init_BobIoAudioReader(PyObject* module);
int PyBobIoAudioReader_Check(PyObject* o);


// Writer
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::io::audio::Writer> v;
} PyBobIoAudioWriterObject;

extern PyTypeObject PyBobIoAudioWriter_Type;
bool init_BobIoAudioWriter(PyObject* module);
int PyBobIoAudioWriter_Check(PyObject* o);

#endif // BOB_IO_AUDIO_MAIN_H
