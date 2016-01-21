/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed 20 Jan 2016 15:38:07 CET
 *
 * @brief Bindings to bob::io::audio::Writer
 */

#include "bobskin.h"

#include <boost/make_shared.hpp>
#include <numpy/arrayobject.h>
#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>
#include <bob.io.base/api.h>
#include <stdexcept>

#include "cpp/utils.h"
#include "cpp/writer.h"

#define AUDIOWRITER_NAME "writer"
PyDoc_STRVAR(s_audiowriter_str, BOB_EXT_MODULE_PREFIX "." AUDIOWRITER_NAME);

PyDoc_STRVAR(s_audiowriter_doc,
"writer(filename, [rate=8000., [encoding='UNKNOWN', [bits_per_sample=16]]]) -> new writer\n\
\n\
Use this object to write samples to audio files.\n\
\n\
Constructor parameters:\n\
\n\
filename\n\
  [str] The file path to the file you want to write data to\n\
\n\
rate\n\
  [float, optional] The number of samples per second\n\
\n\
encoding\n\
  [str, optional] The encoding to use\n\
\n\
bits_per_sample\n\
  [int, optional] The number of bits per sample to be recorded\n\
\n\
Audio writer objects can write data to audio files. The current\n\
implementation uses `SoX <http://sox.sourceforge.net/>`_. \n\
Audio files are objects composed potentially multiple channels.\n\
The numerical representation are 2-D arrays where the first\n\
dimension corresponds to the channels of the audio stream and\n\
the second dimension represents the samples through time.\n\
");

typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::io::audio::Writer> v;
} PyBobIoAudioWriterObject;

extern PyTypeObject PyBobIoAudioWriter_Type;

/* How to create a new PyBobIoAudioWriterObject */
static PyObject* PyBobIoAudioWriter_New(PyTypeObject* type, PyObject*, PyObject*) {

  /* Allocates the python object itself */
  PyBobIoAudioWriterObject* self = (PyBobIoAudioWriterObject*)type->tp_alloc(type, 0);

  self->v.reset();

  return reinterpret_cast<PyObject*>(self);
}

static void PyBobIoAudioWriter_Delete (PyBobIoAudioWriterObject* o) {

  o->v.reset();
  Py_TYPE(o)->tp_free((PyObject*)o);

}

/* The __init__(self) method */
static int PyBobIoAudioWriter_Init(PyBobIoAudioWriterObject* self,
    PyObject *args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {
    "filename", "rate", "encoding", "bits_per_sample",
    0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* filename = 0;
  double rate = 8000.;
  char* encoding = 0;
  Py_ssize_t bits_per_sample = 16;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|dsn", kwlist,
        &PyBobIo_FilenameConverter, &filename,
        &rate, &encoding, &bits_per_sample)) return -1;

  auto filename_ = make_safe(filename);

  std::string encoding_str = encoding?encoding:"UNKNOWN";
  sox_encoding_t sox_encoding = bob::io::audio::string2encoding(encoding_str.c_str());

#if PY_VERSION_HEX >= 0x03000000
  const char* c_filename = PyBytes_AS_STRING(filename);
#else
  const char* c_filename = PyString_AS_STRING(filename);
#endif

  try {
    self->v = boost::make_shared<bob::io::audio::Writer>(c_filename,
        rate, sox_encoding, bits_per_sample);
  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot open audio file `%s' for writing: unknown exception caught", c_filename);
    return -1;
  }

  return 0; ///< SUCCESS
}

PyObject* PyBobIoAudioWriter_Filename(PyBobIoAudioWriterObject* self) {
  return Py_BuildValue("s", self->v->filename());
}

PyDoc_STRVAR(s_filename_str, "filename");
PyDoc_STRVAR(s_filename_doc,
"[str] The full path to the file that will be decoded by this object");

PyObject* PyBobIoAudioWriter_Rate(PyBobIoAudioWriterObject* self) {
  return Py_BuildValue("d", self->v->rate());
}

PyDoc_STRVAR(s_rate_str, "rate");
PyDoc_STRVAR(s_rate_doc,
"[float] The sampling rate of the audio stream");

PyObject* PyBobIoAudioWriter_NumberOfChannels(PyBobIoAudioWriterObject* self) {
  return Py_BuildValue("n", self->v->numberOfChannels());
}

PyDoc_STRVAR(s_number_of_channels_str, "number_of_channels");
PyDoc_STRVAR(s_number_of_channels_doc,
"[int] The number of channels on the audio stream");

PyObject* PyBobIoAudioWriter_BitsPerSample(PyBobIoAudioWriterObject* self) {
  return Py_BuildValue("n", self->v->bitsPerSample());
}

PyDoc_STRVAR(s_bits_per_sample_str, "bits_per_sample");
PyDoc_STRVAR(s_bits_per_sample_doc,
"[int] The number of bits per sample in this audio stream");

PyObject* PyBobIoAudioWriter_NumberOfSamples(PyBobIoAudioWriterObject* self) {
  return Py_BuildValue("n", self->v->numberOfSamples());
}

PyDoc_STRVAR(s_number_of_samples_str, "number_of_samples");
PyDoc_STRVAR(s_number_of_samples_doc,
"[int] The number of samples in this audio stream");

PyObject* PyBobIoAudioWriter_Duration(PyBobIoAudioWriterObject* self) {
  return Py_BuildValue("d", self->v->duration());
}

PyDoc_STRVAR(s_duration_str, "duration");
PyDoc_STRVAR(s_duration_doc,
"[float] Total duration of this audio file in seconds");

PyObject* PyBobIoAudioWriter_CompressionFactor(PyBobIoAudioWriterObject* self) {
  return Py_BuildValue("d", self->v->compressionFactor());
}

PyDoc_STRVAR(s_compression_factor_str, "compressionfactor");
PyDoc_STRVAR(s_compression_factor_doc,
"[float] Compression factor on the audio stream");

PyObject* PyBobIoAudioWriter_EncodingName(PyBobIoAudioWriterObject* self) {
  return Py_BuildValue("s", bob::io::audio::encoding2string(self->v->encoding()));
}

PyDoc_STRVAR(s_encoding_name_str, "encoding");
PyDoc_STRVAR(s_encoding_name_doc,
"[str] Name of the encoding in which this audio file was recorded in");

PyObject* PyBobIoAudioWriter_TypeInfo(PyBobIoAudioWriterObject* self) {
  return PyBobIo_TypeInfoAsTuple(self->v->type());
}

PyDoc_STRVAR(s_type_str, "type");
PyDoc_STRVAR(s_type_doc,
"[tuple] Typing information to load all of the file at once");

static PyObject* PyBobIoAudioWriter_IsOpened(PyBobIoAudioWriterObject* self) {
  if (self->v->is_opened()) Py_RETURN_TRUE;
  Py_RETURN_FALSE;
}

PyDoc_STRVAR(s_is_opened_str, "is_opened");
PyDoc_STRVAR(s_is_opened_doc,
"[bool] A flag, indicating if the audio is still opened for writing\n\
(or has already been closed by the user using ``close()``)");

static PyGetSetDef PyBobIoAudioWriter_getseters[] = {
    {
      s_filename_str,
      (getter)PyBobIoAudioWriter_Filename,
      0,
      s_filename_doc,
      0,
    },
    {
      s_rate_str,
      (getter)PyBobIoAudioWriter_Rate,
      0,
      s_rate_doc,
      0,
    },
    {
      s_number_of_channels_str,
      (getter)PyBobIoAudioWriter_NumberOfChannels,
      0,
      s_number_of_channels_doc,
      0,
    },
    {
      s_bits_per_sample_str,
      (getter)PyBobIoAudioWriter_BitsPerSample,
      0,
      s_bits_per_sample_doc,
      0,
    },
    {
      s_number_of_samples_str,
      (getter)PyBobIoAudioWriter_NumberOfSamples,
      0,
      s_number_of_samples_doc,
      0,
    },
    {
      s_duration_str,
      (getter)PyBobIoAudioWriter_Duration,
      0,
      s_duration_doc,
      0,
    },
    {
      s_encoding_name_str,
      (getter)PyBobIoAudioWriter_EncodingName,
      0,
      s_encoding_name_doc,
      0,
    },
    {
      s_compression_factor_str,
      (getter)PyBobIoAudioWriter_CompressionFactor,
      0,
      s_compression_factor_doc,
      0,
    },
    {
      s_type_str,
      (getter)PyBobIoAudioWriter_TypeInfo,
      0,
      s_type_doc,
      0,
    },
    {
      s_is_opened_str,
      (getter)PyBobIoAudioWriter_IsOpened,
      0,
      s_is_opened_doc,
      0,
    },
    {0}  /* Sentinel */
};

static PyObject* PyBobIoAudioWriter_Repr(PyBobIoAudioWriterObject* self) {
  if (!self->v->is_opened()) {
    PyErr_Format(PyExc_RuntimeError, "`%s' for `%s' is closed",
        Py_TYPE(self)->tp_name, self->v->filename());
    return 0;
  }

  return
# if PY_VERSION_HEX >= 0x03000000
  PyUnicode_FromFormat
# else
  PyString_FromFormat
# endif
  ("%s(filename='%s', rate=%g, encoding=%s, bits_per_sample=%" PY_FORMAT_SIZE_T "d)", Py_TYPE(self)->tp_name, self->v->filename(), self->v->rate(), bob::io::audio::encoding2string(self->v->encoding()), self->v->bitsPerSample());
}

static PyObject* PyBobIoAudioWriter_Append(PyBobIoAudioWriterObject* self, PyObject *args, PyObject* kwds) {

  if (!self->v->is_opened()) {
    PyErr_Format(PyExc_RuntimeError, "`%s' for `%s' is closed",
        Py_TYPE(self)->tp_name, self->v->filename());
    return 0;
  }

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"sample", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* sample = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&", kwlist, &PyBlitzArray_BehavedConverter, &sample)) return 0;
  auto sample_ = make_safe(sample);

  if (sample->ndim != 1 && sample->ndim != 2) {
    PyErr_Format(PyExc_ValueError, "input array should have 1 or 2 dimensions, but you passed an array with %" PY_FORMAT_SIZE_T "d dimensions", sample->ndim);
    return 0;
  }

  if (sample->type_num != NPY_FLOAT64) {
    PyErr_Format(PyExc_TypeError, "input array should have dtype `float64', but you passed an array with dtype == `%s'", PyBlitzArray_TypenumAsString(sample->type_num));
    return 0;
  }

  try {
    if (sample->ndim == 1) {
      self->v->append(*PyBlitzArrayCxx_AsBlitz<double,1>(sample));
    }
    else {
      self->v->append(*PyBlitzArrayCxx_AsBlitz<double,2>(sample));
    }
  }
  catch (std::exception& e) {
    if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, e.what());
    return 0;
  }
  catch (...) {
    if (!PyErr_Occurred()) PyErr_Format(PyExc_RuntimeError, "caught unknown exception while writing sample #%" PY_FORMAT_SIZE_T "d to file `%s'", self->v->numberOfSamples(), self->v->filename());
    return 0;
  }

  Py_RETURN_NONE;

}

PyDoc_STRVAR(s_append_str, "append");
PyDoc_STRVAR(s_append_doc,
"x.append(sample) -> None\n\
\n\
Writes a new sample or set of samples to the file.\n\
\n\
The frame should be setup as a array with 1 dimension where each\n\
entry corresponds to one stream channel. Sets of samples should\n\
be setup as a 2D array in this way: (channels, samples).\n\
Arrays should contain only 64-bit float numbers.\n\
\n\
.. note::\n\
  At present time we only support arrays that have C-style storages\n\
  (if you pass reversed arrays or arrays with Fortran-style storage,\n\
  the result is undefined).\n\
\n\
");

static PyObject* PyBobIoAudioWriter_Close(PyBobIoAudioWriterObject* self) {
  self->v->close();
  Py_RETURN_NONE;
}

PyDoc_STRVAR(s_close_str, "close");
PyDoc_STRVAR(s_close_doc,
"x.close() -> None\n\
\n\
Closes the current audio stream and forces writing the trailer.\n\
After this point the audio is finalized and cannot be written to\n\
anymore.\n\
");

static PyMethodDef PyBobIoAudioWriter_Methods[] = {
    {
      s_append_str,
      (PyCFunction)PyBobIoAudioWriter_Append,
      METH_VARARGS|METH_KEYWORDS,
      s_append_doc,
    },
    {
      s_close_str,
      (PyCFunction)PyBobIoAudioWriter_Close,
      METH_NOARGS,
      s_close_doc,
    },
    {0}  /* Sentinel */
};

Py_ssize_t PyBobIoAudioWriter_Len(PyBobIoAudioWriterObject* self) {
  return self->v->numberOfSamples();
}

static PyMappingMethods PyBobIoAudioWriter_Mapping = {
    (lenfunc)PyBobIoAudioWriter_Len, //mp_length
    0, /* (binaryfunc)PyBobIoAudioWriter_GetItem, //mp_subscript */
    0  /* (objobjargproc)PyBobIoAudioWriter_SetItem //mp_ass_subscript */
};

PyTypeObject PyBobIoAudioWriter_Type = {
    PyVarObject_HEAD_INIT(0, 0)
    s_audiowriter_str,                          /*tp_name*/
    sizeof(PyBobIoAudioWriterObject),           /*tp_basicsize*/
    0,                                          /*tp_itemsize*/
    (destructor)PyBobIoAudioWriter_Delete,      /*tp_dealloc*/
    0,                                          /*tp_print*/
    0,                                          /*tp_getattr*/
    0,                                          /*tp_setattr*/
    0,                                          /*tp_compare*/
    (reprfunc)PyBobIoAudioWriter_Repr,          /*tp_repr*/
    0,                                          /*tp_as_number*/
    0,                                          /*tp_as_sequence*/
    &PyBobIoAudioWriter_Mapping,                /*tp_as_mapping*/
    0,                                          /*tp_hash */
    0,                                          /*tp_call*/
    (reprfunc)PyBobIoAudioWriter_Repr,          /*tp_str*/
    0,                                          /*tp_getattro*/
    0,                                          /*tp_setattro*/
    0,                                          /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /*tp_flags*/
    s_audiowriter_doc,                          /* tp_doc */
    0,		                                      /* tp_traverse */
    0,		                                      /* tp_clear */
    0,                                          /* tp_richcompare */
    0,		                                      /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,		                                      /* tp_iternext */
    PyBobIoAudioWriter_Methods,                 /* tp_methods */
    0,                                          /* tp_members */
    PyBobIoAudioWriter_getseters,               /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc)PyBobIoAudioWriter_Init,          /* tp_init */
    0,                                          /* tp_alloc */
    PyBobIoAudioWriter_New,                     /* tp_new */
};
