/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed 20 Jan 2016 14:33:42 CET
 *
 * @brief Bindings to bob::io::audio::Reader
 */

#include "bobskin.h"

#include <boost/make_shared.hpp>
#include <numpy/arrayobject.h>
#include <bob.blitz/capi.h>
#include <bob.blitz/cleanup.h>
#include <bob.io.base/api.h>
#include <stdexcept>

#include "cpp/utils.h"
#include "cpp/reader.h"

#define AUDIOREADER_NAME "reader"
PyDoc_STRVAR(s_audioreader_str, BOB_EXT_MODULE_PREFIX "." AUDIOREADER_NAME);

PyDoc_STRVAR(s_audioreader_doc,
"reader(filename) -> new reader\n\
\n\
Use this object to read samples from audio files.\n\
\n\
Constructor parameters:\n\
\n\
filename\n\
  [str] The file path to the file you want to read data from\n\
\n\
Audio reader objects can read data from audio files. The current\n\
implementation uses `SoX <http://sox.sourceforge.net/>`_  which is\n\
a stable freely available audio encoding and decoding library,\n\
designed specifically for these tasks. You can read an entire\n\
audio in memory by using the :py:meth:`bob.io.audio.reader.load`\n\
method.\n\
\n\
");

typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::io::audio::Reader> v;
} PyBobIoAudioReaderObject;

extern PyTypeObject PyBobIoAudioReader_Type;

/* How to create a new PyBobIoAudioReaderObject */
static PyObject* PyBobIoAudioReader_New(PyTypeObject* type, PyObject*, PyObject*) {

  /* Allocates the python object itself */
  PyBobIoAudioReaderObject* self = (PyBobIoAudioReaderObject*)type->tp_alloc(type, 0);

  self->v.reset();

  return reinterpret_cast<PyObject*>(self);
}

static void PyBobIoAudioReader_Delete (PyBobIoAudioReaderObject* o) {

  o->v.reset();
  Py_TYPE(o)->tp_free((PyObject*)o);

}

/* The __init__(self) method */
static int PyBobIoAudioReader_Init(PyBobIoAudioReaderObject* self,
    PyObject *args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"filename", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* filename = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&", kwlist,
        &PyBobIo_FilenameConverter, &filename)) return -1;

  auto filename_ = make_safe(filename);

#if PY_VERSION_HEX >= 0x03000000
  const char* c_filename = PyBytes_AS_STRING(filename);
#else
  const char* c_filename = PyString_AS_STRING(filename);
#endif

  try {
    self->v.reset(new bob::io::audio::Reader(c_filename));
  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot open audio file `%s' for reading: unknown exception caught", c_filename);
    return -1;
  }

  return 0; ///< SUCCESS
}

PyObject* PyBobIoAudioReader_Filename(PyBobIoAudioReaderObject* self) {
  return Py_BuildValue("s", self->v->filename());
}

PyDoc_STRVAR(s_filename_str, "filename");
PyDoc_STRVAR(s_filename_doc,
"[str] The full path to the file that will be decoded by this object");

PyObject* PyBobIoAudioReader_Rate(PyBobIoAudioReaderObject* self) {
  return Py_BuildValue("d", self->v->rate());
}

PyDoc_STRVAR(s_rate_str, "rate");
PyDoc_STRVAR(s_rate_doc,
"[float] The sampling rate of the audio stream");

PyObject* PyBobIoAudioReader_NumberOfChannels(PyBobIoAudioReaderObject* self) {
  return Py_BuildValue("n", self->v->numberOfChannels());
}

PyDoc_STRVAR(s_number_of_channels_str, "number_of_channels");
PyDoc_STRVAR(s_number_of_channels_doc,
"[int] The number of channels on the audio stream");

PyObject* PyBobIoAudioReader_BitsPerSample(PyBobIoAudioReaderObject* self) {
  return Py_BuildValue("n", self->v->bitsPerSample());
}

PyDoc_STRVAR(s_bits_per_sample_str, "bits_per_sample");
PyDoc_STRVAR(s_bits_per_sample_doc,
"[int] The number of bits per sample in this audio stream");

PyObject* PyBobIoAudioReader_NumberOfSamples(PyBobIoAudioReaderObject* self) {
  return Py_BuildValue("n", self->v->numberOfSamples());
}

PyDoc_STRVAR(s_number_of_samples_str, "number_of_samples");
PyDoc_STRVAR(s_number_of_samples_doc,
"[int] The number of samples in this audio stream");

PyObject* PyBobIoAudioReader_Duration(PyBobIoAudioReaderObject* self) {
  return Py_BuildValue("d", self->v->duration());
}

PyDoc_STRVAR(s_duration_str, "duration");
PyDoc_STRVAR(s_duration_doc,
"[float] Total duration of this audio file in seconds");

PyObject* PyBobIoAudioReader_CompressionFactor(PyBobIoAudioReaderObject* self) {
  return Py_BuildValue("d", self->v->compressionFactor());
}

PyDoc_STRVAR(s_compression_factor_str, "compressionfactor");
PyDoc_STRVAR(s_compression_factor_doc,
"[float] Compression factor on the audio stream");

PyObject* PyBobIoAudioReader_EncodingName(PyBobIoAudioReaderObject* self) {
  return Py_BuildValue("s", bob::io::audio::encoding2string(self->v->encoding()));
}

PyDoc_STRVAR(s_encoding_name_str, "encoding");
PyDoc_STRVAR(s_encoding_name_doc,
"[str] Name of the encoding in which this audio file was recorded in");

PyObject* PyBobIoAudioReader_TypeInfo(PyBobIoAudioReaderObject* self) {
  return PyBobIo_TypeInfoAsTuple(self->v->type());
}

PyDoc_STRVAR(s_type_str, "type");
PyDoc_STRVAR(s_type_doc,
"[tuple] Typing information to load all of the file at once");

static PyGetSetDef PyBobIoAudioReader_getseters[] = {
    {
      s_filename_str,
      (getter)PyBobIoAudioReader_Filename,
      0,
      s_filename_doc,
      0,
    },
    {
      s_rate_str,
      (getter)PyBobIoAudioReader_Rate,
      0,
      s_rate_doc,
      0,
    },
    {
      s_number_of_channels_str,
      (getter)PyBobIoAudioReader_NumberOfChannels,
      0,
      s_number_of_channels_doc,
      0,
    },
    {
      s_bits_per_sample_str,
      (getter)PyBobIoAudioReader_BitsPerSample,
      0,
      s_bits_per_sample_doc,
      0,
    },
    {
      s_number_of_samples_str,
      (getter)PyBobIoAudioReader_NumberOfSamples,
      0,
      s_number_of_samples_doc,
      0,
    },
    {
      s_duration_str,
      (getter)PyBobIoAudioReader_Duration,
      0,
      s_duration_doc,
      0,
    },
    {
      s_encoding_name_str,
      (getter)PyBobIoAudioReader_EncodingName,
      0,
      s_encoding_name_doc,
      0,
    },
    {
      s_compression_factor_str,
      (getter)PyBobIoAudioReader_CompressionFactor,
      0,
      s_compression_factor_doc,
      0,
    },
    {
      s_type_str,
      (getter)PyBobIoAudioReader_TypeInfo,
      0,
      s_type_doc,
      0,
    },
    {0}  /* Sentinel */
};

static PyObject* PyBobIoAudioReader_Repr(PyBobIoAudioReaderObject* self) {
  return
# if PY_VERSION_HEX >= 0x03000000
  PyUnicode_FromFormat
# else
  PyString_FromFormat
# endif
  ("%s(filename='%s')", Py_TYPE(self)->tp_name, self->v->filename());
}

/**
 * If a keyboard interruption occurs, then it is translated into a C++
 * exception that makes the loop stops.
 */
static void Check_Interrupt() {
  if(PyErr_CheckSignals() == -1) {
    if (!PyErr_Occurred()) PyErr_SetInterrupt();
    throw std::runtime_error("error is already set");
  }
}

static PyObject* PyBobIoAudioReader_Load(PyBobIoAudioReaderObject* self, PyObject *args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* raise = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "", kwlist, &raise)) return 0;

  const bob::io::base::array::typeinfo& info = self->v->type();

  npy_intp shape[NPY_MAXDIMS];
  for (size_t k=0; k<info.nd; ++k) shape[k] = info.shape[k];

  int type_num = PyBobIo_AsTypenum(info.dtype);
  if (type_num == NPY_NOTYPE) return 0; ///< failure

  PyObject* retval = PyArray_SimpleNew(info.nd, shape, type_num);
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

  Py_ssize_t samples_read = 0;

  try {
    bobskin skin((PyArrayObject*)retval, info.dtype);
    samples_read = self->v->load(skin, &Check_Interrupt);
  }
  catch (std::exception& e) {
    if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, e.what());
    return 0;
  }
  catch (...) {
    if (!PyErr_Occurred()) PyErr_Format(PyExc_RuntimeError, "caught unknown exception while reading audio from file `%s'", self->v->filename());
    return 0;
  }

  if (samples_read != shape[1]) {
    //resize
    shape[1] = samples_read;
    PyArray_Dims newshape;
    newshape.ptr = shape;
    newshape.len = info.nd;
    PyArray_Resize((PyArrayObject*)retval, &newshape, 1, NPY_ANYORDER);
  }

  Py_INCREF(retval);
  return retval;

}

PyDoc_STRVAR(s_load_str, "load");
PyDoc_STRVAR(s_load_doc,
"x.load() -> numpy.ndarray\n\
\n\
Loads all of the audio stream in a numpy ndarray organized\n\
in this way: (channels, data). I'll dynamically allocate the\n\
output array and return it to you.\n\
\n\
");

static PyMethodDef PyBobIoAudioReader_Methods[] = {
    {
      s_load_str,
      (PyCFunction)PyBobIoAudioReader_Load,
      METH_VARARGS|METH_KEYWORDS,
      s_load_doc,
    },
    {0}  /* Sentinel */
};

Py_ssize_t PyBobIoAudioReader_Len(PyBobIoAudioReaderObject* self) {
  return self->v->numberOfSamples();
}

static PyMappingMethods PyBobIoAudioReader_Mapping = {
    (lenfunc)PyBobIoAudioReader_Len, //mp_lenght
    0, /* (binaryfunc)PyBobIoAudioReader_GetItem, //mp_subscript */
    0  /* (objobjargproc)PyBobIoAudioReader_SetItem //mp_ass_subscript */
};

PyTypeObject PyBobIoAudioReader_Type = {
    PyVarObject_HEAD_INIT(0, 0)
    s_audioreader_str,                          /*tp_name*/
    sizeof(PyBobIoAudioReaderObject),           /*tp_basicsize*/
    0,                                          /*tp_itemsize*/
    (destructor)PyBobIoAudioReader_Delete,      /*tp_dealloc*/
    0,                                          /*tp_print*/
    0,                                          /*tp_getattr*/
    0,                                          /*tp_setattr*/
    0,                                          /*tp_compare*/
    (reprfunc)PyBobIoAudioReader_Repr,          /*tp_repr*/
    0,                                          /*tp_as_number*/
    0,                                          /*tp_as_sequence*/
    &PyBobIoAudioReader_Mapping,                /*tp_as_mapping*/
    0,                                          /*tp_hash */
    0,                                          /*tp_call*/
    (reprfunc)PyBobIoAudioReader_Repr,          /*tp_str*/
    0,                                          /*tp_getattro*/
    0,                                          /*tp_setattro*/
    0,                                          /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /*tp_flags*/
    s_audioreader_doc,                          /* tp_doc */
    0,		                                      /* tp_traverse */
    0,		                                      /* tp_clear */
    0,                                          /* tp_richcompare */
    0,		                                      /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,		                                      /* tp_iternext */
    PyBobIoAudioReader_Methods,                 /* tp_methods */
    0,                                          /* tp_members */
    PyBobIoAudioReader_getseters,               /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc)PyBobIoAudioReader_Init,          /* tp_init */
    0,                                          /* tp_alloc */
    PyBobIoAudioReader_New,                     /* tp_new */
};
