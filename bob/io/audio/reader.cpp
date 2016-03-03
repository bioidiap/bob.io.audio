/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed 20 Jan 2016 14:33:42 CET
 *
 * @brief Bindings to bob::io::audio::Reader
 */

#include "main.h"

static auto s_reader = bob::extension::ClassDoc(
  "reader",
  "Use this object to read samples from audio files",
  "Audio reader objects can read data from audio files. "
  "The current implementation uses `SoX <http://sox.sourceforge.net/>`_ , which is a stable freely available audio encoding and decoding library, designed specifically for these tasks. "
  "You can read an entire audio in memory by using the :py:meth:`load` method."
)
.add_constructor(
  bob::extension::FunctionDoc(
    "reader",
    "Opens an audio file for reading",
    "Opens the audio file with the given filename for reading, i.e., using the :py:meth:`load` function",
    true
  )
  .add_prototype("filename", "")
  .add_parameter("filename", "str", "The file path to the file you want to read data from")
);

/* The __init__(self) method */
static int PyBobIoAudioReader_Init(PyBobIoAudioReaderObject* self,
    PyObject *args, PyObject* kwds) {
BOB_TRY
  char** kwlist = s_reader.kwlist();

  char* filename = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &filename)) return -1;

  self->v.reset(new bob::io::audio::Reader(filename));
  return 0;
BOB_CATCH_MEMBER("constructor", -1)
}

static void PyBobIoAudioReader_Delete (PyBobIoAudioReaderObject* o) {
  o->v.reset();
  Py_TYPE(o)->tp_free((PyObject*)o);
}


static auto s_filename = bob::extension::VariableDoc(
  "filename",
  "str",
  "The full path to the file that will be decoded by this object"
);
PyObject* PyBobIoAudioReader_Filename(PyBobIoAudioReaderObject* self) {
  return Py_BuildValue("s", self->v->filename());
}


static auto s_rate = bob::extension::VariableDoc(
  "rate",
  "float",
  "The sampling rate of the audio stream"
);
PyObject* PyBobIoAudioReader_Rate(PyBobIoAudioReaderObject* self) {
  return Py_BuildValue("d", self->v->rate());
}


static auto s_number_of_channels = bob::extension::VariableDoc(
  "number_of_channels",
  "int",
  "The number of channels on the audio stream"
);
PyObject* PyBobIoAudioReader_NumberOfChannels(PyBobIoAudioReaderObject* self) {
  return Py_BuildValue("n", self->v->numberOfChannels());
}


static auto s_bits_per_sample = bob::extension::VariableDoc(
  "bits_per_sample",
  "int",
  "The number of bits per sample in this audio stream"
);
PyObject* PyBobIoAudioReader_BitsPerSample(PyBobIoAudioReaderObject* self) {
  return Py_BuildValue("n", self->v->bitsPerSample());
}


static auto s_number_of_samples = bob::extension::VariableDoc(
  "number_of_samples",
  "int",
  "The number of samples in this audio stream"
);
PyObject* PyBobIoAudioReader_NumberOfSamples(PyBobIoAudioReaderObject* self) {
  return Py_BuildValue("n", self->v->numberOfSamples());
}


static auto s_duration = bob::extension::VariableDoc(
  "duration",
  "float",
  "Total duration of this audio file in seconds"
);
PyObject* PyBobIoAudioReader_Duration(PyBobIoAudioReaderObject* self) {
  return Py_BuildValue("d", self->v->duration());
}


static auto s_compression_factor = bob::extension::VariableDoc(
  "compression_factor",
  "float",
  "Compression factor on the audio stream"
);
PyObject* PyBobIoAudioReader_CompressionFactor(PyBobIoAudioReaderObject* self) {
  return Py_BuildValue("d", self->v->compressionFactor());
}


static auto s_encoding = bob::extension::VariableDoc(
  "encoding",
  "str",
  "Name of the encoding in which this audio file was recorded"
);
PyObject* PyBobIoAudioReader_EncodingName(PyBobIoAudioReaderObject* self) {
  return Py_BuildValue("s", bob::io::audio::encoding2string(self->v->encoding()));
}


static auto s_type = bob::extension::VariableDoc(
  "type",
  "tuple",
  "Typing information to load all of the file at once"
);
PyObject* PyBobIoAudioReader_TypeInfo(PyBobIoAudioReaderObject* self) {
  return PyBobIo_TypeInfoAsTuple(self->v->type());
}


static PyGetSetDef PyBobIoAudioReader_getseters[] = {
    {
      s_filename.name(),
      (getter)PyBobIoAudioReader_Filename,
      0,
      s_filename.doc(),
      0,
    },
    {
      s_rate.name(),
      (getter)PyBobIoAudioReader_Rate,
      0,
      s_rate.doc(),
      0,
    },
    {
      s_number_of_channels.name(),
      (getter)PyBobIoAudioReader_NumberOfChannels,
      0,
      s_number_of_channels.doc(),
      0,
    },
    {
      s_bits_per_sample.name(),
      (getter)PyBobIoAudioReader_BitsPerSample,
      0,
      s_bits_per_sample.doc(),
      0,
    },
    {
      s_number_of_samples.name(),
      (getter)PyBobIoAudioReader_NumberOfSamples,
      0,
      s_number_of_samples.doc(),
      0,
    },
    {
      s_duration.name(),
      (getter)PyBobIoAudioReader_Duration,
      0,
      s_duration.doc(),
      0,
    },
    {
      s_encoding.name(),
      (getter)PyBobIoAudioReader_EncodingName,
      0,
      s_encoding.doc(),
      0,
    },
    {
      s_compression_factor.name(),
      (getter)PyBobIoAudioReader_CompressionFactor,
      0,
      s_compression_factor.doc(),
      0,
    },
    {
      s_type.name(),
      (getter)PyBobIoAudioReader_TypeInfo,
      0,
      s_type.doc(),
      0,
    },
    {0}  /* Sentinel */
};

static PyObject* PyBobIoAudioReader_Repr(PyBobIoAudioReaderObject* self) {
  return PyString_FromFormat("%s(filename='%s')", Py_TYPE(self)->tp_name, self->v->filename());
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


static auto s_load = bob::extension::FunctionDoc(
  "load",
  "Loads all of the audio stream in a :py:class:`numpy.ndarray`",
  "The data is organized in this way: ``(channels, data)``. ",
  true
)
.add_prototype("","data")
.add_return("data", ":py:class:`numpy.ndarray`", "The data read from this file")
;
static PyObject* PyBobIoAudioReader_Load(PyBobIoAudioReaderObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  char** kwlist = s_load.kwlist();

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

  bobskin skin((PyArrayObject*)retval, info.dtype);
  samples_read = self->v->load(skin, &Check_Interrupt);

  if (samples_read != shape[1]) {
    //resize
    shape[1] = samples_read;
    PyArray_Dims newshape;
    newshape.ptr = shape;
    newshape.len = info.nd;
    PyArray_Resize((PyArrayObject*)retval, &newshape, 1, NPY_ANYORDER);
  }

  return Py_BuildValue("O", retval);
BOB_CATCH_MEMBER("load", 0)
}

static PyMethodDef PyBobIoAudioReader_Methods[] = {
    {
      s_load.name(),
      (PyCFunction)PyBobIoAudioReader_Load,
      METH_VARARGS|METH_KEYWORDS,
      s_load.doc(),
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
  0
};

bool init_BobIoAudioReader(PyObject* module){

  // initialize the File
  PyBobIoAudioReader_Type.tp_name = s_reader.name();
  PyBobIoAudioReader_Type.tp_basicsize = sizeof(PyBobIoAudioReaderObject);
  PyBobIoAudioReader_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  PyBobIoAudioReader_Type.tp_doc = s_reader.doc();

  // set the functions
  PyBobIoAudioReader_Type.tp_new = PyType_GenericNew;
  PyBobIoAudioReader_Type.tp_init = reinterpret_cast<initproc>(PyBobIoAudioReader_Init);
  PyBobIoAudioReader_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobIoAudioReader_Delete);
  PyBobIoAudioReader_Type.tp_methods = PyBobIoAudioReader_Methods;
  PyBobIoAudioReader_Type.tp_getset = PyBobIoAudioReader_getseters;

  PyBobIoAudioReader_Type.tp_str = reinterpret_cast<reprfunc>(PyBobIoAudioReader_Repr);
  PyBobIoAudioReader_Type.tp_repr = reinterpret_cast<reprfunc>(PyBobIoAudioReader_Repr);
  PyBobIoAudioReader_Type.tp_as_mapping = &PyBobIoAudioReader_Mapping;


  // check that everything is fine
  if (PyType_Ready(&PyBobIoAudioReader_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobIoAudioReader_Type);
  return PyModule_AddObject(module, "reader", (PyObject*)&PyBobIoAudioReader_Type) >= 0;
}
