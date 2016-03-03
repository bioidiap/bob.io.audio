/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed 20 Jan 2016 15:38:07 CET
 *
 * @brief Bindings to bob::io::audio::Writer
 */

#include "main.h"

static auto s_writer = bob::extension::ClassDoc(
  "writer",
  "Use this object to write samples to audio files",
  "Audio writer objects can write data to audio files. "
  "The current implementation uses `SoX <http://sox.sourceforge.net/>`_.\n\n"
  "Audio files are objects composed potentially multiple channels. "
  "The numerical representation are 2-D arrays where the first dimension corresponds to the channels of the audio stream and the second dimension represents the samples through time."
)
.add_constructor(
  bob::extension::FunctionDoc(
    "reader",
    "Opens an audio file for writing",
    "Opens the audio file with the given filename for writing, i.e., using the :py:meth:`append` function",
    true
  )
  .add_prototype("filename, [rate], [encoding], [bits_per_sample]", "")
  .add_parameter("filename", "str", "The file path to the file you want to write data to")
  .add_parameter("rate", "float", "[Default: ``8000.``] The number of samples per second")
  .add_parameter("encoding", "str", "[Default: ``'UNKNOWN'``] The encoding to use")
  .add_parameter("bits_per_sample", "int", "[Default: ``16``] The number of bits per sample to be recorded")
);

/* The __init__(self) method */
static int PyBobIoAudioWriter_Init(PyBobIoAudioWriterObject* self,
    PyObject *args, PyObject* kwds) {
BOB_TRY
  char** kwlist = s_writer.kwlist();

  char* filename = 0;
  double rate = 8000.;
  char* encoding = "UNKNOWN";
  Py_ssize_t bits_per_sample = 16;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|dsn", kwlist,
        &filename, &rate, &encoding, &bits_per_sample)) return -1;

  sox_encoding_t sox_encoding = bob::io::audio::string2encoding(encoding);

  self->v = boost::make_shared<bob::io::audio::Writer>(filename,
      rate, sox_encoding, bits_per_sample);

  return 0;
BOB_CATCH_MEMBER("constructor", -1)
}

static void PyBobIoAudioWriter_Delete (PyBobIoAudioWriterObject* o) {

  o->v.reset();
  Py_TYPE(o)->tp_free((PyObject*)o);

}


static auto s_filename = bob::extension::VariableDoc(
  "filename",
  "str",
  "The full path to the file that will be decoded by this object"
);
PyObject* PyBobIoAudioWriter_Filename(PyBobIoAudioWriterObject* self) {
  return Py_BuildValue("s", self->v->filename());
}


static auto s_rate = bob::extension::VariableDoc(
  "rate",
  "float",
  "The sampling rate of the audio stream"
);
PyObject* PyBobIoAudioWriter_Rate(PyBobIoAudioWriterObject* self) {
  return Py_BuildValue("d", self->v->rate());
}


static auto s_number_of_channels = bob::extension::VariableDoc(
  "number_of_channels",
  "int",
  "The number of channels on the audio stream"
);
PyObject* PyBobIoAudioWriter_NumberOfChannels(PyBobIoAudioWriterObject* self) {
  return Py_BuildValue("n", self->v->numberOfChannels());
}


static auto s_bits_per_sample = bob::extension::VariableDoc(
  "bits_per_sample",
  "int",
  "The number of bits per sample in this audio stream"
);
PyObject* PyBobIoAudioWriter_BitsPerSample(PyBobIoAudioWriterObject* self) {
  return Py_BuildValue("n", self->v->bitsPerSample());
}


static auto s_number_of_samples = bob::extension::VariableDoc(
  "number_of_samples",
  "int",
  "The number of samples in this audio stream"
);
PyObject* PyBobIoAudioWriter_NumberOfSamples(PyBobIoAudioWriterObject* self) {
  return Py_BuildValue("n", self->v->numberOfSamples());
}


static auto s_duration = bob::extension::VariableDoc(
  "duration",
  "float",
  "Total duration of this audio file in seconds"
);
PyObject* PyBobIoAudioWriter_Duration(PyBobIoAudioWriterObject* self) {
  return Py_BuildValue("d", self->v->duration());
}


static auto s_compression_factor = bob::extension::VariableDoc(
  "compression_factor",
  "float",
  "Compression factor on the audio stream"
);
PyObject* PyBobIoAudioWriter_CompressionFactor(PyBobIoAudioWriterObject* self) {
  return Py_BuildValue("d", self->v->compressionFactor());
}


static auto s_encoding = bob::extension::VariableDoc(
  "encoding",
  "str",
  "Name of the encoding in which this audio file will be written"
);
PyObject* PyBobIoAudioWriter_EncodingName(PyBobIoAudioWriterObject* self) {
  return Py_BuildValue("s", bob::io::audio::encoding2string(self->v->encoding()));
}


static auto s_type = bob::extension::VariableDoc(
  "type",
  "tuple",
  "Typing information to load all of the file at once"
);
PyObject* PyBobIoAudioWriter_TypeInfo(PyBobIoAudioWriterObject* self) {
  return PyBobIo_TypeInfoAsTuple(self->v->type());
}


static auto s_is_opened = bob::extension::VariableDoc(
  "is_opened",
  "bool",
  "A flag indicating if the audio is still opened for writing, or has already been closed by the user using :py:meth:`close`"
);
static PyObject* PyBobIoAudioWriter_IsOpened(PyBobIoAudioWriterObject* self) {
  if (self->v->is_opened()) Py_RETURN_TRUE;
  Py_RETURN_FALSE;
}


static PyGetSetDef PyBobIoAudioWriter_getseters[] = {
    {
      s_filename.name(),
      (getter)PyBobIoAudioWriter_Filename,
      0,
      s_filename.doc(),
      0,
    },
    {
      s_rate.name(),
      (getter)PyBobIoAudioWriter_Rate,
      0,
      s_rate.doc(),
      0,
    },
    {
      s_number_of_channels.name(),
      (getter)PyBobIoAudioWriter_NumberOfChannels,
      0,
      s_number_of_channels.doc(),
      0,
    },
    {
      s_bits_per_sample.name(),
      (getter)PyBobIoAudioWriter_BitsPerSample,
      0,
      s_bits_per_sample.doc(),
      0,
    },
    {
      s_number_of_samples.name(),
      (getter)PyBobIoAudioWriter_NumberOfSamples,
      0,
      s_number_of_samples.doc(),
      0,
    },
    {
      s_duration.name(),
      (getter)PyBobIoAudioWriter_Duration,
      0,
      s_duration.doc(),
      0,
    },
    {
      s_encoding.name(),
      (getter)PyBobIoAudioWriter_EncodingName,
      0,
      s_encoding.doc(),
      0,
    },
    {
      s_compression_factor.name(),
      (getter)PyBobIoAudioWriter_CompressionFactor,
      0,
      s_compression_factor.doc(),
      0,
    },
    {
      s_type.name(),
      (getter)PyBobIoAudioWriter_TypeInfo,
      0,
      s_type.doc(),
      0,
    },
    {
      s_is_opened.name(),
      (getter)PyBobIoAudioWriter_IsOpened,
      0,
      s_is_opened.doc(),
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


static auto s_append = bob::extension::FunctionDoc(
  "append",
  "Writes a new sample or set of samples to the file",
  "The frame should be setup as a array with 1 dimension where each entry corresponds to one stream channel. "
  "Sets of samples should be setup as a 2D array in this way: (channels, samples). "
  "Arrays should contain only 64-bit float numbers.\n\n"
  ".. note::\n"
  "  At present time we only support arrays that have C-style storages (if you pass reversed arrays or arrays with Fortran-style storage, the result is undefined)",
  true
)
.add_prototype("sample")
.add_parameter("sample", "array-like (1D or 2D, float)", "The sample(s) that should be appended to the file")
;
static PyObject* PyBobIoAudioWriter_Append(PyBobIoAudioWriterObject* self, PyObject *args, PyObject* kwds) {
BOB_TRY
  if (!self->v->is_opened()) {
    PyErr_Format(PyExc_RuntimeError, "`%s' for `%s' is closed",
        Py_TYPE(self)->tp_name, self->v->filename());
    return 0;
  }

  char** kwlist = s_append.kwlist();

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

  if (sample->ndim == 1) {
    self->v->append(*PyBlitzArrayCxx_AsBlitz<double,1>(sample));
  }
  else {
    self->v->append(*PyBlitzArrayCxx_AsBlitz<double,2>(sample));
  }
  Py_RETURN_NONE;
BOB_CATCH_MEMBER("append", 0)
}


static auto s_close = bob::extension::FunctionDoc(
  "close",
  "Closes the current audio stream and forces writing the trailer",
  "After this point the audio is finalized and cannot be written to anymore.",
  true
)
.add_prototype("")
;
static PyObject* PyBobIoAudioWriter_Close(PyBobIoAudioWriterObject* self) {
BOB_TRY
  self->v->close();
  Py_RETURN_NONE;
BOB_CATCH_MEMBER("close", 0)
}

static PyMethodDef PyBobIoAudioWriter_Methods[] = {
    {
      s_append.name(),
      (PyCFunction)PyBobIoAudioWriter_Append,
      METH_VARARGS|METH_KEYWORDS,
      s_append.doc(),
    },
    {
      s_close.name(),
      (PyCFunction)PyBobIoAudioWriter_Close,
      METH_NOARGS,
      s_close.doc(),
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
  0
};

bool init_BobIoAudioWriter(PyObject* module){

  // initialize the File
  PyBobIoAudioWriter_Type.tp_name = s_writer.name();
  PyBobIoAudioWriter_Type.tp_basicsize = sizeof(PyBobIoAudioWriterObject);
  PyBobIoAudioWriter_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  PyBobIoAudioWriter_Type.tp_doc = s_writer.doc();

  // set the functions
  PyBobIoAudioWriter_Type.tp_new = PyType_GenericNew;
  PyBobIoAudioWriter_Type.tp_init = reinterpret_cast<initproc>(PyBobIoAudioWriter_Init);
  PyBobIoAudioWriter_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobIoAudioWriter_Delete);
  PyBobIoAudioWriter_Type.tp_methods = PyBobIoAudioWriter_Methods;
  PyBobIoAudioWriter_Type.tp_getset = PyBobIoAudioWriter_getseters;

  PyBobIoAudioWriter_Type.tp_str = reinterpret_cast<reprfunc>(PyBobIoAudioWriter_Repr);
  PyBobIoAudioWriter_Type.tp_repr = reinterpret_cast<reprfunc>(PyBobIoAudioWriter_Repr);
  PyBobIoAudioWriter_Type.tp_as_mapping = &PyBobIoAudioWriter_Mapping;


  // check that everything is fine
  if (PyType_Ready(&PyBobIoAudioWriter_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobIoAudioWriter_Type);
  return PyModule_AddObject(module, "writer", (PyObject*)&PyBobIoAudioWriter_Type) >= 0;
}
