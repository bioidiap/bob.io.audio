/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed 20 Jan 2016 18:30:19 CET
 *
 * @brief Pythonic bindings
 */

#ifdef NO_IMPORT_ARRAY
#undef NO_IMPORT_ARRAY
#endif

#include <bob.blitz/capi.h>
#include <bob.blitz/cleanup.h>
#include <bob.core/api.h>
#include <bob.io.base/api.h>

static PyMethodDef module_methods[] = {
  {0}  /* Sentinel */
};

PyDoc_STRVAR(module_docstr, "Audio I/O support for Bob");

#if PY_VERSION_HEX >= 0x03000000
static PyModuleDef module_definition = {
  PyModuleDef_HEAD_INIT,
  BOB_EXT_MODULE_NAME,
  module_docstr,
  -1,
  module_methods,
  0, 0, 0, 0
};
#endif

static PyObject* create_module (void) {

# if PY_VERSION_HEX >= 0x03000000
  PyObject* m = PyModule_Create(&module_definition);
  auto m_ = make_xsafe(m);
  const char* ret = "O";
# else
  PyObject* m = Py_InitModule3(BOB_EXT_MODULE_NAME, module_methods, module_docstr);
  const char* ret = "N";
# endif
  if (!m) return 0;

  /* imports dependencies */
  if (import_bob_blitz() < 0) return 0;
  if (import_bob_core_logging() < 0) return 0;
  if (import_bob_io_base() < 0) return 0;

  /* activates image plugins */
  if (!PyBobIoCodec_Register(".tif", "TIFF, compresssed (libtiff)",
        &make_tiff_file)) {
    PyErr_Print();
  }

  if (!PyBobIoCodec_Register(".tiff", "TIFF, compresssed (libtiff)",
        &make_tiff_file)) {
    PyErr_Print();
  }

  if (BITS_IN_JSAMPLE == 8) {
    if (!PyBobIoCodec_Register(".jpg", "JPEG, compresssed (libjpeg)",
          &make_jpeg_file)) {
      PyErr_Print();
    }
    if (!PyBobIoCodec_Register(".jpeg", "JPEG, compresssed (libjpeg)",
          &make_jpeg_file)) {
      PyErr_Print();
    }
  }
  else {
    PyErr_Format(PyExc_RuntimeError, "libjpeg compiled with `%d' bits depth (instead of 8). JPEG images are hence not supported.", BITS_IN_JSAMPLE);
    PyErr_Print();
  }

  if (!PyBobIoCodec_Register(".gif", "GIF (giflib)", &make_gif_file)) {
    PyErr_Print();
  }

  if (!PyBobIoCodec_Register(".pbm", "PBM, indexed (libnetpbm)",
        &make_netpbm_file)) {
    PyErr_Print();
  }

  if (!PyBobIoCodec_Register(".pgm", "PGM, indexed (libnetpbm)",
        &make_netpbm_file)) {
    PyErr_Print();
  }

  if (!PyBobIoCodec_Register(".ppm", "PPM, indexed (libnetpbm)",
        &make_netpbm_file)) {
    PyErr_Print();
  }

  if (!PyBobIoCodec_Register(".png", "PNG, compressed (libpng)", &make_png_file)) {
    PyErr_Print();
  }

  if (!PyBobIoCodec_Register(".bmp", "BMP, (built-in codec)", &make_bmp_file)) {
    PyErr_Print();
  }

  return Py_BuildValue(ret, m);
}

PyMODINIT_FUNC BOB_EXT_ENTRY_NAME (void) {
# if PY_VERSION_HEX >= 0x03000000
  return
# endif
    create_module();
}
