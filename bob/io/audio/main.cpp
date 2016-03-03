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

#include <map>
#include "file.h"
#include "main.h"

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
  PyObject* module = PyModule_Create(&module_definition);
  auto module_ = make_xsafe(module);
  const char* ret = "O";
# else
  PyObject* module = Py_InitModule3(BOB_EXT_MODULE_NAME, module_methods, module_docstr);
  const char* ret = "N";
# endif
  if (!module) return 0;

  /* register the types to python */
  if (!init_BobIoAudioReader(module)) return 0;
  if (!init_BobIoAudioWriter(module)) return 0;


  /* imports dependencies */
  if (import_bob_blitz() < 0) return 0;
  if (import_bob_core_logging() < 0) return 0;
  if (import_bob_io_base() < 0) return 0;

  /* activates audio plugins */
  for (auto k=bob::io::audio::SUPPORTED_FORMATS.begin();
      k!=bob::io::audio::SUPPORTED_FORMATS.end(); ++k) {
    if (!PyBobIoCodec_Register(k->first.c_str(), k->second.c_str(), &make_file)) {
      PyErr_Print();
      //do not return 0, or we may crash badly
    }
  }

  return Py_BuildValue(ret, module);
}

PyMODINIT_FUNC BOB_EXT_ENTRY_NAME (void) {
# if PY_VERSION_HEX >= 0x03000000
  return
# endif
    create_module();
}
