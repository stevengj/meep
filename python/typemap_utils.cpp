/* Copyright (C) 2005-2021 Massachusetts Institute of Technology
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2, or (at your option)
 *  any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software Foundation,
 *  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 */

// Utility functions for pymeep typemaps

#if PY_MAJOR_VERSION >= 3
#define PyObject_ToCharPtr(n) PyUnicode_AsUTF8(n)
#define IsPyString(n) PyUnicode_Check(n)
#define PyInteger_Check(n) PyLong_Check(n)
#define PyInteger_AsLong(n) PyLong_AsLong(n)
#define PyInteger_FromLong(n) PyLong_FromLong(n)
#else
#define PyObject_ToCharPtr(n) py2_string_as_utf8(n)
#define IsPyString(n) PyString_Check(n) || PyUnicode_Check(n)
#define PyInteger_Check(n) PyInt_Check(n)
#define PyInteger_AsLong(n) PyInt_AsLong(n)
#define PyInteger_FromLong(n) PyInt_FromLong(n)
#endif

PyObject *py_callback = NULL;
PyObject *py_callback_v3 = NULL;
PyObject *py_amp_func = NULL;

static void abort_with_stack_trace() {
  PyErr_PrintEx(0);
  meep::abort("Error in typemaps");
}

static int pymedium_to_medium(PyObject *po, medium_struct *m);
static int pymaterial_to_material(PyObject *po, material_type *mt);

#if PY_MAJOR_VERSION == 2
static char *py2_string_as_utf8(PyObject *po) {
  if (PyString_Check(po)) { return PyString_AsString(po); }
  else if (PyUnicode_Check(po)) {
    PyObject *s = PyUnicode_AsUTF8String(po);
    char *result = PyString_AsString(s);
    Py_DECREF(s);
    return result;
  }
  else {
    return NULL;
  }
}
#endif

static PyObject *get_meep_mod() {
  // Return value: Borrowed reference
  static PyObject *meep_mod = NULL;
  if (meep_mod == NULL) { meep_mod = PyImport_ImportModule("meep"); }
  return meep_mod;
}

static PyObject *get_geom_mod() {
  // Return value: Borrowed reference
  static PyObject *geom_mod = NULL;
  if (geom_mod == NULL) { geom_mod = PyImport_ImportModule("meep.geom"); }
  return geom_mod;
}

static PyObject *py_material_object() {
  // Return value; Borrowed reference
  static PyObject *material_object = NULL;
  if (material_object == NULL) {
    PyObject *geom_mod = get_geom_mod();
    material_object = PyObject_GetAttrString(geom_mod, "Medium");
  }
  return material_object;
}

static PyObject *py_material_grid_object() {
  // Return value: Borrowed reference
  static PyObject *material_object = NULL;
  if (material_object == NULL) {
    PyObject *geom_mod = get_geom_mod();
    material_object = PyObject_GetAttrString(geom_mod, "MaterialGrid");
  }
  return material_object;
}

static PyObject *py_vector3_object() {
  // Return value: Borrowed reference
  static PyObject *vector3_object = NULL;
  if (vector3_object == NULL) {
    PyObject *geom_mod = get_geom_mod();
    vector3_object = PyObject_GetAttrString(geom_mod, "Vector3");
  }
  return vector3_object;
}

static PyObject *py_volume_object() {
  // Return value: Borrowed reference
  static PyObject *volume_object = NULL;
  if (volume_object == NULL) {
    PyObject *meep_mod = get_meep_mod();
    volume_object = PyObject_GetAttrString(meep_mod, "Volume");
  }
  return volume_object;
}

static PyObject *vec2py(const meep::vec &v, bool newobj = false) {
  // Return value: New reference
  double x = 0, y = 0, z = 0;

  switch (v.dim) {
    case meep::D1: z = v.z(); break;
    case meep::D2:
      x = v.x();
      y = v.y();
      break;
    case meep::D3:
      x = v.x();
      y = v.y();
      z = v.z();
      break;
    case meep::Dcyl:
      x = v.r();
      z = v.z();
      break;
  }

  if (newobj) {
    PyObject *v3_class = py_vector3_object();
    PyObject *args = Py_BuildValue("(d,d,d)", x, y, z);
    PyObject *res = PyObject_Call(v3_class, args, NULL);

    Py_DECREF(args);

    return res;
  }
  else {
    if (py_callback_v3 == NULL) {
      PyObject *v3_class = py_vector3_object();
      PyObject *args = PyTuple_New(0);
      py_callback_v3 = PyObject_Call(v3_class, args, NULL);

      Py_DECREF(args);
    }

    PyObject *pyx = PyFloat_FromDouble(x);
    PyObject *pyy = PyFloat_FromDouble(y);
    PyObject *pyz = PyFloat_FromDouble(z);

    PyObject_SetAttrString(py_callback_v3, "x", pyx);
    PyObject_SetAttrString(py_callback_v3, "y", pyy);
    PyObject_SetAttrString(py_callback_v3, "z", pyz);

    Py_DECREF(pyx);
    Py_DECREF(pyy);
    Py_DECREF(pyz);

    Py_INCREF(py_callback_v3);
    return py_callback_v3;
  }
}

static void py_user_material_func_wrap(vector3 x, void *user_data, medium_struct *medium) {
  PyObject *py_vec = vec2py(vector3_to_vec(x));

  PyObject *pyret = PyObject_CallFunctionObjArgs((PyObject *)user_data, py_vec, NULL);

  if (!pyret) { abort_with_stack_trace(); }

  if (!pymedium_to_medium(pyret, medium)) { abort_with_stack_trace(); }

  Py_DECREF(py_vec);
  Py_DECREF(pyret);
}

static void py_epsilon_func_wrap(vector3 x, void *user_data, medium_struct *medium) {
  PyObject *py_vec = vec2py(vector3_to_vec(x));

  PyObject *pyret = PyObject_CallFunctionObjArgs((PyObject *)user_data, py_vec, NULL);

  if (!pyret) { abort_with_stack_trace(); }

  double eps = PyFloat_AsDouble(pyret);

  medium->epsilon_diag.x = eps;
  medium->epsilon_diag.y = eps;
  medium->epsilon_diag.z = eps;

  Py_DECREF(py_vec);
  Py_DECREF(pyret);
}

static std::string py_class_name_as_string(PyObject *po) {
  PyObject *py_type = PyObject_Type(po);
  PyObject *name = PyObject_GetAttrString(py_type, "__name__");

  const char *bytes = PyObject_ToCharPtr(name);

  std::string class_name(bytes);

  Py_XDECREF(py_type);
  Py_XDECREF(name);

  return class_name;
}

static int pyv3_to_v3(PyObject *po, vector3 *v) {

  PyObject *py_x = PyObject_GetAttrString(po, "x");
  PyObject *py_y = PyObject_GetAttrString(po, "y");
  PyObject *py_z = PyObject_GetAttrString(po, "z");

  if (!py_x || !py_y || !py_z) { abort_with_stack_trace(); }

  double x = PyFloat_AsDouble(py_x);
  double y = PyFloat_AsDouble(py_y);
  double z = PyFloat_AsDouble(py_z);
  Py_DECREF(py_x);
  Py_DECREF(py_y);
  Py_DECREF(py_z);

  v->x = x;
  v->y = y;
  v->z = z;

  return 1;
}

static int pyv3_to_cv3(PyObject *po, cvector3 *v) {
  PyObject *py_x = PyObject_GetAttrString(po, "x");
  PyObject *py_y = PyObject_GetAttrString(po, "y");
  PyObject *py_z = PyObject_GetAttrString(po, "z");

  if (!py_x || !py_y || !py_z) { abort_with_stack_trace(); }

  std::complex<double> x =
      std::complex<double>(PyComplex_RealAsDouble(py_x), PyComplex_ImagAsDouble(py_x));
  std::complex<double> y =
      std::complex<double>(PyComplex_RealAsDouble(py_y), PyComplex_ImagAsDouble(py_y));
  std::complex<double> z =
      std::complex<double>(PyComplex_RealAsDouble(py_z), PyComplex_ImagAsDouble(py_z));
  Py_DECREF(py_x);
  Py_DECREF(py_y);
  Py_DECREF(py_z);

  v->x.re = x.real();
  v->x.im = x.imag();
  v->y.re = y.real();
  v->y.im = y.imag();
  v->z.re = z.real();
  v->z.im = z.imag();

  return 1;
}

static PyObject *v3_to_pyv3(vector3 *v) {
  // Return value: New reference
  PyObject *v3_class = py_vector3_object();
  PyObject *args = Py_BuildValue("(ddd)", v->x, v->y, v->z);
  PyObject *py_v = PyObject_Call(v3_class, args, NULL);

  Py_DECREF(args);

  return py_v;
}

static int get_attr_v3(PyObject *py_obj, vector3 *v, const char *name) {
  int rval = 1;
  PyObject *py_attr = PyObject_GetAttrString(py_obj, name);

  if (!py_attr) { abort_with_stack_trace(); }

  if (!pyv3_to_v3(py_attr, v)) { rval = 0; }

  Py_XDECREF(py_attr);
  return rval;
}

static int get_attr_v3_cmplx(PyObject *py_obj, cvector3 *v, const char *name) {
  int rval = 1;
  PyObject *py_attr = PyObject_GetAttrString(py_obj, name);

  if (!py_attr) { abort_with_stack_trace(); }

  if (!pyv3_to_cv3(py_attr, v)) { rval = 0; }

  Py_XDECREF(py_attr);
  return rval;
}

static int get_attr_dbl(PyObject *py_obj, double *result, const char *name) {
  PyObject *py_attr = PyObject_GetAttrString(py_obj, name);

  if (!py_attr) { abort_with_stack_trace(); }

  *result = PyFloat_AsDouble(py_attr);
  Py_XDECREF(py_attr);
  return 1;
}

static int get_attr_int(PyObject *py_obj, int *result, const char *name) {
  PyObject *py_attr = PyObject_GetAttrString(py_obj, name);

  if (!py_attr) { abort_with_stack_trace(); }

  *result = PyInteger_AsLong(py_attr);
  Py_XDECREF(py_attr);
  return 1;
}

static int get_attr_material(PyObject *po, material_type *m) {
  int rval = 1;
  PyObject *py_material = PyObject_GetAttrString(po, "material");

  if (!py_material) { abort_with_stack_trace(); }

  if (!pymaterial_to_material(py_material, m)) { rval = 0; }

  Py_XDECREF(py_material);

  return rval;
}

static int pytransition_to_transition(PyObject *py_trans, transition *trans) {

  int from, to;
  double trans_rate, freq, gamma, pump_rate;
  vector3 sigma_diag;

  if (!get_attr_int(py_trans, &from, "from_level") || !get_attr_int(py_trans, &to, "to_level") ||
      !get_attr_dbl(py_trans, &trans_rate, "transition_rate") ||
      !get_attr_dbl(py_trans, &freq, "frequency") || !get_attr_dbl(py_trans, &gamma, "gamma") ||
      !get_attr_dbl(py_trans, &pump_rate, "pumping_rate") ||
      !get_attr_v3(py_trans, &sigma_diag, "sigma_diag")) {

    return 0;
  }

  trans->from_level = from;
  trans->to_level = to;
  trans->transition_rate = trans_rate;
  trans->frequency = freq;
  trans->gamma = gamma;
  trans->pumping_rate = pump_rate;
  trans->sigma_diag.x = sigma_diag.x;
  trans->sigma_diag.y = sigma_diag.y;
  trans->sigma_diag.z = sigma_diag.z;

  return 1;
}

static int py_susceptibility_to_susceptibility(PyObject *po, susceptibility_struct *s) {
  if (!get_attr_v3(po, &s->sigma_diag, "sigma_diag") ||
      !get_attr_v3(po, &s->sigma_offdiag, "sigma_offdiag")) {

    return 0;
  }

  s->frequency = 0;
  s->gamma = 0;
  s->alpha = 0;
  s->noise_amp = 0;
  s->bias.x = s->bias.y = s->bias.z = 0;
  s->saturated_gyrotropy = false;
  s->transitions.resize(0);
  s->initial_populations.resize(0);

  if (PyObject_HasAttrString(po, "frequency")) {
    if (!get_attr_dbl(po, &s->frequency, "frequency")) { return 0; }
  }

  if (PyObject_HasAttrString(po, "gamma")) {
    if (!get_attr_dbl(po, &s->gamma, "gamma")) { return 0; }
  }

  if (PyObject_HasAttrString(po, "noise_amp")) {
    if (!get_attr_dbl(po, &s->noise_amp, "noise_amp")) { return 0; }
  }

  if (PyObject_HasAttrString(po, "bias")) {
    if (!get_attr_v3(po, &s->bias, "bias")) return 0;
  }

  if (PyObject_HasAttrString(po, "alpha")) {
    s->saturated_gyrotropy = true;
    if (!get_attr_dbl(po, &s->alpha, "alpha")) { return 0; }
  }

  if (PyObject_HasAttrString(po, "transitions")) {
    // MultilevelAtom
    PyObject *py_trans = PyObject_GetAttrString(po, "transitions");
    if (!py_trans) { return 0; }
    int length = PyList_Size(py_trans);
    s->transitions.resize(length);

    for (int i = 0; i < length; ++i) {
      if (!pytransition_to_transition(PyList_GetItem(py_trans, i), &s->transitions[i])) {
        return 0;
      }
    }
    Py_DECREF(py_trans);

    PyObject *py_pop = PyObject_GetAttrString(po, "initial_populations");
    if (!py_pop) { return 0; }
    length = PyList_Size(py_pop);
    s->initial_populations.resize(length);

    for (int i = 0; i < length; ++i) {
      s->initial_populations[i] = PyFloat_AsDouble(PyList_GetItem(py_pop, i));
    }
    Py_DECREF(py_pop);
  }

  std::string class_name = py_class_name_as_string(po);

  if (class_name.find(std::string("Drude")) != std::string::npos) { s->drude = true; }
  else {
    s->drude = false;
  }

  s->is_file = false;

  return 1;
}

static int py_list_to_susceptibility_list(PyObject *po, susceptibility_list *sl) {
  if (!PyList_Check(po)) { abort_with_stack_trace(); }

  int length = PyList_Size(po);
  sl->resize(length);

  for (int i = 0; i < length; i++) {
    if (!py_susceptibility_to_susceptibility(PyList_GetItem(po, i), &sl->at(i))) { return 0; }
  }

  return 1;
}

static int pymaterial_grid_to_material_grid(PyObject *po, material_data *md) {
  // po must be a python MaterialGrid object

  int rval = 1;

  // specify the type of material grid
  PyObject * type = PyObject_GetAttrString(po, "grid_type");
  long gt_enum = PyInt_AsLong(type);
  Py_DECREF(type);

  switch (gt_enum) {
    case 0: md->material_grid_kinds = material_data::U_MIN; break;
    case 1: md->material_grid_kinds = material_data::U_PROD; break;
    case 2: md->material_grid_kinds = material_data::U_MEAN; break;
    case 3: md->material_grid_kinds = material_data::U_DEFAULT; break;
    default: meep::abort("Invalid material grid enumeration code: %d.\n", (int) gt_enum);
  }

  // initialize grid size
  if (!get_attr_v3(po, &md->grid_size, "grid_size")) {
    meep::abort("MaterialGrid grid_size failed to init.");
  }

  // initialize user specified materials
  PyObject *po_medium1 = PyObject_GetAttrString(po, "medium1");
  if (!pymedium_to_medium(po_medium1, &md->medium_1)) {
    meep::abort("MaterialGrid medium1 failed to init.");
  }
  PyObject *po_medium2 = PyObject_GetAttrString(po, "medium2");
  if (!pymedium_to_medium(po_medium2, &md->medium_2)) {
    meep::abort("MaterialGrid medium2 failed to init.");
  }

  // Initialize weights
  PyObject *po_dp = PyObject_GetAttrString(po, "weights");
  PyArrayObject *pao = (PyArrayObject *)po_dp;
  if (!PyArray_Check(pao)) { meep::abort("MaterialGrid weights failed to init."); }
  if (!PyArray_ISCARRAY(pao)) {
    meep::abort("Numpy array weights must be C-style contiguous.");
  }
  md->weights = new double[PyArray_SIZE(pao)];
  memcpy(md->weights, (double *)PyArray_DATA(pao), PyArray_SIZE(pao) * sizeof(double));

  // if needed, combine sus structs to main object
  PyObject *py_e_sus_m1 = PyObject_GetAttrString(po_medium1, "E_susceptibilities");
  PyObject *py_e_sus_m2 = PyObject_GetAttrString(po_medium2, "E_susceptibilities");
  if (!py_e_sus_m1 || !py_e_sus_m2) { rval = 0; }

  PyObject *py_sus = NULL;
  if (rval != 0) {
    py_sus = PyList_New(0);
    for (int i = 0; i < PyList_Size(py_e_sus_m1); i++) {
      if (PyList_Append(py_sus, PyList_GetItem(py_e_sus_m1, i)) != 0)
        meep::abort("unable to merge e sus lists.\n");
    }
    for (int i = 0; i < PyList_Size(py_e_sus_m2); i++) {
      if (PyList_Append(py_sus, PyList_GetItem(py_e_sus_m2, i)) != 0)
        meep::abort("unable to merge e sus lists.\n");
    }

    if (!py_list_to_susceptibility_list(py_sus, &md->medium.E_susceptibilities)) { rval = 0; }
  }

  Py_DECREF(po_medium1);
  Py_DECREF(po_medium2);
  Py_DECREF(po_dp);
  Py_DECREF(py_e_sus_m1);
  Py_DECREF(py_e_sus_m2);
  Py_XDECREF(py_sus);

  return rval;
}

static int pymaterial_to_material(PyObject *po, material_type *mt) {
  material_data *md;

  if (PyObject_IsInstance(po, py_material_object())) {
    md = make_dielectric(1);
    if (!pymedium_to_medium(po, &md->medium)) { return 0; }
  }
  else if (PyObject_IsInstance(po, py_material_grid_object())) { // Material grid subclass
    PyObject *py_do_averaging = PyObject_GetAttrString(po, "do_averaging");
    bool do_averaging = false;
    if (py_do_averaging) { do_averaging = PyObject_IsTrue(py_do_averaging); }
    PyObject *py_beta = PyObject_GetAttrString(po, "beta");
    double beta = 0;
    if (py_beta) { beta = PyFloat_AsDouble(py_beta); }
    PyObject *py_eta = PyObject_GetAttrString(po, "eta");
    double eta = 0;
    if (py_eta) { eta = PyFloat_AsDouble(py_eta); }
    md = make_material_grid(do_averaging,beta,eta);
    if (!pymaterial_grid_to_material_grid(po, md)) { return 0; }
    Py_XDECREF(py_do_averaging);
    Py_XDECREF(py_beta);
    Py_XDECREF(py_eta);
  }
  else if (PyFunction_Check(po)) {
    PyObject *eps = PyObject_GetAttrString(po, "eps");
    PyObject *py_do_averaging = PyObject_GetAttrString(po, "do_averaging");
    PyErr_Clear(); // clear errors from attributes not present
    bool do_averaging = false;
    if (py_do_averaging) { do_averaging = PyObject_IsTrue(py_do_averaging); }
    if (eps && PyObject_IsTrue(eps)) { md = make_user_material(py_epsilon_func_wrap, po, do_averaging); }
    else {
      md = make_user_material(py_user_material_func_wrap, po, do_averaging);
    }
    Py_XDECREF(eps);
    Py_XDECREF(py_do_averaging);
  }
  else if (IsPyString(po)) {
    const char *eps_input_file = PyObject_ToCharPtr(po);
    md = make_file_material(eps_input_file);
  }
  else if (PyArray_Check(po)) {
    PyArrayObject *pao = (PyArrayObject *)po;

    if (!PyArray_ISCARRAY(pao)) { meep::abort("Numpy array must be C-style contiguous."); }
    md = new material_data();
    md->which_subclass = material_data::MATERIAL_FILE;
    md->epsilon_dims[0] = md->epsilon_dims[1] = md->epsilon_dims[2] = 1;
    md->epsilon_data = new double[PyArray_SIZE(pao)];
    memcpy(md->epsilon_data, (double *)PyArray_DATA(pao), PyArray_SIZE(pao) * sizeof(double));

    for (int i = 0; i < PyArray_NDIM(pao); ++i) {
      md->epsilon_dims[i] = (size_t)PyArray_DIMS(pao)[i];
    }

    master_printf("read in %zdx%zdx%zd numpy array for epsilon\n", md->epsilon_dims[0],
                  md->epsilon_dims[1], md->epsilon_dims[2]);
  }
  else {
    meep::abort("Expected a Medium, a Material Grid, a function, or a filename");
  }

  *mt = md;

  return 1;
}

template <class T> static void set_v3_on_pyobj(PyObject *py_obj, const T *v3, const char *attr) {
  PyObject *v3_class = py_vector3_object();
  PyObject *v3_args = Py_BuildValue("(d,d,d)", v3->x, v3->y, v3->z);
  PyObject *pyv3 = PyObject_Call(v3_class, v3_args, NULL);
  PyObject_SetAttrString(py_obj, attr, pyv3);

  Py_DECREF(v3_args);
  Py_DECREF(pyv3);
}

static PyObject *susceptibility_to_py_obj(const susceptibility_struct *s) {
  // Return value: New reference
  PyObject *geom_mod = get_geom_mod();

  PyObject *res;
  PyObject *args = PyTuple_New(0);

  if (s->saturated_gyrotropy || s->bias.x || s->bias.y || s->bias.z) {
    if (s->saturated_gyrotropy) {
      PyObject *py_gyrotropic_class =
          PyObject_GetAttrString(geom_mod, "GyrotropicSaturatedSusceptibility");
      res = PyObject_Call(py_gyrotropic_class, args, NULL);
      Py_DECREF(py_gyrotropic_class);
      PyObject *py_alpha = PyFloat_FromDouble(s->alpha);
      PyObject_SetAttrString(res, "alpha", py_alpha);
      Py_DECREF(py_alpha);
    }
    else if (s->drude) {
      PyObject *py_gyrotropic_drude_class =
          PyObject_GetAttrString(geom_mod, "GyrotropicDrudeSusceptibility");
      res = PyObject_Call(py_gyrotropic_drude_class, args, NULL);
      Py_DECREF(py_gyrotropic_drude_class);
    }
    else {
      PyObject *py_gyrotropic_lorentz_class =
          PyObject_GetAttrString(geom_mod, "GyrotropicLorentzianSusceptibility");
      res = PyObject_Call(py_gyrotropic_lorentz_class, args, NULL);
      Py_DECREF(py_gyrotropic_lorentz_class);
    }
    PyObject *py_bias = vec2py(vector3_to_vec(s->bias));
    PyObject_SetAttrString(res, "bias", py_bias);
    Py_DECREF(py_bias);
  }
  else if (s->noise_amp == 0) {
    if (s->drude) {
      PyObject *py_drude_class = PyObject_GetAttrString(geom_mod, "DrudeSusceptibility");
      res = PyObject_Call(py_drude_class, args, NULL);
      Py_DECREF(py_drude_class);
    }
    else {
      PyObject *py_lorentz_class = PyObject_GetAttrString(geom_mod, "LorentzianSusceptibility");
      res = PyObject_Call(py_lorentz_class, args, NULL);
      Py_DECREF(py_lorentz_class);
    }
  }
  else {
    if (s->drude) {
      PyObject *py_noisy_drude_class = PyObject_GetAttrString(geom_mod, "NoisyDrudeSusceptibility");
      res = PyObject_Call(py_noisy_drude_class, args, NULL);
      Py_DECREF(py_noisy_drude_class);
    }
    else {
      PyObject *py_noisy_lorentz_class =
          PyObject_GetAttrString(geom_mod, "NoisyLorentzianSusceptibility");
      res = PyObject_Call(py_noisy_lorentz_class, args, NULL);
      Py_DECREF(py_noisy_lorentz_class);
    }
    PyObject *py_noise = PyFloat_FromDouble(s->noise_amp);
    PyObject_SetAttrString(res, "noise_amp", py_noise);
    Py_DECREF(py_noise);
  }

  set_v3_on_pyobj<vector3>(res, &s->sigma_diag, "sigma_diag");
  set_v3_on_pyobj<vector3>(res, &s->sigma_offdiag, "sigma_offdiag");

  PyObject *py_freq = PyFloat_FromDouble(s->frequency);
  PyObject *py_gamma = PyFloat_FromDouble(s->gamma);

  PyObject_SetAttrString(res, "frequency", py_freq);
  PyObject_SetAttrString(res, "gamma", py_gamma);

  Py_DECREF(args);
  Py_DECREF(py_freq);
  Py_DECREF(py_gamma);

  return res;
}

static PyObject *susceptibility_list_to_py_list(const susceptibility_list *sl) {
  // Return value: New reference
  PyObject *res = PyList_New(sl->size());

  for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(sl->size()); ++i) {
    PyList_SetItem(res, i, susceptibility_to_py_obj(&sl->at(i)));
  }

  return res;
}

static PyObject *material_to_py_material(material_type mat) {
  // Return value: New reference
  switch (mat->which_subclass) {
    case meep_geom::material_data::MEDIUM: {
      PyObject *geom_mod = get_geom_mod();
      PyObject *medium_class = PyObject_GetAttrString(geom_mod, "Medium");

      PyObject *medium_args = PyTuple_New(0);
      PyObject *py_mat = PyObject_Call(medium_class, medium_args, NULL);

      PyObject *py_E_sus = susceptibility_list_to_py_list(&mat->medium.E_susceptibilities);
      PyObject *py_H_sus = susceptibility_list_to_py_list(&mat->medium.H_susceptibilities);
      PyObject_SetAttrString(py_mat, "E_susceptibilities", py_E_sus);
      PyObject_SetAttrString(py_mat, "H_susceptibilities", py_H_sus);

      set_v3_on_pyobj<vector3>(py_mat, &mat->medium.epsilon_diag, "epsilon_diag");
      set_v3_on_pyobj<vector3>(py_mat, &mat->medium.mu_diag, "mu_diag");
      set_v3_on_pyobj<vector3>(py_mat, &mat->medium.E_chi2_diag, "E_chi2_diag");
      set_v3_on_pyobj<vector3>(py_mat, &mat->medium.E_chi3_diag, "E_chi3_diag");
      set_v3_on_pyobj<vector3>(py_mat, &mat->medium.H_chi2_diag, "H_chi2_diag");
      set_v3_on_pyobj<vector3>(py_mat, &mat->medium.H_chi3_diag, "H_chi3_diag");
      set_v3_on_pyobj<vector3>(py_mat, &mat->medium.D_conductivity_diag, "D_conductivity_diag");
      set_v3_on_pyobj<vector3>(py_mat, &mat->medium.B_conductivity_diag, "B_conductivity_diag");
      set_v3_on_pyobj<cvector3>(py_mat, &mat->medium.epsilon_offdiag, "epsilon_offdiag");
      set_v3_on_pyobj<cvector3>(py_mat, &mat->medium.mu_offdiag, "mu_offdiag");

      Py_DECREF(medium_args);
      Py_DECREF(medium_class);
      Py_DECREF(py_E_sus);
      Py_DECREF(py_H_sus);

      return py_mat;
    }
    case meep_geom::material_data::MATERIAL_USER: {
      PyObject *py_mat = (PyObject *) mat->user_data;
      Py_INCREF(py_mat);
      return py_mat;
    }
    default:
      // Only Medium is supported at this time.
      meep::abort("Can only convert C++ medium_struct subtype %d to Python", mat->which_subclass);
  }
}

static int pymedium_to_medium(PyObject *po, medium_struct *m) {
  if (!get_attr_v3(po, &m->epsilon_diag, "epsilon_diag") ||
      !get_attr_v3(po, &m->mu_diag, "mu_diag")) {

    return 0;
  }
  if (!get_attr_v3_cmplx(po, &m->mu_offdiag, "mu_offdiag") ||
      !get_attr_v3_cmplx(po, &m->epsilon_offdiag, "epsilon_offdiag")) {

    return 0;
  }

  PyObject *py_e_susceptibilities = PyObject_GetAttrString(po, "E_susceptibilities");
  PyObject *py_h_susceptibilities = PyObject_GetAttrString(po, "H_susceptibilities");

  if (!py_e_susceptibilities || !py_h_susceptibilities) {
    Py_XDECREF(py_e_susceptibilities);
    Py_XDECREF(py_h_susceptibilities);
    return 0;
  }

  if (!py_list_to_susceptibility_list(py_e_susceptibilities, &m->E_susceptibilities) ||
      !py_list_to_susceptibility_list(py_h_susceptibilities, &m->H_susceptibilities)) {
    Py_XDECREF(py_e_susceptibilities);
    Py_XDECREF(py_h_susceptibilities);
    return 0;
  }

  Py_XDECREF(py_e_susceptibilities);
  Py_XDECREF(py_h_susceptibilities);

  if (!get_attr_v3(po, &m->E_chi2_diag, "E_chi2_diag") ||
      !get_attr_v3(po, &m->E_chi3_diag, "E_chi3_diag") ||
      !get_attr_v3(po, &m->H_chi2_diag, "H_chi2_diag") ||
      !get_attr_v3(po, &m->H_chi3_diag, "H_chi3_diag") ||
      !get_attr_v3(po, &m->D_conductivity_diag, "D_conductivity_diag") ||
      !get_attr_v3(po, &m->B_conductivity_diag, "B_conductivity_diag")) {

    return 0;
  }

  return 1;
}

static int pysphere_to_sphere(PyObject *py_sphere, geometric_object *go) {

  material_type material;
  vector3 center;
  double radius;

  if (!get_attr_v3(py_sphere, &center, "center") || !get_attr_dbl(py_sphere, &radius, "radius") ||
      !get_attr_material(py_sphere, &material)) {

    go->subclass.sphere_data = NULL;
    return 0;
  }

  *go = make_sphere(material, center, radius);

  return 1;
}

static int pycylinder_to_cylinder(PyObject *py_cyl, geometric_object *cyl) {
  material_type material;
  vector3 center, axis;
  double radius, height;

  if (!get_attr_v3(py_cyl, &center, "center") || !get_attr_v3(py_cyl, &axis, "axis") ||
      !get_attr_dbl(py_cyl, &radius, "radius") || !get_attr_dbl(py_cyl, &height, "height") ||
      !get_attr_material(py_cyl, &material)) {

    cyl->subclass.cylinder_data = NULL;
    return 0;
  }

  *cyl = make_cylinder(material, center, radius, height, axis);

  return 1;
}

static int pywedge_to_wedge(PyObject *py_wedge, geometric_object *wedge) {
  geometric_object cyl;
  if (!pycylinder_to_cylinder(py_wedge, &cyl)) { return 0; }

  double wedge_angle;
  vector3 wedge_start;

  if (!get_attr_dbl(py_wedge, &wedge_angle, "wedge_angle") ||
      !get_attr_v3(py_wedge, &wedge_start, "wedge_start")) {

    wedge->subclass.cylinder_data = NULL;
    geometric_object_destroy(cyl);
    return 0;
  }

  double radius = cyl.subclass.cylinder_data->radius;
  double height = cyl.subclass.cylinder_data->height;
  vector3 axis = cyl.subclass.cylinder_data->axis;

  *wedge = make_wedge(cyl.material, cyl.center, radius, height, axis, wedge_angle, wedge_start);

  geometric_object_destroy(cyl);

  return 1;
}

static int pycone_to_cone(PyObject *py_cone, geometric_object *cone) {
  geometric_object cyl;
  if (!pycylinder_to_cylinder(py_cone, &cyl)) { return 0; }

  double radius2;
  if (!get_attr_dbl(py_cone, &radius2, "radius2")) {
    cone->subclass.cylinder_data = NULL;
    geometric_object_destroy(cyl);
    return 0;
  }

  double radius = cyl.subclass.cylinder_data->radius;
  double height = cyl.subclass.cylinder_data->height;
  vector3 axis = cyl.subclass.cylinder_data->axis;

  *cone = make_cone(cyl.material, cyl.center, radius, height, axis, radius2);

  geometric_object_destroy(cyl);

  return 1;
}

static int pyblock_to_block(PyObject *py_blk, geometric_object *blk) {
  material_type material;
  vector3 center, e1, e2, e3, size;

  if (!get_attr_material(py_blk, &material) || !get_attr_v3(py_blk, &center, "center") ||
      !get_attr_v3(py_blk, &e1, "e1") || !get_attr_v3(py_blk, &e2, "e2") ||
      !get_attr_v3(py_blk, &e3, "e3") || !get_attr_v3(py_blk, &size, "size")) {

    blk->subclass.block_data = NULL;
    return 0;
  }

  *blk = make_block(material, center, e1, e2, e3, size);
  return 1;
}

static int pyellipsoid_to_ellipsoid(PyObject *py_ell, geometric_object *e) {
  geometric_object blk;
  if (!pyblock_to_block(py_ell, &blk)) { return 0; }

  material_type material = (material_type)blk.material;
  vector3 center = blk.center;
  vector3 e1 = blk.subclass.block_data->e1;
  vector3 e2 = blk.subclass.block_data->e2;
  vector3 e3 = blk.subclass.block_data->e3;
  vector3 size = blk.subclass.block_data->size;

  *e = make_ellipsoid(material, center, e1, e2, e3, size);

  geometric_object_destroy(blk);

  return 1;
}

static int pyprism_to_prism(PyObject *py_prism, geometric_object *p) {
  material_type material;
  double height, sidewall_angle;
  vector3 axis, center;

  if (!get_attr_material(py_prism, &material) || !get_attr_dbl(py_prism, &height, "height") ||
      !get_attr_dbl(py_prism, &sidewall_angle, "sidewall_angle") ||
      !get_attr_v3(py_prism, &center, "center") || !get_attr_v3(py_prism, &axis, "axis")) {

    return 0;
  }

  PyObject *py_vert_list = PyObject_GetAttrString(py_prism, "vertices");

  if (!py_vert_list) { abort_with_stack_trace(); }

  if (!PyList_Check(py_vert_list)) { meep::abort("Expected Prism.vertices to be a list\n"); }

  int num_vertices = PyList_Size(py_vert_list);
  vector3 *vertices = new vector3[num_vertices];

  for (Py_ssize_t i = 0; i < num_vertices; ++i) {
    vector3 v3;
    if (!pyv3_to_v3(PyList_GetItem(py_vert_list, i), &v3)) {
      Py_DECREF(py_vert_list);
      return 0;
    }
    vertices[i] = v3;
  }

#if defined(LIBCTL_MAJOR_VERSION) &&                                                               \
    (LIBCTL_MAJOR_VERSION > 4 || (LIBCTL_MAJOR_VERSION == 4 && LIBCTL_MINOR_VERSION >= 5))
  *p = make_slanted_prism(material, vertices, num_vertices, height, axis, sidewall_angle);
#else
  if (sidewall_angle != 0) { meep::abort("slanted prisms require libctl 4.5 or later\n"); }
  *p = make_prism(material, vertices, num_vertices, height, axis);
#endif
  p->center = center;

  delete[] vertices;
  Py_DECREF(py_vert_list);

  return 1;
}

static int py_gobj_to_gobj(PyObject *po, geometric_object *o) {
  int success = 0;
  std::string go_type = py_class_name_as_string(po);

  if (go_type == "Sphere") { success = pysphere_to_sphere(po, o); }
  else if (go_type == "Cylinder") {
    success = pycylinder_to_cylinder(po, o);
  }
  else if (go_type == "Wedge") {
    success = pywedge_to_wedge(po, o);
  }
  else if (go_type == "Cone") {
    success = pycone_to_cone(po, o);
  }
  else if (go_type == "Block") {
    success = pyblock_to_block(po, o);
  }
  else if (go_type == "Ellipsoid") {
    success = pyellipsoid_to_ellipsoid(po, o);
  }
  else if (go_type == "Prism") {
    success = pyprism_to_prism(po, o);
  }
  else {
    meep::abort("Error: %s is not a valid GeometricObject type\n", go_type.c_str());
    return 0;
  }

  return success;
}

static int py_list_to_gobj_list(PyObject *po, geometric_object_list *l) {
  if (!PyList_Check(po)) { meep::abort("Expected a list"); }

  int length = PyList_Size(po);

  l->num_items = length;
  l->items = new geometric_object[length];

  for (int i = 0; i < length; i++) {
    PyObject *py_gobj = PyList_GetItem(po, i);
    if (!py_gobj_to_gobj(py_gobj, &l->items[i])) { return 0; }
  }

  return 1;
}

static PyObject *gobj_to_py_obj(geometric_object *gobj) {
  // Return value: New reference
  switch (gobj->which_subclass) {
    case geometric_object::PRISM: {
      PyObject *geom_mod = get_geom_mod();
      PyObject *prism_class = PyObject_GetAttrString(geom_mod, "Prism");

      int num_verts = gobj->subclass.prism_data->vertices.num_items;
      prism *prsm = gobj->subclass.prism_data;

      PyObject *py_verts = PyList_New(num_verts);

      for (int i = 0; i < num_verts; ++i) {
        PyList_SetItem(py_verts, i, v3_to_pyv3(prsm->vertices.items + i));
      }

      double height = prsm->height;
      vector3 axis = prsm->axis;
      PyObject *py_axis = v3_to_pyv3(&axis);

      PyObject *py_mat = material_to_py_material((meep_geom::material_type)gobj->material);

      PyObject *args = Py_BuildValue("(OdO)", py_verts, height, py_axis);
      PyObject *kwargs = Py_BuildValue("{s:O}", "material", py_mat);
      PyObject *res = PyObject_Call(prism_class, args, kwargs);

      Py_DECREF(prism_class);
      Py_DECREF(args);
      Py_DECREF(kwargs);
      Py_DECREF(py_verts);
      Py_DECREF(py_axis);
      Py_DECREF(py_mat);

      return res;
    }
    case geometric_object::BLOCK:
    case geometric_object::SPHERE:
    case geometric_object::CYLINDER:
    default:
      // We currently only have the need to create python Prisms from C++.
      // Other geometry can be added as needed.
      meep::abort("Conversion of non-prism geometric_object to Python is not supported");
  }
}

static PyObject *gobj_list_to_py_list(geometric_object_list *objs) {
  // Return value: New reference
  PyObject *py_res = PyList_New(objs->num_items);

  for (int i = 0; i < objs->num_items; ++i) {
    PyList_SetItem(py_res, i, gobj_to_py_obj(&objs->items[i]));
    geometric_object_destroy(objs->items[i]);
  }

  delete[] objs->items;

  return py_res;
}

void gobj_list_freearg(geometric_object_list* objs) {
    for(int i = 0; i < objs->num_items; ++i) {
        material_free((material_data *)objs->items[i].material);
        delete (material_data *)objs->items[i].material;
        geometric_object_destroy(objs->items[i]);
    }
    delete[] objs->items;
}

static std::unique_ptr<meep::binary_partition> py_bp_to_bp(PyObject *pybp) {
  std::unique_ptr<meep::binary_partition> bp;
  if (pybp == Py_None) return bp;

  PyObject *id = PyObject_GetAttrString(pybp, "proc_id");
  PyObject *split_dir = PyObject_GetAttrString(pybp, "split_dir");
  PyObject *split_pos = PyObject_GetAttrString(pybp, "split_pos");
  PyObject *left = PyObject_GetAttrString(pybp, "left");
  PyObject *right = PyObject_GetAttrString(pybp, "right");

  if (!id || !split_dir || !split_pos || !left || !right) {
    meep::abort("BinaryPartition class object is incorrectly defined.");
  }

  if (PyLong_Check(id)) { bp.reset(new meep::binary_partition(PyLong_AsLong(id))); }
  else {
    bp.reset(new meep::binary_partition(
        meep::split_plane{direction(PyLong_AsLong(split_dir)), PyFloat_AsDouble(split_pos)},
        py_bp_to_bp(left),
        py_bp_to_bp(right)));
  }

  Py_XDECREF(id);
  Py_XDECREF(split_dir);
  Py_XDECREF(split_pos);
  Py_XDECREF(left);
  Py_XDECREF(right);
  return bp;
}

static PyObject *py_binary_partition_object() {
  // Return value: Borrowed reference
  static PyObject *bp_type = NULL;
  if (bp_type == NULL) {
    bp_type = PyObject_GetAttrString(get_meep_mod(), "BinaryPartition");
  }
  return bp_type;
}

// Converts a meep::binary_partition object into a Python class instance
static PyObject *bp_to_py_bp(const meep::binary_partition *bp) {
  PyObject *bp_class = py_binary_partition_object();
  PyObject *args = PyTuple_New(0);  // no numbered arguments to pass
  if (bp->is_leaf()) {
    // leaf nodes will have proc_id and no other properties
    int proc_id = bp->get_proc_id();
    PyObject *kwargs = Py_BuildValue("{s:i}", "proc_id", proc_id);
    PyObject *py_bp = PyObject_Call(bp_class, args, kwargs);
    Py_DECREF(args);
    Py_DECREF(kwargs);
    return py_bp;
  } else {
    // other nodes will have left, right, split_dir, split_pos
    PyObject *left = bp_to_py_bp(bp->left_tree());
    PyObject *right = bp_to_py_bp(bp->right_tree());
    meep::direction split_dir = bp->get_plane().dir;
    double split_pos = bp->get_plane().pos;
    PyObject *kwargs =
        Py_BuildValue("{s:O,s:O,s:i,s:d}", "left", left, "right", right,
                      "split_dir", split_dir, "split_pos", split_pos);
    PyObject *py_bp = PyObject_Call(bp_class, args, kwargs);
    Py_DECREF(args);
    Py_DECREF(kwargs);
    return py_bp;
  }
}
