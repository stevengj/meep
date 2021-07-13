#include <algorithm>
#include <complex>
#include <cstdlib>
#include <cstddef>
#include <iostream>

#include "config.h"
#include "pympb.hpp"
#include "meep/mympi.hpp"

// If the MPB lib is not new enough to have the mpb_verbosity global then make
// one here to give the swig wrapper something to link with.
#if MPB_VERSION_MAJOR > 1 || (MPB_VERSION_MAJOR == 1 && MPB_VERSION_MINOR >= 11)
// do nothing, libmpb should have the mpb_verbosity symbol
#else
extern "C" int mpb_verbosity = 2;
#endif

// xyz_loop.h
#ifndef HAVE_MPI
#define LOOP_XYZ(md)                                                                               \
  {                                                                                                \
    int n1 = md->nx, n2 = md->ny, n3 = md->nz, i1, i2, i3;                                         \
    for (i1 = 0; i1 < n1; ++i1)                                                                    \
      for (i2 = 0; i2 < n2; ++i2)                                                                  \
        for (i3 = 0; i3 < n3; ++i3) {                                                              \
          int xyz_index = ((i1 * n2 + i2) * n3 + i3);
#else /* HAVE_MPI */
/* first two dimensions are transposed in MPI output: */
#define LOOP_XYZ(md)                                                                               \
  {                                                                                                \
    int n1 = md->nx, n2 = md->ny, n3 = md->nz, i1, i2_, i3;                                        \
    int local_n2 = md->local_ny, local_y_start = md->local_y_start;                                \
    for (i2_ = 0; i2_ < local_n2; ++i2_)                                                           \
      for (i1 = 0; i1 < n1; ++i1)                                                                  \
        for (i3 = 0; i3 < n3; ++i3) {                                                              \
          int i2 = i2_ + local_y_start;                                                            \
          int xyz_index = ((i2_ * n1 + i1) * n3 + i3);
#endif /* HAVE_MPI */

typedef mpb_real real; // needed for the CASSIGN macros below

// TODO: Support MPI
#define mpi_allreduce(sb, rb, n, ctype, t, op, comm)                                               \
  {                                                                                                \
    CHECK((sb) != (rb), "MPI_Allreduce doesn't work for sendbuf == recvbuf");                      \
    memcpy((rb), (sb), (n) * sizeof(ctype));                                                       \
  }

/* "in-place" Allreduce wrapper for reducing a single value */
#define mpi_allreduce_1(b, ctype, t, op, comm)                                                     \
  {                                                                                                \
    ctype bbbb = *(b);                                                                             \
    mpi_allreduce(&bbbb, (b), 1, ctype, t, op, comm);                                              \
  }

#ifdef CHECK_DISABLE
#define CHECK(cond, s) // Do nothing
#else
#define CHECK(cond, s)                                                                             \
  if (!(cond)) { meep::abort(s "\n"); }
#endif

namespace py_mpb {



// TODO: Placeholder
int mpb_comm;

const double inf = 1.0e20;

// This is the function passed to `set_maxwell_dielectric`
static void dielectric_function(symmetric_matrix *eps, symmetric_matrix *eps_inv,
                                const mpb_real r[3], void *epsilon_data) {

  mode_solver *ms = static_cast<mode_solver *>(epsilon_data);
  meep_geom::material_type mat;
  vector3 p;

  // p needs to be in the lattice *unit* vector basis, while r is in the lattice
  // vector basis.  Also, shift origin to the center of the grid.
  p.x = (r[0] - 0.5) * geometry_lattice.size.x;
  p.y = (r[1] - 0.5) * geometry_lattice.size.y;
  p.z = (r[2] - 0.5) * geometry_lattice.size.z;

  // p = shift_to_unit_cell(p);

  ms->get_material_pt(mat, p);
  ms->material_epsmu(mat, eps, eps_inv, eps);
}

static int mean_epsilon_func(symmetric_matrix *meps, symmetric_matrix *meps_inv, mpb_real n[3],
                             mpb_real d1, mpb_real d2, mpb_real d3, mpb_real tol,
                             const mpb_real r[3], void *edata) {

  mode_solver *ms = static_cast<mode_solver *>(edata);
  return ms->mean_epsilon(meps, meps_inv, n, d1, d2, d3, tol, r);
}

/****** utils ******/

/* a couple of utilities to convert libctl data types to the data
   types of the eigensolver & maxwell routines: */

void vector3_to_arr(mpb_real arr[3], vector3 v) {
  arr[0] = v.x;
  arr[1] = v.y;
  arr[2] = v.z;
}

void matrix3x3_to_arr(mpb_real arr[3][3], matrix3x3 m) {
  vector3_to_arr(arr[0], m.c0);
  vector3_to_arr(arr[1], m.c1);
  vector3_to_arr(arr[2], m.c2);
}

cnumber cscalar2cnumber(scalar_complex cs) { return make_cnumber(CSCALAR_RE(cs), CSCALAR_IM(cs)); }

// Return a string describing the current parity, used for frequency and filename
// prefixes
const char *parity_string(maxwell_data *d) {
  static char s[128];
  strcpy(s, "");
  if (d->parity & EVEN_Z_PARITY) { strcat(s, (d->nz == 1) ? "te" : "zeven"); }
  else if (d->parity & ODD_Z_PARITY) {
    strcat(s, (d->nz == 1) ? "tm" : "zodd");
  }
  if (d->parity & EVEN_Y_PARITY) { strcat(s, "yeven"); }
  else if (d->parity & ODD_Y_PARITY) {
    strcat(s, "yodd");
  }
  return s;
}

/* Extract the mean epsilon from the effective inverse dielectric tensor,
   which contains two eigenvalues that correspond to the mean epsilon,
   and one which corresponds to the harmonic mean. */
mpb_real mean_medium_from_matrix(const symmetric_matrix *eps_inv) {
  mpb_real eps_eigs[3];
  maxwell_sym_matrix_eigs(eps_eigs, eps_inv);
  /* the harmonic mean should be the largest eigenvalue (smallest
     epsilon), so we'll ignore it and average the other two: */
  return 2.0 / (eps_eigs[0] + eps_eigs[1]);
}

/* When we are solving for a few bands at a time, we solve for the
   upper bands by "deflation"--by continually orthogonalizing them
   against the already-computed lower bands.  (This constraint
   commutes with the eigen-operator, of course, so all is well.) */

typedef struct {
  evectmatrix Y;  /* the vectors to orthogonalize against; Y must
                     itself be normalized (Yt B Y = 1) */
  evectmatrix BY; /* B * Y */
  int p;          /* the number of columns of Y to orthogonalize against */
  scalar *S;      /* a matrix for storing the dot products; should have
                     at least p * X.p elements (see below for X) */
  scalar *S2;     /* a scratch matrix the same size as S */
} deflation_data;

extern "C" {
void blasglue_gemm(char transa, char transb, int m, int n, int k, mpb_real a, scalar *A, int fdA,
                   scalar *B, int fdB, mpb_real b, scalar *C, int fdC);
}

static void deflation_constraint(evectmatrix X, void *data) {
  deflation_data *d = (deflation_data *)data;

  CHECK(X.n == d->BY.n && d->BY.p >= d->p && d->Y.p >= d->p, "invalid dimensions");

  /* compute (1 - Y (BY)t) X = (1 - Y Yt B) X
      = projection of X so that Yt B X = 0 */

  /* (Sigh...call the BLAS functions directly since we are not
     using all the columns of BY...evectmatrix is not set up for
     this case.) */

  /* compute S = Xt BY (i.e. all the dot products): */
  blasglue_gemm('C', 'N', X.p, d->p, X.n, 1.0, X.data, X.p, d->BY.data, d->BY.p, 0.0, d->S2, d->p);
  // TODO
  // #if HAVE_MPI
  //   MPI_Allreduce(d->S2, d->S, d->p * X.p * SCALAR_NUMVALS, SCALAR_MPI_TYPE,
  //                 MPI_SUM, mpb_comm);
  // #else
  memcpy(d->S, d->S2, sizeof(mpb_real) * d->p * X.p * SCALAR_NUMVALS);
  // #endif

  /* compute X = X - Y*St = (1 - BY Yt B) X */
  blasglue_gemm('N', 'C', X.n, X.p, d->p, -1.0, d->Y.data, d->Y.p, d->S, d->p, 1.0, X.data, X.p);
}

/******* mode_solver *******/

mode_solver::mode_solver(int num_bands, double resolution[3], lattice lat, double tolerance,
                         int mesh_size, meep_geom::material_data *_default_material,
                         bool deterministic, double target_freq, int dims, bool verbose,
                         bool periodicity, double flops, bool negative_epsilon_ok,
                         std::string epsilon_input_file, std::string mu_input_file, bool force_mu,
                         bool use_simple_preconditioner, vector3 grid_size, int eigensolver_nwork,
                         int eigensolver_block_size)
    : num_bands(num_bands), resolution{resolution[0], resolution[1], resolution[2]},
      target_freq(target_freq), tolerance(tolerance), mesh_size(mesh_size),
      negative_epsilon_ok(negative_epsilon_ok), epsilon_input_file(epsilon_input_file),
      mu_input_file(mu_input_file), force_mu(force_mu),
      use_simple_preconditioner(use_simple_preconditioner), grid_size(grid_size), nwork_alloc(0),
      eigensolver_nwork(eigensolver_nwork), eigensolver_block_size(eigensolver_block_size),
      last_parity(-2), iterations(0), eigensolver_flops(flops), geometry_list{},
      geometry_tree(NULL),  vol(0), R{}, G{},  mdata(NULL), mtdata(NULL),
      curfield_band(0), H{}, Hblock{}, muinvH{}, W{}, freqs(num_bands), verbose(verbose),
      deterministic(deterministic), kpoint_index(0), curfield(NULL), curfield_type('-'), eps(true) {

  // See geom-ctl-io-defaults.c in libctl
  geometry_lattice = lat;
  dimensions = dims;
  ensure_periodicity = periodicity;

#ifndef WITH_HERMITIAN_EPSILON
  meep_geom::medium_struct *m;
  if (meep_geom::is_medium(_default_material, &m)) { m->check_offdiag_im_zero_or_abort(); }
#else
  (void)_default_material;
#endif
}

mode_solver::~mode_solver() {
  destroy_maxwell_data(mdata);
  destroy_maxwell_target_data(mtdata);
  destroy_geom_box_tree(geometry_tree);
  clear_geometry_list();
  destroy_evectmatrix(H);

  for (int i = 0; i < nwork_alloc; ++i) {
    destroy_evectmatrix(W[i]);
  }

  if (Hblock.data != H.data) { destroy_evectmatrix(Hblock); }

  if (muinvH.data != H.data) { destroy_evectmatrix(muinvH); }
}

int mode_solver::mean_epsilon(symmetric_matrix *meps, symmetric_matrix *meps_inv, mpb_real n[3],
                              mpb_real d1, mpb_real d2, mpb_real d3, mpb_real tol,
                              const mpb_real r[3]) {

  const geometric_object *o1 = 0;
  const geometric_object *o2 = 0;
  geom_box pixel;
  double fill;
  meep_geom::material_type mat1;
  meep_geom::material_type mat2;

  int id1 = -1;
  int id2 = -1;

  const int num_neighbors[3] = {3, 5, 9};
  const int neighbors[3][9][3] = {{{0, 0, 0},
                                   {-1, 0, 0},
                                   {1, 0, 0},
                                   {0, 0, 0},
                                   {0, 0, 0},
                                   {0, 0, 0},
                                   {0, 0, 0},
                                   {0, 0, 0},
                                   {0, 0, 0}},
                                  {{0, 0, 0},
                                   {-1, -1, 0},
                                   {1, 1, 0},
                                   {-1, 1, 0},
                                   {1, -1, 0},
                                   {0, 0, 0},
                                   {0, 0, 0},
                                   {0, 0, 0},
                                   {0, 0, 0}},
                                  {{0, 0, 0},
                                   {1, 1, 1},
                                   {1, 1, -1},
                                   {1, -1, 1},
                                   {1, -1, -1},
                                   {-1, 1, 1},
                                   {-1, 1, -1},
                                   {-1, -1, 1},
                                   {-1, -1, -1}}};

  /* p needs to be in the lattice *unit* vector basis, while r is
     in the lattice vector basis.  Also, shift origin to the center
     of the grid. */
  vector3 p = {(r[0] - 0.5) * geometry_lattice.size.x, (r[1] - 0.5) * geometry_lattice.size.y,
               (r[2] - 0.5) * geometry_lattice.size.z};

  d1 *= geometry_lattice.size.x * 0.5;
  d2 *= geometry_lattice.size.y * 0.5;
  d3 *= geometry_lattice.size.z * 0.5;

  vector3 shiftby1;
  vector3 shiftby2;
  vector3 normal;

  for (int i = 0; i < num_neighbors[dimensions - 1]; ++i) {
    const geometric_object *o;
    vector3 q, z, shiftby;
    int id;
    q.x = p.x + neighbors[dimensions - 1][i][0] * d1;
    q.y = p.y + neighbors[dimensions - 1][i][1] * d2;
    q.z = p.z + neighbors[dimensions - 1][i][2] * d3;

    geometry_lattice.size.x = geometry_lattice.size.x == 0 ? 1e-20 : geometry_lattice.size.x;
    geometry_lattice.size.y = geometry_lattice.size.y == 0 ? 1e-20 : geometry_lattice.size.y;
    geometry_lattice.size.z = geometry_lattice.size.z == 0 ? 1e-20 : geometry_lattice.size.z;

    z = shift_to_unit_cell(q);

    geometry_lattice.size.x = geometry_lattice.size.x == 1e-20 ? 0 : geometry_lattice.size.x;
    geometry_lattice.size.y = geometry_lattice.size.y == 1e-20 ? 0 : geometry_lattice.size.y;
    geometry_lattice.size.z = geometry_lattice.size.z == 1e-20 ? 0 : geometry_lattice.size.z;

    o = object_of_point_in_tree(z, geometry_tree, &shiftby, &id);
    shiftby = vector3_plus(shiftby, vector3_minus(q, z));

    if ((id == id1 && vector3_equal(shiftby, shiftby1)) ||
        (id == id2 && vector3_equal(shiftby, shiftby2))) {
      continue;
    }

    meep_geom::material_type mat = (meep_geom::material_type)default_material;
    if (o) {
      meep_geom::material_data *md = (meep_geom::material_data *)o->material;
      if (md->which_subclass != meep_geom::material_data::MATERIAL_FILE) { mat = md; }
    }

    if (id1 == -1) {
      o1 = o;
      shiftby1 = shiftby;
      id1 = id;
      mat1 = mat;
    }
    else if (id2 == -1 || ((id >= id1 && id >= id2) &&
                           (id1 == id2 || meep_geom::material_type_equal(mat1, mat2)))) {
      o2 = o;
      shiftby2 = shiftby;
      id2 = id;
      mat2 = mat;
    }
    else if (!(id1 < id2 && (id1 == id || meep_geom::material_type_equal(mat1, mat))) &&
             !(id2 < id1 && (id2 == id || meep_geom::material_type_equal(mat2, mat)))) {
      return 0; /* too many nearby objects for analysis */
    }
  }

  CHECK(id1 > -1, "bug in object_of_point_in_tree?");
  if (id2 == -1) { /* only one nearby object/material */
    id2 = id1;
    o2 = o1;
    mat2 = mat1;
    shiftby2 = shiftby1;
  }

  bool o1_is_var = o1 && meep_geom::is_variable(o1->material);
  bool o2_is_var = o2 && meep_geom::is_variable(o2->material);
  bool default_is_var = meep_geom::is_variable(default_material);

  if (o1_is_var || o2_is_var || (default_is_var && (!o1 || !o2))) {
    return 0; /* arbitrary material functions are non-analyzable */
  }

  material_epsmu(mat1, meps, meps_inv, eps);

  /* check for trivial case of only one object/material */
  if (id1 == id2 || meep_geom::material_type_equal(mat1, mat2)) {
    n[0] = n[1] = n[2] = 0;
    return 1;
  }

  if (id1 > id2) { normal = normal_to_fixed_object(vector3_minus(p, shiftby1), *o1); }
  else {
    normal = normal_to_fixed_object(vector3_minus(p, shiftby2), *o2);
  }

  n[0] = normal.x / (geometry_lattice.size.x == 0 ? 1e-20 : geometry_lattice.size.x);
  n[1] = normal.y / (geometry_lattice.size.y == 0 ? 1e-20 : geometry_lattice.size.y);
  n[2] = normal.z / (geometry_lattice.size.z == 0 ? 1e-20 : geometry_lattice.size.z);

  pixel.low.x = p.x - d1;
  pixel.high.x = p.x + d1;
  pixel.low.y = p.y - d2;
  pixel.high.y = p.y + d2;
  pixel.low.z = p.z - d3;
  pixel.high.z = p.z + d3;

  tol = tol > 0.01 ? 0.01 : tol;
  if (id1 > id2) {
    pixel.low = vector3_minus(pixel.low, shiftby1);
    pixel.high = vector3_minus(pixel.high, shiftby1);
    fill = box_overlap_with_object(pixel, *o1, tol, 100 / tol);
  }
  else {
    pixel.low = vector3_minus(pixel.low, shiftby2);
    pixel.high = vector3_minus(pixel.high, shiftby2);
    fill = 1 - box_overlap_with_object(pixel, *o2, tol, 100 / tol);
  }

  {
    symmetric_matrix eps2, epsinv2;
    symmetric_matrix eps1, delta;
    double Rot[3][3], norm, n0, n1, n2;
    material_epsmu(mat2, &eps2, &epsinv2, eps);
    eps1 = *meps;

    /* make Cartesian orthonormal frame relative to interface */
    n0 = R[0][0] * n[0] + R[1][0] * n[1] + R[2][0] * n[2];
    n1 = R[0][1] * n[0] + R[1][1] * n[1] + R[2][1] * n[2];
    n2 = R[0][2] * n[0] + R[1][2] * n[1] + R[2][2] * n[2];
    norm = sqrt(n0 * n0 + n1 * n1 + n2 * n2);

    if (norm == 0.0) { return 0; }

    norm = 1.0 / norm;
    Rot[0][0] = n0 = n0 * norm;
    Rot[1][0] = n1 = n1 * norm;
    Rot[2][0] = n2 = n2 * norm;

    if (fabs(n0) > 1e-2 || fabs(n1) > 1e-2) { /* (z x n) */
      Rot[0][2] = n1;
      Rot[1][2] = -n0;
      Rot[2][2] = 0;
    }
    else { /* n is ~ parallel to z direction, use (x x n) instead */
      Rot[0][2] = 0;
      Rot[1][2] = -n2;
      Rot[2][2] = n1;
    }
    { /* normalize second column */
      double s = Rot[0][2] * Rot[0][2] + Rot[1][2] * Rot[1][2] + Rot[2][2] * Rot[2][2];
      s = 1.0 / sqrt(s);
      Rot[0][2] *= s;
      Rot[1][2] *= s;
      Rot[2][2] *= s;
    }
    /* 1st column is 2nd column x 0th column */
    Rot[0][1] = Rot[1][2] * Rot[2][0] - Rot[2][2] * Rot[1][0];
    Rot[1][1] = Rot[2][2] * Rot[0][0] - Rot[0][2] * Rot[2][0];
    Rot[2][1] = Rot[0][2] * Rot[1][0] - Rot[1][2] * Rot[0][0];

    /* rotate epsilon tensors to surface parallel/perpendicular axes */
    maxwell_sym_matrix_rotate(&eps1, &eps1, Rot);
    maxwell_sym_matrix_rotate(&eps2, &eps2, Rot);

#define AVG (fill * (EXPR(eps1)) + (1 - fill) * (EXPR(eps2)))

#define EXPR(eps) (-1 / eps.m00)
    delta.m00 = AVG;
#undef EXPR
#define EXPR(eps) (eps.m11 - ESCALAR_NORMSQR(eps.m01) / eps.m00)
    delta.m11 = AVG;
#undef EXPR
#define EXPR(eps) (eps.m22 - ESCALAR_NORMSQR(eps.m02) / eps.m00)
    delta.m22 = AVG;
#undef EXPR

#define EXPR(eps) (ESCALAR_RE(eps.m01) / eps.m00)
    ESCALAR_RE(delta.m01) = AVG;
#undef EXPR
#define EXPR(eps) (ESCALAR_RE(eps.m02) / eps.m00)
    ESCALAR_RE(delta.m02) = AVG;
#undef EXPR
#define EXPR(eps) (ESCALAR_RE(eps.m12) - ESCALAR_MULT_CONJ_RE(eps.m02, eps.m01) / eps.m00)
    ESCALAR_RE(delta.m12) = AVG;
#undef EXPR

#ifdef WITH_HERMITIAN_EPSILON
#define EXPR(eps) (ESCALAR_IM(eps.m01) / eps.m00)
    ESCALAR_IM(delta.m01) = AVG;
#undef EXPR
#define EXPR(eps) (ESCALAR_IM(eps.m02) / eps.m00)
    ESCALAR_IM(delta.m02) = AVG;
#undef EXPR
#define EXPR(eps) (ESCALAR_IM(eps.m12) - ESCALAR_MULT_CONJ_IM(eps.m02, eps.m01) / eps.m00)
    ESCALAR_IM(delta.m12) = AVG;
#undef EXPR
#endif /* WITH_HERMITIAN_EPSILON */

    meps->m00 = -1 / delta.m00;
    meps->m11 = delta.m11 - ESCALAR_NORMSQR(delta.m01) / delta.m00;
    meps->m22 = delta.m22 - ESCALAR_NORMSQR(delta.m02) / delta.m00;
    ASSIGN_ESCALAR(meps->m01, -ESCALAR_RE(delta.m01) / delta.m00,
                   -ESCALAR_IM(delta.m01) / delta.m00);
    ASSIGN_ESCALAR(meps->m02, -ESCALAR_RE(delta.m02) / delta.m00,
                   -ESCALAR_IM(delta.m02) / delta.m00);
    ASSIGN_ESCALAR(meps->m12,
                   ESCALAR_RE(delta.m12) - ESCALAR_MULT_CONJ_RE(delta.m02, delta.m01) / delta.m00,
                   ESCALAR_IM(delta.m12) - ESCALAR_MULT_CONJ_IM(delta.m02, delta.m01) / delta.m00);

#define SWAP(a, b)                                                                                 \
  {                                                                                                \
    double xxx = a;                                                                                \
    a = b;                                                                                         \
    b = xxx;                                                                                       \
  }
    /* invert rotation matrix = transpose */
    SWAP(Rot[0][1], Rot[1][0]);
    SWAP(Rot[0][2], Rot[2][0]);
    SWAP(Rot[2][1], Rot[1][2]);
    maxwell_sym_matrix_rotate(meps, meps, Rot); /* rotate back */
#undef SWAP

#ifdef DEBUG
    CHECK(negative_epsilon_ok || maxwell_sym_matrix_positive_definite(meps),
          "negative mean epsilon from Kottke algorithm");
#endif
  }
  return 1;
}

void mode_solver::material_epsmu(meep_geom::material_type material, symmetric_matrix *epsmu,
                                 symmetric_matrix *epsmu_inv, bool eps) {

  meep_geom::material_data *md = material;

#ifndef WITH_HERMITIAN_EPSILON
  if (md->which_subclass == meep_geom::material_data::MATERIAL_USER ||
      md->which_subclass == meep_geom::material_data::MATERIAL_FILE) {
    md->medium.check_offdiag_im_zero_or_abort();
  }
#endif

  if (eps) {
    switch (md->which_subclass) {
      case meep_geom::material_data::MEDIUM:
      case meep_geom::material_data::MATERIAL_FILE:
      case meep_geom::material_data::MATERIAL_USER:
        epsmu->m00 = md->medium.epsilon_diag.x;
        epsmu->m11 = md->medium.epsilon_diag.y;
        epsmu->m22 = md->medium.epsilon_diag.z;
#ifdef WITH_HERMITIAN_EPSILON
        epsmu->m01.re = md->medium.epsilon_offdiag.x.re;
        epsmu->m01.im = md->medium.epsilon_offdiag.x.im;
        epsmu->m02.re = md->medium.epsilon_offdiag.y.re;
        epsmu->m02.im = md->medium.epsilon_offdiag.y.im;
        epsmu->m12.re = md->medium.epsilon_offdiag.z.re;
        epsmu->m12.im = md->medium.epsilon_offdiag.z.im;
#else
        epsmu->m01 = md->medium.epsilon_offdiag.x.re;
        epsmu->m02 = md->medium.epsilon_offdiag.y.re;
        epsmu->m12 = md->medium.epsilon_offdiag.z.re;
#endif
        maxwell_sym_matrix_invert(epsmu_inv, epsmu);
        break;
      case meep_geom::material_data::PERFECT_METAL:
        epsmu->m00 = -inf;
        epsmu->m11 = -inf;
        epsmu->m22 = -inf;
#ifdef WITH_HERMITIAN_EPSILON
        epsmu->m01.re = 0.0;
        epsmu->m01.im = 0.0;
        epsmu->m02.re = 0.0;
        epsmu->m02.im = 0.0;
        epsmu->m12.re = 0.0;
        epsmu->m12.im = 0.0;

        epsmu_inv->m01.re = 0.0;
        epsmu_inv->m01.im = 0.0;
        epsmu_inv->m02.re = 0.0;
        epsmu_inv->m02.im = 0.0;
        epsmu_inv->m12.re = 0.0;
        epsmu_inv->m12.im = 0.0;
#else
        epsmu->m01 = 0.0;
        epsmu->m02 = 0.0;
        epsmu->m12 = 0.0;
        epsmu_inv->m01 = 0.0;
        epsmu_inv->m02 = 0.0;
        epsmu_inv->m12 = 0.0;
#endif
        epsmu_inv->m00 = -0.0;
        epsmu_inv->m11 = -0.0;
        epsmu_inv->m22 = -0.0;
        break;
      default: meep::abort("Unknown material type");
    }
  }
  else {
    switch (md->which_subclass) {
      case meep_geom::material_data::MEDIUM:
      case meep_geom::material_data::MATERIAL_FILE:
      case meep_geom::material_data::MATERIAL_USER:
        epsmu->m00 = md->medium.mu_diag.x;
        epsmu->m11 = md->medium.mu_diag.y;
        epsmu->m22 = md->medium.mu_diag.z;
#ifdef WITH_HERMITIAN_EPSILON
        epsmu->m01.re = md->medium.mu_offdiag.x.re;
        epsmu->m01.im = md->medium.mu_offdiag.x.im;
        epsmu->m02.re = md->medium.mu_offdiag.y.re;
        epsmu->m02.im = md->medium.mu_offdiag.y.im;
        epsmu->m12.re = md->medium.mu_offdiag.z.re;
        epsmu->m12.im = md->medium.mu_offdiag.z.im;
#else
        epsmu->m01 = md->medium.mu_offdiag.x.re;
        epsmu->m02 = md->medium.mu_offdiag.y.re;
        epsmu->m12 = md->medium.mu_offdiag.z.re;
#endif
        maxwell_sym_matrix_invert(epsmu_inv, epsmu);
        break;
      case meep_geom::material_data::PERFECT_METAL:
        epsmu->m00 = 1.0;
        epsmu->m11 = 1.0;
        epsmu->m22 = 1.0;
        epsmu_inv->m00 = 1.0;
        epsmu_inv->m11 = 1.0;
        epsmu_inv->m22 = 1.0;
#ifdef WITH_HERMITIAN_EPSILON
        epsmu->m01.re = 0.0;
        epsmu->m01.im = 0.0;
        epsmu->m02.re = 0.0;
        epsmu->m02.im = 0.0;
        epsmu->m12.re = 0.0;
        epsmu->m12.im = 0.0;

        epsmu_inv->m01.re = 0.0;
        epsmu_inv->m01.im = 0.0;
        epsmu_inv->m02.re = 0.0;
        epsmu_inv->m02.im = 0.0;
        epsmu_inv->m12.re = 0.0;
        epsmu_inv->m12.im = 0.0;
#else
        epsmu->m01 = 0.0;
        epsmu->m02 = 0.0;
        epsmu->m12 = 0.0;
        epsmu_inv->m01 = 0.0;
        epsmu_inv->m02 = 0.0;
        epsmu_inv->m12 = 0.0;
#endif
        break;
      default: meep::abort("unknown material type");
    }
  }
}

void mode_solver::get_material_pt(meep_geom::material_type &material, vector3 p) {
  boolean inobject;
  material = (meep_geom::material_type)material_of_unshifted_point_in_tree_inobject(
      p, geometry_tree, &inobject);
  meep_geom::material_data *md = material;

  switch (md->which_subclass) {
    // material read from file: interpolate to get properties at r
    case meep_geom::material_data::MATERIAL_FILE:
      if (md->epsilon_data) { meep_geom::epsilon_file_material(md, p); }
      else {
        material = (meep_geom::material_type)default_material;
      }
      return;

    // material specified by user-supplied function: call user
    // function to get properties at r.
    // Note that we initialize the medium to vacuum, so that
    // the user's function only needs to fill in whatever is
    // different from vacuum.
    case meep_geom::material_data::MATERIAL_USER:
      md->medium = meep_geom::medium_struct();
      md->user_func(p, md->user_data, &(md->medium));
      return;

    // position-independent material or metal: there is nothing to do
    case meep_geom::material_data::MEDIUM:
    case meep_geom::material_data::PERFECT_METAL: return;
    default: meep::abort("unknown material type");
  }
}

bool mode_solver::using_mu() { return mdata && mdata->mu_inv != NULL; }

void mode_solver::init(int p, bool reset_fields, geometric_object_list *geometry,
                       meep_geom::material_data *_default_material) {
  int have_old_fields = 0;

  default_material = _default_material;

  n[0] = grid_size.x;
  n[1] = grid_size.y;
  n[2] = grid_size.z;

  if (target_freq != 0.0 && mpb_verbosity >= 1) { meep::master_printf("Target frequency is %g\n", target_freq); }

  int true_rank = n[2] > 1 ? 3 : (n[1] > 1 ? 2 : 1);
  if (true_rank < dimensions) { dimensions = true_rank; }
  else if (true_rank > dimensions) {
    if (mpb_verbosity >= 1)
      meep::master_printf("WARNING: rank of grid is > dimensions.\n"
                        "         setting extra grid dims. to 1.\n");
    // force extra dims to be 1
    if (dimensions <= 2) { n[2] = 1; }
    if (dimensions <= 1) { n[1] = 1; }
  }

  if (mpb_verbosity >= 1) {
    meep::master_printf("Working in %d dimensions.\n", dimensions);
    meep::master_printf("Grid size is %d x %d x %d.\n", n[0], n[1], n[2]);
  }

  int block_size;

  if (eigensolver_block_size != 0 && eigensolver_block_size < num_bands) {
    block_size = eigensolver_block_size;
    if (block_size < 0) {
      // Guess a block_size near -block_size, chosen so that all blocks are nearly equal in size
      block_size = (num_bands - block_size - 1) / (-block_size);
      block_size = (num_bands + block_size - 1) / block_size;
    }
    if (mpb_verbosity >= 1)
      meep::master_printf("Solving for %d bands at a time.\n", block_size);
  }
  else {
    block_size = num_bands;
  }

  if (mdata) {
    if (n[0] == mdata->nx && n[1] == mdata->ny && n[2] == mdata->nz &&
        block_size == Hblock.alloc_p && num_bands == H.p &&
        eigensolver_nwork + (mdata->mu_inv != NULL) == nwork_alloc) {

      have_old_fields = 1;
    }
    else {
      destroy_evectmatrix(H);
      for (int i = 0; i < nwork_alloc; ++i) {
        destroy_evectmatrix(W[i]);
      }

      if (Hblock.data != H.data) { destroy_evectmatrix(Hblock); }
      if (muinvH.data != H.data) { destroy_evectmatrix(muinvH); }
    }
    destroy_maxwell_target_data(mtdata);
    mtdata = NULL;
    destroy_maxwell_data(mdata);
    mdata = NULL;
    curfield_reset();
  }
  else {
    srand(time(NULL));
  }

  if (deterministic) {
    // seed should be the same for each run, although
    // it should be different for each process.
    // TODO: MPI
    // int rank = meep::my_rank();
    srand(314159); // * (rank + 1));
  }

  if (mpb_verbosity >= 1)
    meep::master_printf("Creating Maxwell data...\n");
  mdata = create_maxwell_data(n[0], n[1], n[2], &local_N, &N_start, &alloc_N, block_size,
                              NUM_FFT_BANDS);

  if (target_freq != 0.0) { mtdata = create_maxwell_target_data(mdata, target_freq); }

  init_epsilon(geometry);

  if (check_maxwell_dielectric(mdata, 0)) { meep::abort("invalid dielectric function for MPB"); }

  if (!have_old_fields) {
    if (mpb_verbosity >= 1)
      meep::master_printf("Allocating fields...\n");

    int N = n[0] * n[1] * n[2];
    int c = 2;

    H = create_evectmatrix(N, c, num_bands, local_N, N_start, alloc_N);
    nwork_alloc = eigensolver_nwork + (mdata->mu_inv != NULL);

    for (int i = 0; i < nwork_alloc; ++i) {
      W[i] = create_evectmatrix(N, c, block_size, local_N, N_start, alloc_N);
    }

    if (block_size < num_bands) {
      Hblock = create_evectmatrix(N, c, block_size, local_N, N_start, alloc_N);
    }
    else {
      Hblock = H;
    }

    if (using_mu() && block_size < num_bands) {
      muinvH = create_evectmatrix(N, c, num_bands, local_N, N_start, alloc_N);
    }
    else {
      muinvH = H;
    }
  }

  set_parity(p);

  if (!have_old_fields || reset_fields) { randomize_fields(); }

  evectmatrix_flops = eigensolver_flops;
}

void mode_solver::init_epsilon(geometric_object_list *geometry_in) {
  // Persist geometry data and move it out of the input argument.
  clear_geometry_list();
  if (geometry_in->num_items && geometry_in->items) {
    geometry_list.items = geometry_in->items;
    geometry_list.num_items = geometry_in->num_items;
    geometry_in->items = NULL;
    geometry_in->num_items = 0;
  }
  mpb_real no_size_x = geometry_lattice.size.x == 0 ? 1 : geometry_lattice.size.x;
  mpb_real no_size_y = geometry_lattice.size.y == 0 ? 1 : geometry_lattice.size.y;
  mpb_real no_size_z = geometry_lattice.size.z == 0 ? 1 : geometry_lattice.size.z;

  if (mpb_verbosity >= 1)
    meep::master_printf("Mesh size is %d.\n", mesh_size);

  Rm.c0 = vector3_scale(no_size_x, geometry_lattice.basis.c0);
  Rm.c1 = vector3_scale(no_size_y, geometry_lattice.basis.c1);
  Rm.c2 = vector3_scale(no_size_z, geometry_lattice.basis.c2);

  if (mpb_verbosity >= 1) {
    meep::master_printf("Lattice vectors:\n");
    meep::master_printf("     (%g, %g, %g)\n", Rm.c0.x, Rm.c0.y, Rm.c0.z);
    meep::master_printf("     (%g, %g, %g)\n", Rm.c1.x, Rm.c1.y, Rm.c1.z);
    meep::master_printf("     (%g, %g, %g)\n", Rm.c2.x, Rm.c2.y, Rm.c2.z);
  }

  vol = fabs(matrix3x3_determinant(Rm));
  if (mpb_verbosity >= 1)
    meep::master_printf("Cell volume = %g\n", vol);

  Gm = matrix3x3_inverse(matrix3x3_transpose(Rm));
  if (mpb_verbosity >= 1) {
    meep::master_printf("Reciprocal lattice vectors (/ 2 pi):\n");
    meep::master_printf("     (%g, %g, %g)\n", Gm.c0.x, Gm.c0.y, Gm.c0.z);
    meep::master_printf("     (%g, %g, %g)\n", Gm.c1.x, Gm.c1.y, Gm.c1.z);
    meep::master_printf("     (%g, %g, %g)\n", Gm.c2.x, Gm.c2.y, Gm.c2.z);
  }

  matrix3x3_to_arr(R, Rm);
  matrix3x3_to_arr(G, Gm);

  geom_fix_object_list(geometry_list);

  if (mpb_verbosity >= 1)
    meep::master_printf("Geometric objects:\n");
  if (meep::am_master()) {
    for (int i = 0; i < geometry_list.num_items; ++i) {

#ifndef WITH_HERMITIAN_EPSILON
      meep_geom::medium_struct *mm;
      if (meep_geom::is_medium(geometry_list.items[i].material, &mm)) { mm->check_offdiag_im_zero_or_abort(); }
#endif

      display_geometric_object_info(5, geometry_list.items[i]);

      // meep_geom::medium_struct *mm;
      // if (meep_geom::is_medium(geometry.items[i].material, &mm)) {
      //   meep::master_printf("%*sdielectric constant epsilon diagonal = (%g,%g,%g)\n", 5 + 5, "",
      //          mm->epsilon_diag.x, mm->epsilon_diag.y, mm->epsilon_diag.z);
      // }
    }
  }

  {
    // Replace 0 with 1e-20 for no size
    vector3 tmp_size;
    tmp_size.x = geometry_lattice.size.x == 0 ? 1e-20 : geometry_lattice.size.x;
    tmp_size.y = geometry_lattice.size.y == 0 ? 1e-20 : geometry_lattice.size.y;
    tmp_size.z = geometry_lattice.size.z == 0 ? 1e-20 : geometry_lattice.size.z;

    geom_box b0;
    b0.low = vector3_plus(geometry_center, vector3_scale(-0.5, tmp_size));
    b0.high = vector3_plus(geometry_center, vector3_scale(0.5, tmp_size));
    /* pad tree boundaries to allow for sub-pixel averaging */
    b0.low.x -= tmp_size.x / mdata->nx;
    b0.low.y -= tmp_size.y / mdata->ny;
    b0.low.z -= tmp_size.z / mdata->nz;
    b0.high.x += tmp_size.x / mdata->nx;
    b0.high.y += tmp_size.y / mdata->ny;
    b0.high.z += tmp_size.z / mdata->nz;
    geometry_tree = create_geom_box_tree0(geometry_list, b0);
  }

  if (verbose && meep::am_master()) {
    if (mpb_verbosity >= 1)
      meep::master_printf("Geometry object bounding box tree:\n");
    display_geom_box_tree(5, geometry_tree);
  }

  int tree_depth;
  int tree_nobjects;
  geom_box_tree_stats(geometry_tree, &tree_depth, &tree_nobjects);
  if (mpb_verbosity >= 1)
    meep::master_printf("Geometric object tree has depth %d and %d object nodes"
                        " (vs. %d actual objects)\n",
                        tree_depth, tree_nobjects, geometry_list.num_items);

  // restricted_tree = geometry_tree;

  reset_epsilon(&geometry_list);
}

void mode_solver::reset_epsilon(geometric_object_list *geometry) {
  int mesh[3] = {
      mesh_size,
      (dimensions > 1) ? mesh_size : 1,
      (dimensions > 2) ? mesh_size : 1,
  };

  if (!epsilon_input_file.empty()) {
    default_material = meep_geom::make_file_material(epsilon_input_file.c_str());
  }

  // TODO: support mu_input_file
  // if (!mu_input_file.empty()) {
  // }

  if (mpb_verbosity >= 1)
    meep::master_printf("Initializing epsilon function...\n");
  set_maxwell_dielectric(mdata, mesh, R, G, dielectric_function, mean_epsilon_func,
                         static_cast<void *>(this));

  if (has_mu(geometry)) {
    if (mpb_verbosity >= 1)
      meep::master_printf("Initializing mu function...\n");
    eps = false;
    set_maxwell_mu(mdata, mesh, R, G, dielectric_function, mean_epsilon_func,
                   static_cast<void *>(this));
    eps = true;
  }
}

bool mode_solver::has_mu(geometric_object_list *geometry) {
  // TODO: mu_file_func
  if (material_has_mu(default_material) || force_mu) { return true; }

  for (int i = 0; i < geometry->num_items; ++i) {
    if (material_has_mu(geometry->items[i].material)) { return true; }
  }
  return false;
}

bool mode_solver::material_has_mu(void *mt) {
  meep_geom::material_type mat = (meep_geom::material_type)mt;
  meep_geom::medium_struct *m = &mat->medium;

  if (mat->which_subclass != meep_geom::material_data::PERFECT_METAL) {
    bool has_nonzero_mu_offdiag = false;

#ifdef WITH_HERMITIAN_EPSILON
    if (m->mu_offdiag.x.re != 0 || m->mu_offdiag.x.im != 0 || m->mu_offdiag.y.re != 0 ||
        m->mu_offdiag.y.im != 0 || m->mu_offdiag.z.re != 0 || m->mu_offdiag.z.im != 0) {
      has_nonzero_mu_offdiag = true;
    }
#else
    if (m->mu_offdiag.x.re != 0 || m->mu_offdiag.y.re != 0 || m->mu_offdiag.z.re != 0) {
      has_nonzero_mu_offdiag = true;
    }
#endif

    if (m->mu_diag.x != 1 || m->mu_diag.y != 1 || m->mu_diag.z != 1 || has_nonzero_mu_offdiag) {
      return true;
    }
  }
  return false;
}

void mode_solver::set_parity(integer p) {
  if (!mdata) {
    meep::master_fprintf(stderr, "init must be called before set-parity!\n");
    return;
  }

  if (p == -1) { p = last_parity < 0 ? NO_PARITY : last_parity; }

  set_maxwell_data_parity(mdata, p);
  if (mdata->parity != p) {
    meep::master_fprintf(stderr, "k vector incompatible with parity\n");
    exit(EXIT_FAILURE);
  }
  if (mpb_verbosity >= 1)
    meep::master_printf("Solving for band polarization: %s.\n", parity_string(mdata));

  last_parity = p;
  set_kpoint_index(0); /* reset index */
}

void mode_solver::set_num_bands(int nb) {
  num_bands = nb;
  freqs.resize(nb);
}

int mode_solver::get_kpoint_index() { return kpoint_index; }

void mode_solver::set_kpoint_index(int i) { kpoint_index = i; }

void mode_solver::randomize_fields() {

  if (!mdata) { return; }
  if (mpb_verbosity >= 1)
    meep::master_printf("Initializing fields to random numbers...\n");

  for (int i = 0; i < H.n * H.p; ++i) {
    ASSIGN_SCALAR(H.data[i], rand() * 1.0 / RAND_MAX, rand() * 1.0 / RAND_MAX);
  }
}

void mode_solver::solve_kpoint(vector3 kvector) {
  // if we get too close to singular k==0 point, just set k=0 exploit our
  // special handling of this k
  if (vector3_norm(kvector) < 1e-10) { kvector.x = kvector.y = kvector.z = 0; }

  if (mpb_verbosity >= 1)
    meep::master_printf("solve_kpoint (%g,%g,%g):\n", kvector.x, kvector.y, kvector.z);

  curfield_reset();

  if (num_bands == 0) {
    if (mpb_verbosity >= 1)
      meep::master_printf("  num-bands is zero, not solving for any bands\n");
    return;
  }

  if (!mdata) {
    meep::master_fprintf(stderr, "init must be called before solve_kpoint!\n");
    return;
  }

  // If this is the first k point, print out a header line for the frequency
  // grep data.
  if (mpb_verbosity >= 1) {
    if (!kpoint_index && meep::am_master()) {
      meep::master_printf("%sfreqs:, k index, k1, k2, k3, kmag/2pi", parity_string(mdata));

      for (int i = 0; i < num_bands; ++i) {
        meep::master_printf(", %s%sband %d", parity_string(mdata),
                            mdata->parity == NO_PARITY ? "" : " ", i + 1);
      }
      meep::master_printf("\n");
    }
  }

  cur_kvector = kvector;
  mpb_real k[3];
  vector3_to_arr(k, kvector);

  update_maxwell_data_k(mdata, k, G[0], G[1], G[2]);

  std::vector<mpb_real> eigvals(num_bands);

  // TODO: Get flags from python
  int flags = EIGS_DEFAULT_FLAGS;
  if (verbose || mpb_verbosity >= 1) { flags |= EIGS_VERBOSE; }

  // Constant (zero frequency) bands at k=0 are handled specially, so remove
  // them from the solutions for the eigensolver.
  int ib0;
  if (mdata->zero_k && !mtdata) {
    ib0 = maxwell_zero_k_num_const_bands(H, mdata);
    for (int in = 0; in < H.n; ++in) {
      for (int ip = 0; ip < H.p - ib0; ++ip) {
        H.data[in * H.p + ip] = H.data[in * H.p + ip + ib0];
      }
    }
    evectmatrix_resize(&H, H.p - ib0, 1);
  }
  else {
    ib0 = 0; /* solve for all bands */
  }

  // Set up deflation data.
  deflation_data deflation;
  if (muinvH.data != Hblock.data) {
    deflation.Y = H;
    deflation.BY = muinvH.data != H.data ? muinvH : H;
    deflation.p = 0;
    deflation.S = (scalar *)malloc(sizeof(scalar) * H.p * Hblock.p);
    deflation.S2 = (scalar *)malloc(sizeof(scalar) * H.p * Hblock.p);
  }

  int total_iters = 0;

  for (int ib = ib0; ib < num_bands; ib += Hblock.alloc_p) {
    evectconstraint_chain *constraints;
    int num_iters;

    // Don't solve for too many bands if the block size doesn't divide the number
    // of bands.
    if (ib + mdata->num_bands > num_bands) {
      maxwell_set_num_bands(mdata, num_bands - ib);

      for (int i = 0; i < nwork_alloc; ++i) {
        evectmatrix_resize(&W[i], num_bands - ib, 0);
      }
      evectmatrix_resize(&Hblock, num_bands - ib, 0);
    }

    if (mpb_verbosity >= 1)
      meep::master_printf("Solving for bands %d to %d...\n", ib + 1, ib + Hblock.p);

    constraints = NULL;
    constraints = evect_add_constraint(constraints, maxwell_parity_constraint, (void *)mdata);

    if (mdata->zero_k) {
      constraints = evect_add_constraint(constraints, maxwell_zero_k_constraint, (void *)mdata);
    }

    if (Hblock.data != H.data) { /* initialize fields of block from H */
      for (int in = 0; in < Hblock.n; ++in) {
        for (int ip = 0; ip < Hblock.p; ++ip) {
          Hblock.data[in * Hblock.p + ip] = H.data[in * H.p + ip + (ib - ib0)];
        }
      }

      deflation.p = ib - ib0;
      if (deflation.p > 0) {
        if (deflation.BY.data != H.data) {
          evectmatrix_resize(&deflation.BY, deflation.p, 0);
          maxwell_muinv_operator(H, deflation.BY, (void *)mdata, 1, deflation.BY);
        }
        constraints = evect_add_constraint(constraints, deflation_constraint, &deflation);
      }
    }

    if (mtdata) { /* solving for bands near a target frequency */
      CHECK(mdata->mu_inv == NULL, "targeted solver doesn't handle mu");

      eigensolver(Hblock, eigvals.data() + ib, maxwell_target_operator, (void *)mtdata, NULL, NULL,
                  use_simple_preconditioner ? maxwell_target_preconditioner
                                            : maxwell_target_preconditioner2,
                  (void *)mtdata, evectconstraint_chain_func, (void *)constraints, W, nwork_alloc,
                  tolerance, &num_iters, flags);

      // now, diagonalize the real Maxwell operator in the solution subspace to
      // get the true eigenvalues and eigenvectors
      CHECK(nwork_alloc >= 2, "not enough workspace");
      eigensolver_get_eigenvals(Hblock, eigvals.data() + ib, maxwell_operator, mdata, W[0], W[1]);
    }
    else {
      eigensolver(Hblock, eigvals.data() + ib, maxwell_operator, (void *)mdata,
                  mdata->mu_inv ? maxwell_muinv_operator : NULL, (void *)mdata,
                  use_simple_preconditioner ? maxwell_preconditioner : maxwell_preconditioner2,
                  (void *)mdata, evectconstraint_chain_func, (void *)constraints, W, nwork_alloc,
                  tolerance, &num_iters, flags);
    }

    if (Hblock.data != H.data) { /* save solutions of current block */
      for (int in = 0; in < Hblock.n; ++in) {
        for (int ip = 0; ip < Hblock.p; ++ip) {
          H.data[in * H.p + ip + (ib - ib0)] = Hblock.data[in * Hblock.p + ip];
        }
      }
    }

    evect_destroy_constraints(constraints);

    if (mpb_verbosity >= 1)
      meep::master_printf("Finished solving for bands %d to %d after %d iterations.\n", ib + 1,
                          ib + Hblock.p, num_iters);

    total_iters += num_iters * Hblock.p;
  }

  if (num_bands - ib0 > Hblock.alloc_p && mpb_verbosity >= 1) {
    meep::master_printf("Finished k-point with %g mean iterations/band.\n",
                        total_iters * 1.0 / num_bands);
  }

  // Manually put in constant (zero-frequency) solutions for k=0.
  if (mdata->zero_k && !mtdata) {
    evectmatrix_resize(&H, H.alloc_p, 1);
    for (int in = 0; in < H.n; ++in) {
      for (int ip = H.p - ib0 - 1; ip >= 0; --ip) {
        H.data[in * H.p + ip + ib0] = H.data[in * H.p + ip];
      }
    }
    maxwell_zero_k_set_const_bands(H, mdata);
    for (int ib = 0; ib < ib0; ++ib) {
      eigvals[ib] = 0;
    }
  }

  /* Reset scratch matrix sizes: */
  evectmatrix_resize(&Hblock, Hblock.alloc_p, 0);
  for (int i = 0; i < nwork_alloc; ++i) {
    evectmatrix_resize(&W[i], W[i].alloc_p, 0);
  }

  maxwell_set_num_bands(mdata, Hblock.alloc_p);

  /* Destroy deflation data: */
  if (H.data != Hblock.data) {
    free(deflation.S2);
    free(deflation.S);
  }

  iterations = total_iters; /* iterations output variable */

  set_kpoint_index(kpoint_index + 1);

  if (mpb_verbosity >= 1)
    meep::master_printf("%sfreqs:, %d, %g, %g, %g, %g", parity_string(mdata), kpoint_index,
                        (double)k[0], (double)k[1], (double)k[2],
                        vector3_norm(matrix3x3_vector3_mult(Gm, kvector)));

  for (int i = 0; i < num_bands; ++i) {
    freqs[i] = negative_epsilon_ok ? eigvals[i] : sqrt(eigvals[i]);
    if (mpb_verbosity >= 1)
      meep::master_printf(", %g", freqs[i]);
  }
  if (mpb_verbosity >= 1)
    meep::master_printf("\n");

  eigensolver_flops = evectmatrix_flops;
}

/* get the epsilon function, and compute some statistics */
void mode_solver::get_epsilon() {
  mpb_real eps_mean = 0;
  mpb_real eps_inv_mean = 0;
  mpb_real eps_high = -1e20;
  mpb_real eps_low = 1e20;

  int fill_count = 0;

  if (!mdata) {
    meep::master_fprintf(stderr, "init-params must be called before get-epsilon!\n");
    return;
  }

  curfield = (scalar_complex *)mdata->fft_data;
  mpb_real *epsilon = (mpb_real *)curfield;
  curfield_band = 0;
  curfield_type = epsilon_CURFIELD_TYPE;

  /* get epsilon.  Recall that we actually have an inverse
     dielectric tensor at each point; define an average index by
     the inverse of the average eigenvalue of the 1/eps tensor.
     i.e. 3/(trace 1/eps). */

  int N = mdata->fft_output_size;

  for (int i = 0; i < N; ++i) {
    if (mdata->eps_inv == NULL) { epsilon[i] = 1.0; }
    else {
      epsilon[i] = mean_medium_from_matrix(mdata->eps_inv + i);
    }
    if (epsilon[i] < eps_low) { eps_low = epsilon[i]; }
    if (epsilon[i] > eps_high) { eps_high = epsilon[i]; }
    eps_mean += epsilon[i];
    eps_inv_mean += 1 / epsilon[i];
    if (epsilon[i] > 1.0001) { ++fill_count; }
  }

  mpi_allreduce_1(&eps_mean, mpb_real, SCALAR_MPI_TYPE, MPI_SUM, mpb_comm);
  mpi_allreduce_1(&eps_inv_mean, mpb_real, SCALAR_MPI_TYPE, MPI_SUM, mpb_comm);
  mpi_allreduce_1(&eps_low, mpb_real, SCALAR_MPI_TYPE, MPI_MIN, mpb_comm);
  mpi_allreduce_1(&eps_high, mpb_real, SCALAR_MPI_TYPE, MPI_MAX, mpb_comm);
  mpi_allreduce_1(&fill_count, int, MPI_INT, MPI_SUM, mpb_comm);

  N = mdata->nx * mdata->ny * mdata->nz;
  eps_mean /= N;
  eps_inv_mean = N / eps_inv_mean;

  if (mpb_verbosity >= 1)
    meep::master_printf("epsilon: %g-%g, mean %g, harm. mean %g, %g%% > 1, %g%% \"fill\"\n", eps_low,
                        eps_high, eps_mean, eps_inv_mean, (100.0 * fill_count) / N,
                        eps_high == eps_low ? 100.0
                                            : 100.0 * (eps_mean - eps_low) / (eps_high - eps_low));
  }

/* get the mu function, and compute some statistics */
void mode_solver::get_mu() {
  mpb_real eps_mean = 0;
  mpb_real mu_inv_mean = 0;
  mpb_real eps_high = -1e20;
  mpb_real eps_low = 1e20;
  int fill_count = 0;

  if (!mdata) {
    meep::master_fprintf(stderr, "mode_solver.init must be called before get-mu!\n");
    return;
  }

  curfield = (scalar_complex *)mdata->fft_data;
  mpb_real *mu = (mpb_real *)curfield;
  curfield_band = 0;
  curfield_type = mu_CURFIELD_TYPE;

  /* get mu.  Recall that we actually have an inverse
     dielectric tensor at each point; define an average index by
     the inverse of the average eigenvalue of the 1/eps tensor.
     i.e. 3/(trace 1/eps). */

  int N = mdata->fft_output_size;

  for (int i = 0; i < N; ++i) {
    if (mdata->mu_inv == NULL) { mu[i] = 1.0; }
    else {
      mu[i] = mean_medium_from_matrix(mdata->mu_inv + i);
    }

    if (mu[i] < eps_low) { eps_low = mu[i]; }
    if (mu[i] > eps_high) { eps_high = mu[i]; }

    eps_mean += mu[i];
    mu_inv_mean += 1 / mu[i];
    if (mu[i] > 1.0001) { ++fill_count; }
  }

  mpi_allreduce_1(&eps_mean, mpb_real, SCALAR_MPI_TYPE, MPI_SUM, mpb_comm);
  mpi_allreduce_1(&mu_inv_mean, mpb_real, SCALAR_MPI_TYPE, MPI_SUM, mpb_comm);
  mpi_allreduce_1(&eps_low, mpb_real, SCALAR_MPI_TYPE, MPI_MIN, mpb_comm);
  mpi_allreduce_1(&eps_high, mpb_real, SCALAR_MPI_TYPE, MPI_MAX, mpb_comm);
  mpi_allreduce_1(&fill_count, int, MPI_INT, MPI_SUM, mpb_comm);

  N = mdata->nx * mdata->ny * mdata->nz;
  eps_mean /= N;
  mu_inv_mean = N / mu_inv_mean;

  if (mpb_verbosity >= 1)
    meep::master_printf("mu: %g-%g, mean %g, harm. mean %g, %g%% > 1, %g%% \"fill\"\n", eps_low,
                        eps_high, eps_mean, mu_inv_mean, (100.0 * fill_count) / N,
                        eps_high == eps_low ? 100.0
                                            : 100.0 * (eps_mean - eps_low) / (eps_high - eps_low));
}

void mode_solver::curfield_reset() {
  curfield = NULL;
  curfield_type = '-';
}

/* get the specified component of the dielectric tensor,
   or the inverse tensor if inv != 0 */
void mode_solver::get_epsilon_tensor(int c1, int c2, int imag, int inv) {
  int conj = 0, offset = 0;

  curfield_type = '-'; /* only used internally, for now */
  mpb_real *epsilon = (mpb_real *)mdata->fft_data;
  int N = mdata->fft_output_size;

  switch (c1 * 3 + c2) {
    case 0: offset = offsetof(symmetric_matrix, m00); break;
    case 1: offset = offsetof(symmetric_matrix, m01); break;
    case 2: offset = offsetof(symmetric_matrix, m02); break;
    case 3:
      offset = offsetof(symmetric_matrix, m01); /* = conj(m10) */
      conj = imag;
      break;
    case 4: offset = offsetof(symmetric_matrix, m11); break;
    case 5: offset = offsetof(symmetric_matrix, m12); break;
    case 6:
      offset = offsetof(symmetric_matrix, m02); /* = conj(m20) */
      conj = imag;
      break;
    case 7:
      offset = offsetof(symmetric_matrix, m12); /* = conj(m21) */
      conj = imag;
      break;
    case 8: offset = offsetof(symmetric_matrix, m22); break;
  }

#ifdef WITH_HERMITIAN_EPSILON
  if (c1 != c2 && imag) offset += offsetof(scalar_complex, im);
#endif

  for (int i = 0; i < N; ++i) {
    if (inv) { epsilon[i] = *((mpb_real *)(((char *)&mdata->eps_inv[i]) + offset)); }
    else {
      symmetric_matrix eps;
      maxwell_sym_matrix_invert(&eps, &mdata->eps_inv[i]);
      epsilon[i] = *((mpb_real *)(((char *)&eps) + offset));
    }
    if (conj) epsilon[i] = -epsilon[i];
  }
}

std::vector<mpb_real> mode_solver::get_freqs() { return freqs; }

size_t mode_solver::get_field_size() { return mdata ? mdata->fft_output_size * 3 : 0; }

void mode_solver::get_efield(int band) {

  get_dfield(band);
  get_efield_from_dfield();
}

void mode_solver::get_efield_from_dfield() {
  if (!curfield || curfield_type != 'd') {
    meep::master_fprintf(stderr, "get_dfield must be called before get-efield-from-dfield!\n");
    return;
  }

  maxwell_compute_e_from_d(mdata, curfield, 1);
  curfield_type = 'e';
}

void mode_solver::get_dfield(int band) {
  if (!kpoint_index) {
    meep::master_fprintf(stderr, "solve_kpoint must be called before get_dfield\n");
    return;
  }

  if (band < 1 || band > H.p) {
    meep::master_fprintf(stderr, "Must have 1 <= band index <= num_bands (%d)\n", H.p);
    return;
  }

  curfield = (scalar_complex *)mdata->fft_data;
  curfield_band = band;
  curfield_type = 'd';

  if (mdata->mu_inv == NULL) { maxwell_compute_d_from_H(mdata, H, curfield, band - 1, 1); }
  else {
    evectmatrix_resize(&W[0], 1, 0);
    maxwell_compute_H_from_B(mdata, H, W[0], curfield, band - 1, 0, 1);
    maxwell_compute_d_from_H(mdata, W[0], curfield, 0, 1);
    evectmatrix_resize(&W[0], W[0].alloc_p, 0);
  }

  // Here, we correct for the fact that compute_d_from_H actually computes just
  // (k+G) x H, whereas the actual D field is i/omega i(k+G) x H...so, there is
  // an added factor of -1/omega.

  // We also divide by the cell volume so that the integral of H*B or of D*E is
  // unity.  (From the eigensolver + FFT, they are initially normalized to sum to
  // nx*ny*nz.)

  double scale;
  int N = mdata->fft_output_size;

  if (freqs[band - 1] != 0.0) { scale = -1.0 / freqs[band - 1]; }
  else
    scale = -1.0; /* arbitrary */

  scale /= sqrt(vol);

  for (int i = 0; i < N * 3; ++i) {
    curfield[i].re *= scale;
    curfield[i].im *= scale;
  }
}

void mode_solver::get_hfield(int band) {
  if (!kpoint_index) {
    meep::master_fprintf(stderr, "solve_kpoint must be called before get_dfield\n");
    return;
  }

  if (band < 1 || band > H.p) {
    meep::master_fprintf(stderr, "Must have 1 <= band index <= num_bands (%d)\n", H.p);
    return;
  }

  curfield = (scalar_complex *)mdata->fft_data;
  curfield_band = band;
  curfield_type = 'h';

  if (mdata->mu_inv == NULL)
    maxwell_compute_h_from_H(mdata, H, curfield, band - 1, 1);
  else {
    evectmatrix_resize(&W[0], 1, 0);
    maxwell_compute_H_from_B(mdata, H, W[0], curfield, band - 1, 0, 1);
    maxwell_compute_h_from_H(mdata, W[0], curfield, 0, 1);
    evectmatrix_resize(&W[0], W[0].alloc_p, 0);
  }

  // Divide by the cell volume so that the integral of H*B or of D*E is unity.
  // (From the eigensolver + FFT, they are initially normalized to sum to
  // nx*ny*nz.)

  double scale;
  scale = 1.0 / sqrt(vol);

  int N = mdata->fft_output_size;
  for (int i = 0; i < N * 3; ++i) {
    curfield[i].re *= scale;
    curfield[i].im *= scale;
  }
}

void mode_solver::get_bfield(int band) {
  if (!kpoint_index) {
    meep::master_fprintf(stderr, "solve_kpoint must be called before get_dfield\n");
    return;
  }

  if (band < 1 || band > H.p) {
    meep::master_fprintf(stderr, "Must have 1 <= band index <= num_bands (%d)\n", H.p);
    return;
  }

  curfield = (scalar_complex *)mdata->fft_data;
  curfield_band = band;
  curfield_type = 'b';
  maxwell_compute_h_from_H(mdata, H, curfield, band - 1, 1);

  // Divide by the cell volume so that the integral of H*B or of D*E is unity.
  // (From the eigensolver + FFT, they are initially normalized to sum to nx*ny*nz.) */

  double scale;
  scale = 1.0 / sqrt(vol);

  int N = mdata->fft_output_size;
  for (int i = 0; i < N * 3; ++i) {
    curfield[i].re *= scale;
    curfield[i].im *= scale;
  }
}

char mode_solver::get_curfield_type() { return curfield_type; }

void mode_solver::set_curfield_type(char t) { curfield_type = t; }

std::string mode_solver::get_parity_string() {
  std::string s(parity_string(mdata));
  return s;
}

std::vector<int> mode_solver::get_dims() {
  std::vector<int> dims;

  if (mdata->nx > 1) { dims.push_back(mdata->nx); }
  if (mdata->ny > 1) { dims.push_back(mdata->ny); }
  if (mdata->nz > 1) { dims.push_back(mdata->nz); }

  return dims;
}

void mode_solver::set_grid_size(vector3 gs) {
  grid_size.x = gs.x;
  grid_size.y = gs.y;
  grid_size.z = gs.z;
}

int mode_solver::get_libctl_dimensions() { return dimensions; }

void mode_solver::set_libctl_dimensions(int val) { dimensions = val; }

bool mode_solver::get_libctl_ensure_periodicity() { return ensure_periodicity; }

void mode_solver::set_libctl_ensure_periodicity(bool val) { ensure_periodicity = val; }

void mode_solver::set_libctl_geometry_lattice(lattice val) { geometry_lattice = val; }

void mode_solver::get_curfield(double *data, int size) {
  mpb_real *p = (mpb_real *)curfield;

  for (int i = 0; i < size; ++i) {
    data[i] = p[i];
  }
}

void mode_solver::get_curfield_cmplx(std::complex<mpb_real> *cdata, int size) {
  scalar_complex *p = (scalar_complex *)curfield;

  for (int i = 0; i < size; ++i) {
    cdata[i] = std::complex<mpb_real>(p[i].re, p[i].im);
  }
}

void mode_solver::set_curfield(double *data, int size) {
  mpb_real *p = (mpb_real *)curfield;

  for (int i = 0; i < size; ++i) {
    p[i] = data[i];
  }
}

void mode_solver::set_curfield_cmplx(std::complex<mpb_real> *cdata, int size) {
  scalar_complex *p = (scalar_complex *)curfield;

  for (int i = 0; i < size; ++i) {
    scalar_complex s = {cdata[i].real(), cdata[i].imag()};
    p[i] = s;
  }
}

// internal function for compute_field_energy, below
double mode_solver::compute_field_energy_internal(mpb_real comp_sum[6]) {
  mpb_real comp_sum2[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  mpb_real energy_sum = 0.0;
  mpb_real *energy_density = (mpb_real *)curfield;

  int N = mdata->fft_output_size;

  for (int i = 0; i < N; ++i) {
    scalar_complex field[3];
    mpb_real comp_sqr0, comp_sqr1, comp_sqr2, comp_sqr3, comp_sqr4, comp_sqr5;

    /* energy is either |curfield|^2 / mu or |curfield|^2 / epsilon,
       depending upon whether it is B or D. */
    if (curfield_type == 'd') {
      assign_symmatrix_vector(field, mdata->eps_inv[i], curfield + 3 * i);
    }
    else if (curfield_type == 'b' && mdata->mu_inv != NULL) {
      assign_symmatrix_vector(field, mdata->mu_inv[i], curfield + 3 * i);
    }
    else {
      field[0] = curfield[3 * i];
      field[1] = curfield[3 * i + 1];
      field[2] = curfield[3 * i + 2];
    }

    comp_sum2[0] += comp_sqr0 = field[0].re * curfield[3 * i].re;
    comp_sum2[1] += comp_sqr1 = field[0].im * curfield[3 * i].im;
    comp_sum2[2] += comp_sqr2 = field[1].re * curfield[3 * i + 1].re;
    comp_sum2[3] += comp_sqr3 = field[1].im * curfield[3 * i + 1].im;
    comp_sum2[4] += comp_sqr4 = field[2].re * curfield[3 * i + 2].re;
    comp_sum2[5] += comp_sqr5 = field[2].im * curfield[3 * i + 2].im;

    /* Note: here, we write to energy_density[i]; this is
       safe, even though energy_density is aliased to curfield,
       since energy_density[i] is guaranteed to come at or before
       curfield[i] (which we are now done with). */
    energy_sum += energy_density[i] =
        comp_sqr0 + comp_sqr1 + comp_sqr2 + comp_sqr3 + comp_sqr4 + comp_sqr5;
  }

  mpi_allreduce_1(&energy_sum, mpb_real, SCALAR_MPI_TYPE, MPI_SUM, mpb_comm);
  mpi_allreduce(comp_sum2, comp_sum, 6, mpb_real, SCALAR_MPI_TYPE, MPI_SUM, mpb_comm);

  // remember that we now have energy density; denoted by capital D/H
  curfield_type = toupper(curfield_type);

  return energy_sum;
}

void mode_solver::clear_geometry_list() {
  if (geometry_list.num_items && geometry_list.items) {
    for(int i = 0; i < geometry_list.num_items; ++i) {
      material_free((meep_geom::material_data *)geometry_list.items[i].material);
      delete (meep_geom::material_data *)geometry_list.items[i].material;
      geometric_object_destroy(geometry_list.items[i]);
    }
    delete[] geometry_list.items;
    geometry_list.items = NULL;
    geometry_list.num_items = 0;
  }
}

/* Replace curfield (either d or h) with the scalar energy density function,
   normalized to one.  While we're at it, compute some statistics about
   the relative strength of different field components.  Also return
   the integral of the energy density, which should be unity. */
std::vector<mpb_real> mode_solver::compute_field_energy() {
  std::vector<mpb_real> retval;

  if (!curfield || !strchr("dhb", curfield_type)) {
    meep::master_fprintf(stderr, "The D or H field must be loaded first.\n");
    return retval;
  }
  else if (curfield_type == 'h' && mdata->mu_inv != NULL) {
    meep::master_fprintf(stderr, "B, not H, must be loaded if we have mu.\n");
    return retval;
  }

  mpb_real comp_sum[6];
  mpb_real energy_sum = compute_field_energy_internal(comp_sum);

  if (mpb_verbosity >= 1)
    meep::master_printf("%c-energy-components:, %d, %d", curfield_type, kpoint_index, curfield_band);
  for (int i = 0; i < 6; ++i) {
    comp_sum[i] /= (energy_sum == 0 ? 1 : energy_sum);
    if (i % 2 == 1 && mpb_verbosity >= 1) {
      meep::master_printf(", %g", comp_sum[i] + comp_sum[i - 1]);
      }
  }
  if (mpb_verbosity >= 1)
    meep::master_printf("\n");

  /* The return value is a list of 7 items: the total energy,
     followed by the 6 elements of the comp_sum array (the fraction
     of the energy in the real/imag. parts of each field component). */

  retval.push_back(energy_sum * vol / H.N);

  for (int i = 0; i < 6; ++i) {
    retval.push_back(comp_sum[i]);
  }

  return retval;
}

std::vector<mpb_real> mode_solver::get_output_k() {
  std::vector<mpb_real> output_k;

  output_k.push_back(R[0][0] * mdata->current_k[0] + R[0][1] * mdata->current_k[1] +
                     R[0][2] * mdata->current_k[2]);

  output_k.push_back(R[1][0] * mdata->current_k[0] + R[1][1] * mdata->current_k[1] +
                     R[1][2] * mdata->current_k[2]);

  output_k.push_back(R[2][0] * mdata->current_k[0] + R[2][1] * mdata->current_k[1] +
                     R[2][2] * mdata->current_k[2]);
  return output_k;
}

mpb_real mode_solver::get_val(int ix, int iy, int iz, int nx, int ny, int nz, int last_dim_size,
                              mpb_real *data, int stride, int conjugate) {
  // #ifdef HAVE_MPI
  //      CHECK(0, "get-*-point not yet implemented for MPI!");
  // #else
  (void)nx;
  (void)last_dim_size;
  (void)conjugate;
  return data[(((ix * ny) + iy) * nz + iz) * stride];
  // #endif
}

mpb_real mode_solver::interp_val(vector3 p, int nx, int ny, int nz, int last_dim_size,
                                 mpb_real *data, int stride, int conjugate) {
  double ipart;
  mpb_real rx, ry, rz, dx, dy, dz;
  int x, y, z, x2, y2, z2;

  mpb_real latx = geometry_lattice.size.x == 0 ? 1e-20 : geometry_lattice.size.x;
  mpb_real laty = geometry_lattice.size.y == 0 ? 1e-20 : geometry_lattice.size.y;
  mpb_real latz = geometry_lattice.size.z == 0 ? 1e-20 : geometry_lattice.size.z;

  rx = modf(p.x / latx + 0.5, &ipart);
  if (rx < 0) rx += 1;
  ry = modf(p.y / laty + 0.5, &ipart);
  if (ry < 0) ry += 1;
  rz = modf(p.z / latz + 0.5, &ipart);
  if (rz < 0) rz += 1;

  /* get the point corresponding to r in the grid: */
  x = rx * nx;
  y = ry * ny;
  z = rz * nz;

  /* get the difference between (x,y,z) and the actual point */
  dx = rx * nx - x;
  dy = ry * ny - y;
  dz = rz * nz - z;

  /* get the other closest point in the grid, with periodic boundaries: */
  x2 = (nx + (dx >= 0.0 ? x + 1 : x - 1)) % nx;
  y2 = (ny + (dy >= 0.0 ? y + 1 : y - 1)) % ny;
  z2 = (nz + (dz >= 0.0 ? z + 1 : z - 1)) % nz;

  /* take abs(d{xyz}) to get weights for {xyz} and {xyz}2: */
  dx = fabs(dx);
  dy = fabs(dy);
  dz = fabs(dz);

#define D(x, y, z) (get_val(x, y, z, nx, ny, nz, last_dim_size, data, stride, conjugate))

  return (((D(x, y, z) * (1.0 - dx) + D(x2, y, z) * dx) * (1.0 - dy) +
           (D(x, y2, z) * (1.0 - dx) + D(x2, y2, z) * dx) * dy) *
              (1.0 - dz) +
          ((D(x, y, z2) * (1.0 - dx) + D(x2, y, z2) * dx) * (1.0 - dy) +
           (D(x, y2, z2) * (1.0 - dx) + D(x2, y2, z2) * dx) * dy) *
              dz);

#undef D
}

scalar_complex mode_solver::interp_cval(vector3 p, int nx, int ny, int nz, int last_dim_size,
                                        mpb_real *data, int stride) {
  scalar_complex cval;
  cval.re = interp_val(p, nx, ny, nz, last_dim_size, data, stride, 0);
  cval.im = interp_val(p, nx, ny, nz, last_dim_size, data + 1, stride, 1);
  return cval;
}

#define f_interp_val(p, f, data, stride, conj)                                                     \
  interp_val(p, f->nx, f->ny, f->nz, f->last_dim_size, data, stride, conj)
#define f_interp_cval(p, f, data, stride)                                                          \
  interp_cval(p, f->nx, f->ny, f->nz, f->last_dim_size, data, stride)

symmetric_matrix mode_solver::interp_eps_inv(vector3 p) {
  int stride = sizeof(symmetric_matrix) / sizeof(mpb_real);
  symmetric_matrix eps_inv;

  eps_inv.m00 = f_interp_val(p, mdata, &mdata->eps_inv->m00, stride, 0);
  eps_inv.m11 = f_interp_val(p, mdata, &mdata->eps_inv->m11, stride, 0);
  eps_inv.m22 = f_interp_val(p, mdata, &mdata->eps_inv->m22, stride, 0);
#ifdef WITH_HERMITIAN_EPSILON
  eps_inv.m01 = f_interp_cval(p, mdata, &mdata->eps_inv->m01.re, stride);
  eps_inv.m02 = f_interp_cval(p, mdata, &mdata->eps_inv->m02.re, stride);
  eps_inv.m12 = f_interp_cval(p, mdata, &mdata->eps_inv->m12.re, stride);
#else
  eps_inv.m01 = f_interp_val(p, mdata, &mdata->eps_inv->m01, stride, 0);
  eps_inv.m02 = f_interp_val(p, mdata, &mdata->eps_inv->m02, stride, 0);
  eps_inv.m12 = f_interp_val(p, mdata, &mdata->eps_inv->m12, stride, 0);
#endif
  return eps_inv;
}

mpb_real mode_solver::get_epsilon_point(vector3 p) {
  symmetric_matrix eps_inv;
  eps_inv = interp_eps_inv(p);
  return mean_medium_from_matrix(&eps_inv);
}

cmatrix3x3 mode_solver::get_epsilon_inverse_tensor_point(vector3 p) {
  symmetric_matrix eps_inv;
  eps_inv = interp_eps_inv(p);

#ifdef WITH_HERMITIAN_EPSILON
  return make_hermitian_cmatrix3x3(eps_inv.m00, eps_inv.m11, eps_inv.m22,
                                   cscalar2cnumber(eps_inv.m01), cscalar2cnumber(eps_inv.m02),
                                   cscalar2cnumber(eps_inv.m12));
#else
  return make_hermitian_cmatrix3x3(eps_inv.m00, eps_inv.m11, eps_inv.m22,
                                   make_cnumber(eps_inv.m01, 0), make_cnumber(eps_inv.m02, 0),
                                   make_cnumber(eps_inv.m12, 0));
#endif
}

mpb_real mode_solver::get_energy_point(vector3 p) {
  CHECK(curfield && strchr("DHBR", curfield_type),
        "compute-field-energy must be called before get-energy-point");
  return f_interp_val(p, mdata, (mpb_real *)curfield, 1, 0);
}

cvector3 mode_solver::get_bloch_field_point(vector3 p) {
  scalar_complex field[3];
  cvector3 F;

  CHECK(curfield && strchr("dhbecv", curfield_type),
        "field must be must be loaded before get-*field*-point");
  field[0] = f_interp_cval(p, mdata, &curfield[0].re, 6);
  field[1] = f_interp_cval(p, mdata, &curfield[1].re, 6);
  field[2] = f_interp_cval(p, mdata, &curfield[2].re, 6);
  F.x = cscalar2cnumber(field[0]);
  F.y = cscalar2cnumber(field[1]);
  F.z = cscalar2cnumber(field[2]);
  return F;
}

cvector3 mode_solver::get_field_point(vector3 p) {
  scalar_complex field[3], phase;
  cvector3 F;

  CHECK(curfield && strchr("dhbecv", curfield_type),
        "field must be must be loaded before get-*field*-point");
  field[0] = f_interp_cval(p, mdata, &curfield[0].re, 6);
  field[1] = f_interp_cval(p, mdata, &curfield[1].re, 6);
  field[2] = f_interp_cval(p, mdata, &curfield[2].re, 6);

  if (curfield_type != 'v') {
    mpb_real latx = geometry_lattice.size.x == 0 ? 1e-20 : geometry_lattice.size.x;
    mpb_real laty = geometry_lattice.size.y == 0 ? 1e-20 : geometry_lattice.size.y;
    mpb_real latz = geometry_lattice.size.z == 0 ? 1e-20 : geometry_lattice.size.z;

    double phase_phi = TWOPI * (cur_kvector.x * (p.x / latx) + cur_kvector.y * (p.y / laty) +
                                cur_kvector.z * (p.z / latz));

    CASSIGN_SCALAR(phase, cos(phase_phi), sin(phase_phi));
    CASSIGN_MULT(field[0], field[0], phase);
    CASSIGN_MULT(field[1], field[1], phase);
    CASSIGN_MULT(field[2], field[2], phase);
  }

  F.x = cscalar2cnumber(field[0]);
  F.y = cscalar2cnumber(field[1]);
  F.z = cscalar2cnumber(field[2]);

  return F;
}

void mode_solver::multiply_bloch_phase(std::complex<double> *cdata) {

  std::vector<mpb_real> kvector = get_output_k();

  scalar_complex *data = cdata ? (scalar_complex *)cdata : (scalar_complex *)curfield;

  int dims[] = {mdata->nx, mdata->ny, mdata->nz};
  int local_dims[] = {mdata->local_nx, mdata->ny, mdata->nz};
  int start[] = {mdata->local_x_start, 0, 0};

  mpb_real s[3]; /* the step size between grid points dotted with k */
  std::vector<scalar_complex> phasex(local_dims[0]);
  std::vector<scalar_complex> phasey(local_dims[1]);
  std::vector<scalar_complex> phasez(local_dims[2]);

  for (int i = 0; i < 3; ++i) {
    s[i] = TWOPI * kvector[i] / dims[i];
  }

  /* cache exp(ikx) along each of the directions, for speed */
  for (int i = 0; i < local_dims[0]; ++i) {
    mpb_real phase = s[0] * (i + start[0]);
    phasex[i].re = cos(phase);
    phasex[i].im = sin(phase);
  }
  for (int j = 0; j < local_dims[1]; ++j) {
    mpb_real phase = s[1] * (j + start[1]);
    phasey[j].re = cos(phase);
    phasey[j].im = sin(phase);
  }
  for (int k = 0; k < local_dims[2]; ++k) {
    mpb_real phase = s[2] * (k + start[2]);
    phasez[k].re = cos(phase);
    phasez[k].im = sin(phase);
  }

  /* Now, multiply field by exp(i k*r): */
  for (int i = 0; i < local_dims[0]; ++i) {
    scalar_complex px = phasex[i];

    for (int j = 0; j < local_dims[1]; ++j) {
      scalar_complex py;
      mpb_real re = phasey[j].re;
      mpb_real im = phasey[j].im;
      py.re = px.re * re - px.im * im;
      py.im = px.re * im + px.im * re;

      for (int k = 0; k < local_dims[2]; ++k) {
        int ijk = ((i * local_dims[1] + j) * local_dims[2] + k) * 3;
        mpb_real p_re, p_im;
        mpb_real re = phasez[k].re, im = phasez[k].im;

        p_re = py.re * re - py.im * im;
        p_im = py.re * im + py.im * re;

        for (int component = 0; component < 3; ++component) {
          int ijkc = ijk + component;
          re = data[ijkc].re;
          im = data[ijkc].im;
          data[ijkc].re = re * p_re - im * p_im;
          data[ijkc].im = im * p_re + re * p_im;
        }
      }
    }
  }
}

// Replace the current field with its scalar divergence; only works for Bloch fields
void mode_solver::compute_field_divergence() {
  scalar *field = (scalar *)curfield;
  scalar *field2 = mdata->fft_data == mdata->fft_data2
                       ? field
                       : (field == mdata->fft_data ? mdata->fft_data2 : mdata->fft_data);
  mpb_real scale;

  if (!curfield || !strchr("dhbec", curfield_type)) {
    meep::master_fprintf(stderr, "A Bloch-periodic field must be loaded.\n");
    return;
  }

  /* convert back to Fourier space */
  maxwell_compute_fft(-1, mdata, field, field2, 3, 3, 1);

  /* compute (k+G) dot field */
  for (int i = 0; i < mdata->other_dims; ++i) {
    for (int j = 0; j < mdata->last_dim; ++j) {
      int ij = i * mdata->last_dim_size + j;
      k_data cur_k = mdata->k_plus_G[ij];
      /* k+G = |k+G| (m x n) */
      mpb_real kx = cur_k.kmag * (cur_k.my * cur_k.nz - cur_k.mz * cur_k.ny);
      mpb_real ky = cur_k.kmag * (cur_k.mz * cur_k.nx - cur_k.mx * cur_k.nz);
      mpb_real kz = cur_k.kmag * (cur_k.mx * cur_k.ny - cur_k.my * cur_k.nz);
      ASSIGN_SCALAR(field2[ij],
                    SCALAR_RE(field2[3 * ij + 0]) * kx + SCALAR_RE(field2[3 * ij + 1]) * ky +
                        SCALAR_RE(field2[3 * ij + 2]) * kz,
                    SCALAR_IM(field2[3 * ij + 0]) * kx + SCALAR_IM(field2[3 * ij + 1]) * ky +
                        SCALAR_IM(field2[3 * ij + 2]) * kz);
    }
  }

  /* convert scalar field back to position space */
  maxwell_compute_fft(+1, mdata, field2, field, 1, 1, 1);

  // multiply by i (from divergence) and normalization (from FFT) and 2*pi (from k+G)
  scale = TWOPI / H.N;
  int N = mdata->fft_output_size;
  for (int i = 0; i < N; ++i) {
    CASSIGN_SCALAR(curfield[i], -CSCALAR_IM(curfield[i]) * scale, CSCALAR_RE(curfield[i]) * scale);
  }

  curfield_type = 'C'; // complex (Bloch) scalar field
}

/* Fix the phase of the current field (e/h/b/d) to a canonical value.
   Also changes the phase of the corresponding eigenvector by the
   same amount, so that future calculations will have a consistent
   phase.

   The following procedure is used, derived from a suggestion by Doug
   Allan of Corning: First, choose the phase to maximize the sum of
   the squares of the real parts of the components.  This doesn't fix
   the overall sign, though.  That is done (after incorporating the
   above phase) by: (1) find the largest absolute value of the real
   part, (2) find the point with the greatest spatial array index that
   has |real part| at least half of the largest value, and (3) make
   that point positive.

   In the case of inversion symmetry, on the other hand, the overall phase
   is already fixed, to within a sign, by the choice to make the Fourier
   transform purely real.  So, in that case we simply pick a sign, in
   a manner similar to (2) and (3) above. */
void mode_solver::fix_field_phase() {
  mpb_real sq_sum2[2] = {0, 0};
  mpb_real sq_sum[2];
  mpb_real maxabs = 0.0;

  int maxabs_index = 0;
  int maxabs_sign = 1;
  int i;

  double theta;
  scalar phase;

  if (!curfield || !strchr("dhbecv", curfield_type)) {
    meep::master_fprintf(stderr, "The D/H/E field must be loaded first.\n");
    return;
  }
  int N = mdata->fft_output_size * 3;

  /* Compute the phase that maximizes the sum of the squares of
     the real parts of the components.  Equivalently, maximize
     the real part of the sum of the squares. */
  for (i = 0; i < N; ++i) {
    mpb_real a = curfield[i].re;
    mpb_real b = curfield[i].im;
    sq_sum2[0] += a * a - b * b;
    sq_sum2[1] += 2 * a * b;
  }
  mpi_allreduce(sq_sum2, sq_sum, 2, mpb_real, SCALAR_MPI_TYPE, MPI_SUM, mpb_comm);

  /* compute the phase = exp(i*theta) maximizing the real part of
     the sum of the squares.  i.e., maximize:
       cos(2*theta)*sq_sum[0] - sin(2*theta)*sq_sum[1] */
  theta = 0.5 * atan2(-sq_sum[1], sq_sum[0]);
  phase.re = cos(theta);
  phase.im = sin(theta);

  /* Next, fix the overall sign.  We do this by first computing the
     maximum |real part| of the jmax component (after multiplying
     by phase), and then finding the last spatial index at which
     |real part| is at least half of this value.  The sign is then
     chosen to make the real part positive at that point.

        (Note that we can't just make the point of maximum |real part|
         positive, as that would be ambiguous in the common case of an
         oscillating field within the unit cell.)

        In the case of inversion symmetry (!SCALAR_COMPLEX), we work with
        (real part - imag part) instead of (real part), to insure that we
        have something that is nonzero somewhere. */

  for (i = 0; i < N; ++i) {
    mpb_real r = fabs(curfield[i].re * phase.re - curfield[i].im * phase.im);

    if (r > maxabs) { maxabs = r; }
  }

  mpi_allreduce_1(&maxabs, mpb_real, SCALAR_MPI_TYPE, MPI_MAX, mpb_comm);

  for (i = N - 1; i >= 0; --i) {
    mpb_real r = curfield[i].re * phase.re - curfield[i].im * phase.im;

    if (fabs(r) >= 0.5 * maxabs) {
      maxabs_index = i;
      maxabs_sign = r < 0 ? -1 : 1;
      break;
    }
  }

  if (i >= 0) { /* convert index to global index in distributed array: */
    maxabs_index += mdata->local_y_start * mdata->nx * mdata->nz;
  }

  {
    /* compute maximum index and corresponding sign over all the
       processors, using the MPI_MAXLOC reduction operation: */
    struct twoint_struct {
      int i;
      int s;
    } x;
    x.i = maxabs_index;
    x.s = maxabs_sign;
    mpi_allreduce_1(&x, struct twoint_struct, MPI_2INT, MPI_MAXLOC, mpb_comm);
    maxabs_index = x.i;
    maxabs_sign = x.s;
  }

  ASSIGN_SCALAR(phase, SCALAR_RE(phase) * maxabs_sign, SCALAR_IM(phase) * maxabs_sign);

  if (mpb_verbosity >= 1)
    meep::master_printf("Fixing %c-field (band %d) phase by %g + %gi; "
                        "max ampl. = %g\n",
                        curfield_type, curfield_band, SCALAR_RE(phase), SCALAR_IM(phase), maxabs);

  /* Now, multiply everything by this phase, *including* the
     stored "raw" eigenvector in H, so that any future fields
     that we compute will have a consistent phase: */
  for (i = 0; i < N; ++i) {
    mpb_real a = curfield[i].re;
    mpb_real b = curfield[i].im;
    curfield[i].re = a * SCALAR_RE(phase) - b * SCALAR_IM(phase);
    curfield[i].im = a * SCALAR_IM(phase) + b * SCALAR_RE(phase);
  }
  for (int i = 0; i < H.n; ++i) {
    mpb_real bbbb_re = H.data[i * H.p + curfield_band - 1].re;
    mpb_real bbbb_im = H.data[i * H.p + curfield_band - 1].im;
    mpb_real cccc_re = phase.re;
    mpb_real cccc_im = phase.im;
    H.data[i * H.p + curfield_band - 1].re = bbbb_re * cccc_re - bbbb_im * cccc_im;
    H.data[i * H.p + curfield_band - 1].im = bbbb_re * cccc_im + bbbb_im * cccc_re;
  }
}

void mode_solver::get_lattice(double data[3][3]) { matrix3x3_to_arr(data, Rm); }

std::vector<int> mode_solver::get_eigenvectors_slice_dims(int num_bands) {
  std::vector<int> res(3);
  res[0] = H.localN;
  res[1] = H.c;
  res[2] = num_bands;

  return res;
}

void mode_solver::get_eigenvectors(int p_start, int p, std::complex<mpb_real> *cdata, int size) {

  for (int i = 0, j = p_start; i < size; i += p, j += H.p) {
    for (int k = 0; k < p; ++k) {
      cdata[i + k] = std::complex<mpb_real>(H.data[j + k].re, H.data[j + k].im);
    }
  }
}

void mode_solver::set_eigenvectors(int b_start, std::complex<mpb_real> *cdata, int size) {
  int columns = size / H.n;

  for (int i = 0, j = b_start; i < size; i += columns, j += H.p) {
    for (int k = 0; k < columns; ++k) {
      H.data[j + k].re = cdata[i + k].real();
      H.data[j + k].im = cdata[i + k].imag();
    }
  }
  curfield_reset();
}

double mode_solver::get_eigensolver_flops() { return eigensolver_flops; }

int mode_solver::get_iterations() { return iterations; }

std::vector<mpb_real> mode_solver::compute_zparities() {
  std::vector<mpb_real> z_parity(num_bands);
  double *d = maxwell_zparity(H, mdata);

  for (int i = 0; i < num_bands; ++i) {
    z_parity[i] = d[i];
  }

  free(d);
  return z_parity;
}

std::vector<mpb_real> mode_solver::compute_yparities() {
  std::vector<mpb_real> y_parity(num_bands);
  double *d = maxwell_yparity(H, mdata);

  for (int i = 0; i < num_bands; ++i) {
    y_parity[i] = d[i];
  }

  free(d);
  return y_parity;
}

/* Compute the group velocity dw/dk in the given direction d (where
   the length of d is ignored).  d is in the reciprocal lattice basis.
   Should only be called after solve_kpoint.  Returns a list of the
   group velocities, one for each band, in units of c. */
std::vector<mpb_real> mode_solver::compute_group_velocity_component(vector3 d) {
  curfield_reset(); // has the side effect of overwriting curfield scratch

  if (!mdata) {
    meep::master_fprintf(stderr, "mode_solver.init must be called first!\n");
    return std::vector<mpb_real>(0);
  }
  if (!kpoint_index) {
    meep::master_fprintf(stderr, "mode_solver.solve_kpoint must be called first!\n");
    return std::vector<mpb_real>(0);
  }

  /* convert d to unit vector in Cartesian coords: */
  d = unit_vector3(matrix3x3_vector3_mult(Gm, d));
  mpb_real u[] = {d.x, d.y, d.z};

  std::vector<mpb_real> group_v(num_bands);
  std::vector<mpb_real> gv_scratch(num_bands * 2);

  /* now, compute group_v.items = diag Re <H| curl 1/eps i u x |H>: */

  /* ...we have to do this in blocks of eigensolver_block_size since
     the work matrix W[0] may not have enough space to do it all at once. */

  for (int ib = 0; ib < num_bands; ib += Hblock.alloc_p) {
    if (ib + mdata->num_bands > num_bands) {
      maxwell_set_num_bands(mdata, num_bands - ib);
      evectmatrix_resize(&W[0], num_bands - ib, 0);
      evectmatrix_resize(&Hblock, num_bands - ib, 0);
    }
    maxwell_compute_H_from_B(mdata, H, Hblock, (scalar_complex *)mdata->fft_data, ib, 0, Hblock.p);
    maxwell_ucross_op(Hblock, W[0], mdata, u);
    evectmatrix_XtY_diag_real(Hblock, W[0], gv_scratch.data(), gv_scratch.data() + group_v.size());
    {
      for (int ip = 0; ip < Hblock.p; ++ip)
        group_v[ib + ip] = gv_scratch[ip];
    }
  }

  /* Reset scratch matrix sizes: */
  evectmatrix_resize(&Hblock, Hblock.alloc_p, 0);
  evectmatrix_resize(&W[0], W[0].alloc_p, 0);
  maxwell_set_num_bands(mdata, Hblock.alloc_p);

  /* The group velocity is given by:

    grad_k(omega)*d = grad_k(omega^2)*d / 2*omega
       = grad_k(<H|maxwell_op|H>)*d / 2*omega
       = Re <H| curl 1/eps i u x |H> / omega

    Note that our k is in units of 2*Pi/a, and omega is in
    units of 2*Pi*c/a, so the result will be in units of c. */

  for (int i = 0; i < num_bands; ++i) {
    if (freqs[i] == 0) { /* v is undefined in this case */
      group_v[i] = 0.0;  /* just set to zero */
    }
    else {
      group_v[i] /= negative_epsilon_ok ? sqrt(fabs(freqs[i])) : freqs[i];
    }
  }

  return group_v;
}

/* as above, but only computes for given band */
mpb_real mode_solver::compute_1_group_velocity_component(vector3 d, int b) {
  mpb_real u[3];
  int ib = b - 1;
  mpb_real group_v = 0;
  mpb_real scratch;

  curfield_reset();

  if (!mdata) {
    meep::master_fprintf(stderr, "mode_solver.init must be called first!\n");
    return group_v;
  }

  if (!kpoint_index) {
    meep::master_fprintf(stderr, "mode_solver.solve_kpoint must be called first!\n");
    return group_v;
  }

  /* convert d to unit vector in Cartesian coords: */
  d = unit_vector3(matrix3x3_vector3_mult(Gm, d));
  u[0] = d.x;
  u[1] = d.y;
  u[2] = d.z;

  evectmatrix_resize(&W[0], 1, 0);
  CHECK(nwork_alloc > 1, "eigensolver-nwork is too small");
  evectmatrix_resize(&W[1], 1, 0);

  maxwell_compute_H_from_B(mdata, H, W[1], (scalar_complex *)mdata->fft_data, ib, 0, 1);
  maxwell_ucross_op(W[1], W[0], mdata, u);
  evectmatrix_XtY_diag_real(W[1], W[0], &group_v, &scratch);

  /* Reset scratch matrix sizes: */
  evectmatrix_resize(&W[1], W[1].alloc_p, 0);
  evectmatrix_resize(&W[0], W[0].alloc_p, 0);

  if (freqs[ib] == 0) { /* v is undefined in this case */
    group_v = 0.0;      /* just set to zero */
  }
  else {
    group_v /= negative_epsilon_ok ? sqrt(fabs(freqs[ib])) : freqs[ib];
  }
  return group_v;
}

/* returns group velocity for band b, in Cartesian coordinates */
vector3 mode_solver::compute_1_group_velocity(int b) {
  vector3 v;
  vector3 d;
  matrix3x3 RmT = matrix3x3_transpose(Rm);
  d.x = 1;
  d.y = 0;
  d.z = 0;
  v.x = compute_1_group_velocity_component(matrix3x3_vector3_mult(RmT, d), b);
  d.y = 1;
  d.x = 0;
  d.z = 0;
  v.y = compute_1_group_velocity_component(matrix3x3_vector3_mult(RmT, d), b);
  d.z = 1;
  d.y = 0;
  d.x = 0;
  v.z = compute_1_group_velocity_component(matrix3x3_vector3_mult(RmT, d), b);

  return v;
}

/* as above, but returns "group velocity" given by gradient of
   frequency with respect to k in reciprocal coords ... this is useful
   for band optimization. */
vector3 mode_solver::compute_1_group_velocity_reciprocal(int b) {
  return matrix3x3_vector3_mult(matrix3x3_transpose(Gm), compute_1_group_velocity(b));
}

/* compute the fraction of the field energy that is located in the
   given range of dielectric constants: */
mpb_real mode_solver::compute_energy_in_dielectric(mpb_real eps_low, mpb_real eps_high) {
  mpb_real *energy = (mpb_real *)curfield;
  mpb_real epsilon = 0.0;
  mpb_real energy_sum = 0.0;

  if (!curfield || !strchr("DHBR", curfield_type)) {
    meep::master_fprintf(stderr, "The D or H energy density must be loaded first.\n");
    return 0.0;
  }

  int N = mdata->fft_output_size;

  for (int i = 0; i < N; ++i) {
    epsilon = mean_medium_from_matrix(mdata->eps_inv + i);
    if (epsilon >= eps_low && epsilon <= eps_high) { energy_sum += energy[i]; }
  }
  mpi_allreduce_1(&energy_sum, mpb_real, SCALAR_MPI_TYPE, MPI_SUM, mpb_comm);
  energy_sum *= vol / H.N;
  return energy_sum;
}

/* For curfield and energy density, compute the fraction of the energy
   that resides inside the given list of geometric objects.   Later
   objects in the list have precedence, just like the ordinary
   geometry list. */
double mode_solver::compute_energy_in_objects(geometric_object_list objects) {

  mpb_real *energy = (mpb_real *)curfield;
  mpb_real energy_sum = 0;

  if (!curfield || !strchr("DHBR", curfield_type)) {
    meep::master_fprintf(stderr, "The D or H energy density must be loaded first.\n");
    return 0.0;
  }

  geom_fix_object_list(objects);

  int n1 = mdata->nx;
  int n2 = mdata->ny;
  int n3 = mdata->nz;

  mpb_real s1 = geometry_lattice.size.x / n1;
  mpb_real s2 = geometry_lattice.size.y / n2;
  mpb_real s3 = geometry_lattice.size.z / n3;
  mpb_real c1 = n1 <= 1 ? 0 : geometry_lattice.size.x * 0.5;
  mpb_real c2 = n2 <= 1 ? 0 : geometry_lattice.size.y * 0.5;
  mpb_real c3 = n3 <= 1 ? 0 : geometry_lattice.size.z * 0.5;

  LOOP_XYZ(mdata) {
    vector3 p;
    int n;
    p.x = i1 * s1 - c1;
    p.y = i2 * s2 - c2;
    p.z = i3 * s3 - c3;
    for (n = objects.num_items - 1; n >= 0; --n) {
      if (point_in_periodic_fixed_objectp(p, objects.items[n])) {
        // TODO:
        // if (((meep_geom::material_data *)objects.items[n].material)->which_subclass ==
        // MATERIAL_TYPE_SELF) {
        //   break; /* treat as a "nothing" object */
        // }
        energy_sum += energy[xyz_index];
        break;
      }
    }
  }
}
} // namespace py_mpb

mpi_allreduce_1(&energy_sum, mpb_real, SCALAR_MPI_TYPE, MPI_SUM, mpb_comm);
energy_sum *= vol / H.N;
return energy_sum;
}

cnumber mode_solver::compute_field_integral(field_integral_func field_func,
                                            field_integral_energy_func energy_func, void *py_func) {
  mpb_real *energy = (mpb_real *)curfield;
  cnumber integral = {0, 0};
  vector3 kvector = {0, 0, 0};

  if (!curfield || !strchr("dhbeDHBRcv", curfield_type)) {
    meep::master_fprintf(stderr, "The D or H energy/field must be loaded first.\n");
    return integral;
  }

  if (curfield_type != 'v') { kvector = cur_kvector; }

  int integrate_energy = strchr("DHBR", curfield_type) != NULL;

  int n1 = mdata->nx;
  int n2 = mdata->ny;
  int n3 = mdata->nz;

  mpb_real latx = geometry_lattice.size.x == 0 ? 1e-20 : geometry_lattice.size.x;
  mpb_real laty = geometry_lattice.size.y == 0 ? 1e-20 : geometry_lattice.size.y;
  mpb_real latz = geometry_lattice.size.z == 0 ? 1e-20 : geometry_lattice.size.z;

  mpb_real s1 = latx / n1;
  mpb_real s2 = laty / n2;
  mpb_real s3 = latz / n3;
  mpb_real c1 = n1 <= 1 ? 0 : latx * 0.5;
  mpb_real c2 = n2 <= 1 ? 0 : laty * 0.5;
  mpb_real c3 = n3 <= 1 ? 0 : latz * 0.5;

  LOOP_XYZ(mdata) {
    mpb_real epsilon = mean_medium_from_matrix(mdata->eps_inv + xyz_index);

    vector3 p;
    p.x = i1 * s1 - c1;
    p.y = i2 * s2 - c2;
    p.z = i3 * s3 - c3;

    if (integrate_energy) { integral.re += energy_func(energy[xyz_index], epsilon, p, py_func); }
    else {
      double phase_phi =
          TWOPI * (kvector.x * (p.x / latx) + kvector.y * (p.y / laty) + kvector.z * (p.z / latz));

      scalar_complex phase;
      CASSIGN_SCALAR(phase, cos(phase_phi), sin(phase_phi));

      cvector3 F;
      CASSIGN_MULT_RE(F.x.re, curfield[3 * xyz_index + 0], phase);
      CASSIGN_MULT_IM(F.x.im, curfield[3 * xyz_index + 0], phase);
      CASSIGN_MULT_RE(F.y.re, curfield[3 * xyz_index + 1], phase);
      CASSIGN_MULT_IM(F.y.im, curfield[3 * xyz_index + 1], phase);
      CASSIGN_MULT_RE(F.z.re, curfield[3 * xyz_index + 2], phase);
      CASSIGN_MULT_IM(F.z.im, curfield[3 * xyz_index + 2], phase);

      cnumber integrand = field_func(F, epsilon, p, py_func);

      integral.re += integrand.re;
      integral.im += integrand.im;
    }
  }
}
}

integral.re *= vol / H.N;
integral.im *= vol / H.N;
{
  cnumber integral_sum;
  mpi_allreduce(&integral, &integral_sum, 2, number, MPI_DOUBLE, MPI_SUM, mpb_comm);
  return integral_sum;
}
}

number mode_solver::compute_energy_integral(field_integral_func field_func,
                                            field_integral_energy_func energy_func, void *py_func) {
  if (!curfield || !strchr("DHBR", curfield_type)) {
    meep::master_fprintf(stderr, "The D or H energy density must be loaded first.\n");
    return 0.0;
  }

  return cnumber_re(compute_field_integral(field_func, energy_func, py_func));
}

vector3 mode_solver::get_dominant_planewave(int band) {
  double kdom[3] = {0, 0, 0};
#if MPB_VERSION_MAJOR > 1 || (MPB_VERSION_MAJOR == 1 && MPB_VERSION_MINOR >= 7)
  maxwell_dominant_planewave(mdata, H, band, kdom);
#endif
  vector3 result = {kdom[0], kdom[1], kdom[2]};
  return result;
}

// Used in MPBData python class

/* A macro to set x = fractional part of x input, xi = integer part,
   with 0 <= x < 1.0.   Note that we need the second test (if x >= 1.0)
   below, because x may start out as -0 or -1e-23 or something so that
   it is < 0 but x + 1.0 == 1.0, thanks to the wonders of floating point.
   (This has actually happened, on an Alpha.) */
#define MODF_POSITIVE(x, xi)                                                                       \
  {                                                                                                \
    x = modf(x, &xi);                                                                              \
    if (x < 0) {                                                                                   \
      x += 1.0;                                                                                    \
      if (x >= 1.0)                                                                                \
        x = 0;                                                                                     \
      else                                                                                         \
        xi -= 1.0;                                                                                 \
    }                                                                                              \
  }

#define ADJ_POINT(i1, i2, nx, dx, xi, xi2)                                                         \
  {                                                                                                \
    if (dx >= 0.0) {                                                                               \
      i2 = i1 + 1;                                                                                 \
      if (i2 >= nx) {                                                                              \
        i2 -= nx;                                                                                  \
        xi2 = xi + 1.0;                                                                            \
      }                                                                                            \
      else                                                                                         \
        xi2 = xi;                                                                                  \
    }                                                                                              \
    else {                                                                                         \
      i2 = i1 - 1;                                                                                 \
      if (i2 < 0) {                                                                                \
        i2 += nx;                                                                                  \
        xi2 = xi - 1.0;                                                                            \
      }                                                                                            \
      else                                                                                         \
        xi2 = xi;                                                                                  \
      dx = -dx;                                                                                    \
    }                                                                                              \
  }

#define MAX2(a, b) ((a) >= (b) ? (a) : (b))
#define MIN2(a, b) ((a) < (b) ? (a) : (b))

void add_cmplx_times_phase(mpb_real *sum_re, mpb_real *sum_im, mpb_real d_re, mpb_real d_im,
                           double ix, double iy, double iz, mpb_real *s, mpb_real scale_by) {
  static mpb_real phase = 0.0, p_re = 1.0, p_im = 0.0;
  mpb_real new_phase;

  new_phase = ix * s[0] + iy * s[1] + iz * s[2];
  if (new_phase != phase) {
    phase = new_phase;
    p_re = cos(phase);
    p_im = sin(phase);
  }
  *sum_re += (d_re * p_re - d_im * p_im) * scale_by;
  *sum_im += (d_re * p_im + d_im * p_re) * scale_by;
}

void map_data(mpb_real *d_in_re, int size_in_re, mpb_real *d_in_im, int size_in_im, int n_in[3],
              mpb_real *d_out_re, int size_out_re, mpb_real *d_out_im, int size_out_im,
              int n_out[3], matrix3x3 coord_map, mpb_real *kvector, bool pick_nearest, bool verbose,
              bool multiply_bloch_phase) {
  (void)size_in_re;
  (void)size_in_im;
  (void)size_out_re;

  mpb_real s[3]; /* phase difference per cell in each lattice direction */
  mpb_real min_out_re = 1e20, max_out_re = -1e20, min_out_im = 1e20, max_out_im = -1e20;
  mpb_real shiftx, shifty, shiftz;

  CHECK(d_in_re && d_out_re, "invalid arguments");
  CHECK((d_out_im && d_in_im) || (!d_out_im && !d_in_im),
        "both input and output must be real or complex");

  coord_map.c0 = vector3_scale(1.0 / n_out[0], coord_map.c0);
  coord_map.c1 = vector3_scale(1.0 / n_out[1], coord_map.c1);
  coord_map.c2 = vector3_scale(1.0 / n_out[2], coord_map.c2);

  for (int i = 0; i < 3; ++i) {
    if (kvector)
      s[i] = kvector[i] * TWOPI;
    else
      s[i] = 0;
  }

  /* Compute shift so that the origin of the output cell
     is mapped to the origin of the original primitive cell: */
  shiftx = 0.5 - (coord_map.c0.x * 0.5 * n_out[0] + coord_map.c1.x * 0.5 * n_out[1] +
                  coord_map.c2.x * 0.5 * n_out[2]);
  shifty = 0.5 - (coord_map.c0.y * 0.5 * n_out[0] + coord_map.c1.y * 0.5 * n_out[1] +
                  coord_map.c2.y * 0.5 * n_out[2]);
  shiftz = 0.5 - (coord_map.c0.z * 0.5 * n_out[0] + coord_map.c1.z * 0.5 * n_out[1] +
                  coord_map.c2.z * 0.5 * n_out[2]);

  for (int i = 0; i < n_out[0]; ++i)
    for (int j = 0; j < n_out[1]; ++j)
      for (int k = 0; k < n_out[2]; ++k) {
        mpb_real x, y, z;
        double xi, yi, zi, xi2, yi2, zi2;
        double dx, dy, dz, mdx, mdy, mdz;
        int i1, j1, k1, i2, j2, k2;
        int ijk;

        ijk = (i * n_out[1] + j) * n_out[2] + k;

        /* find the point corresponding to d_out[i,j,k] in
           the input array, and also find the next-nearest
           points. */
        x = coord_map.c0.x * i + coord_map.c1.x * j + coord_map.c2.x * k + shiftx;
        y = coord_map.c0.y * i + coord_map.c1.y * j + coord_map.c2.y * k + shifty;
        z = coord_map.c0.z * i + coord_map.c1.z * j + coord_map.c2.z * k + shiftz;
        MODF_POSITIVE(x, xi);
        MODF_POSITIVE(y, yi);
        MODF_POSITIVE(z, zi);

        if (multiply_bloch_phase) {
          xi += x;
          yi += y;
          zi += z;
        }

        i1 = x * n_in[0];
        j1 = y * n_in[1];
        k1 = z * n_in[2];
        dx = x * n_in[0] - i1;
        dy = y * n_in[1] - j1;
        dz = z * n_in[2] - k1;
        ADJ_POINT(i1, i2, n_in[0], dx, xi, xi2);
        ADJ_POINT(j1, j2, n_in[1], dy, yi, yi2);
        ADJ_POINT(k1, k2, n_in[2], dz, zi, zi2);

        /* dx, mdx, etcetera, are the weights for the various
           points in the input data, which we use for linearly
           interpolating to get the output point. */
        if (pick_nearest) {
          /* don't interpolate */
          dx = dx <= 0.5 ? 0.0 : 1.0;
          dy = dy <= 0.5 ? 0.0 : 1.0;
          dz = dz <= 0.5 ? 0.0 : 1.0;
        }
        mdx = 1.0 - dx;
        mdy = 1.0 - dy;
        mdz = 1.0 - dz;

        /* Now, linearly interpolate the input to get the
           output.  If the input/output are complex, we
           also need to multiply by the appropriate phase
           factor, depending upon which unit cell we are in. */

#define IN_INDEX(i, j, k) ((i * n_in[1] + j) * n_in[2] + k)
        if (size_out_im > 0) {
          d_out_re[ijk] = 0.0;
          d_out_im[ijk] = 0.0;
          add_cmplx_times_phase(d_out_re + ijk, d_out_im + ijk, d_in_re[IN_INDEX(i1, j1, k1)],
                                d_in_im[IN_INDEX(i1, j1, k1)], xi, yi, zi, s, mdx * mdy * mdz);
          add_cmplx_times_phase(d_out_re + ijk, d_out_im + ijk, d_in_re[IN_INDEX(i1, j1, k2)],
                                d_in_im[IN_INDEX(i1, j1, k2)], xi, yi, zi2, s, mdx * mdy * dz);
          add_cmplx_times_phase(d_out_re + ijk, d_out_im + ijk, d_in_re[IN_INDEX(i1, j2, k1)],
                                d_in_im[IN_INDEX(i1, j2, k1)], xi, yi2, zi, s, mdx * dy * mdz);
          add_cmplx_times_phase(d_out_re + ijk, d_out_im + ijk, d_in_re[IN_INDEX(i1, j2, k2)],
                                d_in_im[IN_INDEX(i1, j2, k2)], xi, yi2, zi2, s, mdx * dy * dz);
          add_cmplx_times_phase(d_out_re + ijk, d_out_im + ijk, d_in_re[IN_INDEX(i2, j1, k1)],
                                d_in_im[IN_INDEX(i2, j1, k1)], xi2, yi, zi, s, dx * mdy * mdz);
          add_cmplx_times_phase(d_out_re + ijk, d_out_im + ijk, d_in_re[IN_INDEX(i2, j1, k2)],
                                d_in_im[IN_INDEX(i2, j1, k2)], xi2, yi, zi2, s, dx * mdy * dz);
          add_cmplx_times_phase(d_out_re + ijk, d_out_im + ijk, d_in_re[IN_INDEX(i2, j2, k1)],
                                d_in_im[IN_INDEX(i2, j2, k1)], xi2, yi2, zi, s, dx * dy * mdz);
          add_cmplx_times_phase(d_out_re + ijk, d_out_im + ijk, d_in_re[IN_INDEX(i2, j2, k2)],
                                d_in_im[IN_INDEX(i2, j2, k2)], xi2, yi2, zi2, s, dx * dy * dz);
          min_out_im = MIN2(min_out_im, d_out_im[ijk]);
          max_out_im = MAX2(max_out_im, d_out_im[ijk]);
        }
        else {
          d_out_re[ijk] = d_in_re[IN_INDEX(i1, j1, k1)] * mdx * mdy * mdz +
                          d_in_re[IN_INDEX(i1, j1, k2)] * mdx * mdy * dz +
                          d_in_re[IN_INDEX(i1, j2, k1)] * mdx * dy * mdz +
                          d_in_re[IN_INDEX(i1, j2, k2)] * mdx * dy * dz +
                          d_in_re[IN_INDEX(i2, j1, k1)] * dx * mdy * mdz +
                          d_in_re[IN_INDEX(i2, j1, k2)] * dx * mdy * dz +
                          d_in_re[IN_INDEX(i2, j2, k1)] * dx * dy * mdz +
                          d_in_re[IN_INDEX(i2, j2, k2)] * dx * dy * dz;
        }
        min_out_re = MIN2(min_out_re, d_out_re[ijk]);
        max_out_re = MAX2(max_out_re, d_out_re[ijk]);
#undef IN_INDEX
      }

  if (verbose || mpb_verbosity >= 1) {
    meep::master_printf("real part range: %g .. %g\n", min_out_re, max_out_re);
    if (size_out_im > 0) meep::master_printf("imag part range: %g .. %g\n", min_out_im, max_out_im);
  }
}

bool with_hermitian_epsilon() {
#ifdef WITH_HERMITIAN_EPSILON
  return true;
#else
  return false;
#endif
}
} // namespace meep_mpb
