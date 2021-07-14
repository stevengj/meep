/* Copyright (C) 2005-2021 Massachusetts Institute of Technology.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "meep.hpp"
#include "config.h"

#ifdef HAVE_MPB
#include <mpb.h>
#include "adjust_verbosity.hpp"
#ifndef SCALAR_COMPLEX
#error Meep requires complex version of MPB
#endif
#endif

using namespace std;

namespace meep {

#ifdef HAVE_MPB

typedef struct {
  const double *s, *o;
  double frequency;
  ndim dim;
  const fields *f;

  /* for parallel efficiency, we first cache the epsinv data from each process, then
     sum_to_all to synchonize, and then re-run set_maxwell_dielectric with the cached data */
  double *cache;  // array of 6*ncache eps_inv matrix entries in the order that they are needed
  size_t ncache;  // allocated size of the cache
  size_t icache;  // current position in the cache
  bool use_cache; // whether we are using the cache
} meep_mpb_eps_data;

static int meep_mpb_eps(symmetric_matrix *eps, symmetric_matrix *eps_inv, mpb_real n[3],
                        mpb_real d1, mpb_real d2, mpb_real d3, mpb_real tol, const mpb_real r[3],
                        void *eps_data_) {
  adjust_mpb_verbosity amv;
  meep_mpb_eps_data *eps_data = (meep_mpb_eps_data *)eps_data_;
  size_t i = eps_data->icache;

  (void)n;
  (void)d1;
  (void)d2;
  (void)d3;
  (void)tol; // unused

  if (eps_data->use_cache) {
    const double *cache = eps_data->cache;
    eps_inv->m00 = cache[6 * i];
    eps_inv->m11 = cache[6 * i + 1];
    eps_inv->m22 = cache[6 * i + 2];
    ASSIGN_ESCALAR(eps_inv->m01, cache[6 * i + 3], 0);
    ASSIGN_ESCALAR(eps_inv->m02, cache[6 * i + 4], 0);
    ASSIGN_ESCALAR(eps_inv->m12, cache[6 * i + 5], 0);
    maxwell_sym_matrix_invert(eps, eps_inv);
  }
  else {
    // get cache pointer, doubling cache size as needed:
    double *cache = i < eps_data->ncache
                        ? eps_data->cache
                        : (eps_data->cache = (double *)realloc(
                               eps_data->cache, sizeof(double) * 6 * (eps_data->ncache *= 2)));

    const double *s = eps_data->s;
    const double *o = eps_data->o;
    double frequency = eps_data->frequency;
    vec p(eps_data->dim == D3 ? vec(o[0] + r[0] * s[0], o[1] + r[1] * s[1], o[2] + r[2] * s[2])
                              : (eps_data->dim == D2 ? vec(o[0] + r[0] * s[0], o[1] + r[1] * s[1]) :
                                                     /* D1 */ vec(o[2] + r[2] * s[2])));
    const fields *f = eps_data->f;

    // call get_chi1inv with parallel=false to get only local epsilon data
    cache[6 * i] = real(f->get_chi1inv(Ex, X, p, frequency, false));
    cache[6 * i + 1] = real(f->get_chi1inv(Ey, Y, p, frequency, false));
    cache[6 * i + 2] = real(f->get_chi1inv(Ez, Z, p, frequency, false));
    cache[6 * i + 3] = real(f->get_chi1inv(Ex, Y, p, frequency, false));
    cache[6 * i + 4] = real(f->get_chi1inv(Ex, Z, p, frequency, false));
    cache[6 * i + 5] = real(f->get_chi1inv(Ey, Z, p, frequency, false));

    // return a dummy value epsilon = 1 while we are building up the cache
    eps->m00 = eps->m11 = eps->m22 = eps_inv->m00 = eps_inv->m11 = eps_inv->m22 = 1;
    ASSIGN_ESCALAR(eps->m01, 0, 0);
    eps->m02 = eps->m12 = eps_inv->m01 = eps_inv->m02 = eps_inv->m12 = eps->m01;
  }

  eps_data->icache += 1; // next call will use the subsequent cache element

  return 1; // tells MPB not to do its own subpixel averaging
}

/**************************************************************/
/* prototype for position-dependent amplitude function passed */
/* to add_volume_source                                       */
/**************************************************************/
typedef complex<double> (*amplitude_function)(const vec &);

// default implementation of amplitude_function
static complex<double> default_amp_func(const vec &pt) {
  (void)pt;
  return 1.0;
}

/*******************************************************************/
/* structure storing all data needed to compute position-dependent */
/* amplitude for eigenmode source (the fields of this structure    */
/* were formerly global variables)                                 */
/* Note: 'Gk' is the k-point in real space, i.e. G*k where         */
/* G = matrix of reciprocal-lattice basis vectors                  */
/* k = k vector in reciprocal-lattice basis                        */
/*******************************************************************/
typedef struct eigenmode_data {
  maxwell_data *mdata;
  scalar_complex *fft_data_H, *fft_data_E;
  evectmatrix H;
  int n[3];
  double s[3];
  double Gk[3];
  vec center;
  amplitude_function amp_func;
  int band_num;
  double frequency;
  double group_velocity;
} eigenmode_data;

#define TWOPI 6.2831853071795864769252867665590057683943388

// utility routine for modular arithmetic that always returns a nonnegative integer
static int pmod(int n, int modulus) {
  n = n % modulus;
  if (n < 0) n += modulus;
  return n;
}

/*******************************************************************/
/* compute position-dependent amplitude for eigenmode source       */
/*  (similar to the routine formerly called meep_mpb_A)            */
/*******************************************************************/
complex<double> eigenmode_amplitude(void *vedata, const vec &p, component c) {
  eigenmode_data *edata = (eigenmode_data *)vedata;
  if (!edata || !(edata->mdata)) meep::abort("%s:%i: internal error", __FILE__, __LINE__);

  int *n = edata->n;
  double *s = edata->s;
  vec center = edata->center;
  amplitude_function amp_func = edata->amp_func;

  complex<mpb_real> *cdata =
      (complex<mpb_real> *)((c >= Hx) ? edata->fft_data_H : edata->fft_data_E);
  const complex<mpb_real> *data;
  switch (c) {
    case Ex:
      cdata = (complex<mpb_real> *)edata->fft_data_E;
      data = cdata + 0;
      break;
    case Ey:
      cdata = (complex<mpb_real> *)edata->fft_data_E;
      data = cdata + 1;
      break;
    case Ez:
      cdata = (complex<mpb_real> *)edata->fft_data_E;
      data = cdata + 2;
      break;
    case Hx:
      cdata = (complex<mpb_real> *)edata->fft_data_H;
      data = cdata + 0;
      break;
    case Hy:
      cdata = (complex<mpb_real> *)edata->fft_data_H;
      data = cdata + 1;
      break;
    case Hz:
      cdata = (complex<mpb_real> *)edata->fft_data_H;
      data = cdata + 2;
      break;
    default: meep::abort("invalid component in eigenmode_amplitude");
  };

  int nx = n[0];
  int ny = n[1];
  int nz = n[2];
  double r[3] = {0, 0, 0};
  vec p0(p - center);
  double phase = 0;
  LOOP_OVER_DIRECTIONS(p.dim, d) {
    double pd = p0.in_direction(d);
    int i = d % 3;
    phase += edata->Gk[i] * pd; // k dot p
    r[i] = pd / s[i] + 0.5;
  }
  double rx = r[0], ry = r[1], rz = r[2];

  /* linearly interpolate the amplitude from MPB at point p */
  int x, y, z, x2, y2, z2;
  double dx, dy, dz;

  /* get the point corresponding to r in the epsilon array grid: */
  x = int(rx * nx);
  y = int(ry * ny);
  z = int(rz * nz);

  /* get the difference between (x,y,z) and the actual point */
  dx = rx * nx - x;
  dy = ry * ny - y;
  dz = rz * nz - z;

  /* wrap around to 0..n-1, assuming periodic boundaries */
  x = pmod(x, nx);
  y = pmod(y, ny);
  z = pmod(z, nz);

  /* get the other closest point in the grid, with periodic boundaries: */
  x2 = pmod((dx >= 0.0 ? x + 1 : x - 1), nx);
  y2 = pmod((dy >= 0.0 ? y + 1 : y - 1), ny);
  z2 = pmod((dz >= 0.0 ? z + 1 : z - 1), nz);
  x = x % nx;
  y = y % ny;
  z = z % nz;

  /* take abs(d{xyz}) to get weights for {xyz} and {xyz}2: */
  dx = fabs(dx);
  dy = fabs(dy);
  dz = fabs(dz);

  /* define a macro to give us data(x,y,z) on the grid,
     in row-major order (the order used by MPB): */
#define D(x, y, z) (data[(((x)*ny + (y)) * nz + (z)) * 3])
  complex<mpb_real> ret;
  ret = (((D(x, y, z) * (1.0 - dx) + D(x2, y, z) * dx) * (1.0 - dy) +
          (D(x, y2, z) * (1.0 - dx) + D(x2, y2, z) * dx) * dy) *
             (1.0 - dz) +
         ((D(x, y, z2) * (1.0 - dx) + D(x2, y, z2) * dx) * (1.0 - dy) +
          (D(x, y2, z2) * (1.0 - dx) + D(x2, y2, z2) * dx) * dy) *
             dz);
#undef D
  return (complex<double>(double(real(ret)), double(imag(ret))) * amp_func(p)) *
         std::polar(1.0, TWOPI * phase);
}

/***************************************************************/
/* entry point to eigenmode_amplitude with the right prototype */
/* for passage as the A parameter to add_volume_source         */
/***************************************************************/
static eigenmode_data *global_eigenmode_data = 0;
static component global_eigenmode_component;
static complex<double> meep_mpb_A(const vec &p) {
  return eigenmode_amplitude((void *)global_eigenmode_data, p, global_eigenmode_component);
}

// compute axb = a cross b
static void cross_product(mpb_real axb[3], const mpb_real a[3], const mpb_real b[3]) {
  axb[0] = a[1] * b[2] - a[2] * b[1];
  axb[1] = a[2] * b[0] - a[0] * b[2];
  axb[2] = a[0] * b[1] - a[1] * b[0];
}
static double dot_product(const mpb_real a[3], const mpb_real b[3]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// return the next number factorizable into powers of 2,3,5,7
// for efficient FFTs, similar to optimize-grid-size! in MPB:
static int nextpow2357(int n) {
  while (1) {
    int m = n;
    while (m % 3 == 0)
      m /= 3;
    while (m % 5 == 0)
      m /= 5;
    while (m % 7 == 0)
      m /= 7;
    if ((m & (m - 1)) == 0) // if m is a power of 2
      return n;
    n += 1;
  }
}

/****************************************************************/
/* call MPB to get the band_numth eigenmode at freq frequency.  */
/*                                                              */
/* this routine constitutes the first 75% of what was formerly  */
/* add_eigenmode_source; it has been split off as a separate    */
/* routine to allow it to be followed either by                 */
/*  (a) add_eigenmode_src()                                     */
/* or                                                           */
/*  (b) get_eigenmode_coefficients()                            */
/*                                                              */
/* the return value is an opaque pointer to an eigenmode_data   */
/* structure (needs to be opaque to allow compilation without   */
/* MPB, in which case maxwell_data and other types aren't       */
/* defined). this structure may then be passed to               */
/* eigenmode_amplitude (above) to compute eigenmode E and H     */
/* field components at arbitrary points in space.               */
/* call destroy_eigenmode_data() to deallocate when finished.   */
/****************************************************************/
void *fields::get_eigenmode(double frequency, direction d, const volume where, const volume eig_vol,
                            int band_num, const vec &_kpoint, bool match_frequency, int parity,
                            double resolution, double eigensolver_tol, double *kdom,
                            void **user_mdata, diffractedplanewave *dp) {
  /*--------------------------------------------------------------*/
  /*- part 1: preliminary setup for calling MPB  -----------------*/
  /*--------------------------------------------------------------*/
  adjust_mpb_verbosity amv;

  // if the mode region extends over the full computational grid and we are bloch-periodic
  // in any direction, set the corresponding component of the eigenmode initial-guess
  // k-vector to be the (real part of the) bloch vector in that direction.
  vec kpoint(_kpoint);
  LOOP_OVER_DIRECTIONS(v.dim, dd) {
    if (dd != d && float(eig_vol.in_direction(dd)) == float(v.in_direction(dd)))
      if (boundaries[High][dd] == Periodic && boundaries[Low][dd] == Periodic)
        kpoint.set_direction(dd, real(k[dd]));
  }

  bool empty_dim[3] = {false, false, false};

  // special case: 2d cell in x and y with non-zero kz
  if ((eig_vol.dim == D3) && (float(v.in_direction(Z)) == float(1 / a)) &&
      (boundaries[High][Z] == Periodic && boundaries[Low][Z] == Periodic) && (kpoint.z() == 0) &&
      (real(k[Z]) != 0)) {
    kpoint.set_direction(Z, real(k[Z]));
    empty_dim[2] = true;
  }

  if (resolution <= 0.0) resolution = 2 * gv.a; // default to twice resolution
  int n[3], local_N, N_start, alloc_N, mesh_size[3] = {1, 1, 1};
  mpb_real k[3] = {0, 0, 0}, kcart[3] = {0, 0, 0};
  double s[3] = {0, 0, 0}, o[3] = {0, 0, 0};
  mpb_real R[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
  mpb_real G[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
  mpb_real kdir[3] = {0, 0, 0};
  double match_tol = eigensolver_tol * 10;

  if ((d == NO_DIRECTION && abs(_kpoint) == 0) || coordinate_mismatch(gv.dim, d))
    meep::abort("invalid direction in add_eigenmode_source");
  if (where.dim != gv.dim || eig_vol.dim != gv.dim)
    meep::abort("invalid volume dimensionality in add_eigenmode_source");

  if (!eig_vol.contains(where))
    meep::abort("invalid grid_volume in add_eigenmode_source (WHERE must be in EIG_VOL)");

  switch (gv.dim) {
    case D3:
      o[0] = eig_vol.in_direction_min(X);
      o[1] = eig_vol.in_direction_min(Y);
      o[2] = eig_vol.in_direction_min(Z);
      s[0] = eig_vol.in_direction(X);
      s[1] = eig_vol.in_direction(Y);
      s[2] = eig_vol.in_direction(Z);
      kcart[0] = kpoint.in_direction(X);
      kcart[1] = kpoint.in_direction(Y);
      kcart[2] = kpoint.in_direction(Z);
      break;
    case D2:
      o[0] = eig_vol.in_direction_min(X);
      o[1] = eig_vol.in_direction_min(Y);
      s[0] = eig_vol.in_direction(X);
      s[1] = eig_vol.in_direction(Y);
      kcart[0] = kpoint.in_direction(X);
      kcart[1] = kpoint.in_direction(Y);
      kcart[2] = beta; // special_kz feature
      empty_dim[2] = true;
      break;
    case D1:
      o[2] = eig_vol.in_direction_min(Z);
      s[2] = eig_vol.in_direction(Z);
      kcart[2] = kpoint.in_direction(Z);
      empty_dim[0] = empty_dim[1] = true;
      break;
    default: meep::abort("unsupported dimensionality in add_eigenmode_source");
  }

  double kcart_len = sqrt(dot_product(kcart, kcart));

  for (int i = 0; i < 3; ++i) {
    n[i] = int(resolution * s[i] + 0.5);
    if (n[i] == 0) n[i] = 1;
    n[i] = nextpow2357(n[i]);
    if (s[i] != 0)
      R[i][i] = s[i];
    else {
      if (d != NO_DIRECTION || empty_dim[i])
        R[i][i] = 1;
      else { // get lattice vector from kpoint
        for (int j = 0; j < 3; ++j)
          R[i][j] = kcart[j] / kcart_len;
      }
      s[i] = 1;
    }
  }

  for (int i = 0; i < 3; ++i) {
    k[i] = dot_product(R[i], kcart); // convert k to reciprocal basis
    // G = inverse of R transpose, via cross-product formula
    cross_product(G[i], R[(i + 1) % 3], R[(i + 2) % 3]);
    double GdotR = dot_product(G[i], R[i]);
    for (int j = 0; j < 3; ++j)
      G[i][j] /= GdotR;
  }

  if (verbosity > 1) master_printf("KPOINT: %g, %g, %g\n", k[0], k[1], k[2]);

  maxwell_data *mdata;
  if (!user_mdata || *user_mdata == NULL) {
    mdata = create_maxwell_data(n[0], n[1], n[2], &local_N, &N_start, &alloc_N, band_num, band_num);
    if (local_N != n[0] * n[1] * n[2]) meep::abort("MPI version of MPB library not supported");

    meep_mpb_eps_data eps_data;
    eps_data.s = s;
    eps_data.o = o;
    eps_data.dim = gv.dim;
    eps_data.f = this;
    eps_data.frequency = frequency;
    eps_data.cache = (double *)malloc(sizeof(double) * 6 * (eps_data.ncache = 512));

    // first, build up a cache of the local epsilon data (while returning dummy values to MPB)
    eps_data.use_cache = false;
    eps_data.icache = 0;
    set_maxwell_dielectric(mdata, mesh_size, R, G, NULL, meep_mpb_eps, &eps_data);

    // then, synchronize the data
    eps_data.ncache = eps_data.icache; // actual amount of cached data
    double *summed_cache = (double *)malloc(sizeof(double) * 6 * eps_data.ncache);
    am_now_working_on(MpiAllTime);
    sum_to_all(eps_data.cache, summed_cache, eps_data.ncache * 6);
    finished_working();
    free(eps_data.cache);
    eps_data.cache = summed_cache;

    // finally, send MPB the real epsilon data using the synchronized cache
    eps_data.use_cache = true;
    eps_data.icache = 0;
    set_maxwell_dielectric(mdata, mesh_size, R, G, NULL, meep_mpb_eps, &eps_data);
    assert(eps_data.icache == eps_data.ncache);

    free(eps_data.cache);

    if (user_mdata) *user_mdata = (void *)mdata;
  }
  else {
    mdata = (maxwell_data *)(*user_mdata);
    maxwell_set_num_bands(mdata, band_num);
    N_start = mdata->N_start;
    local_N = mdata->local_N;
    alloc_N = mdata->alloc_N;
  }

  if (check_maxwell_dielectric(mdata, 0)) meep::abort("invalid dielectric function for MPB");

  double kmatch;
  if (d == NO_DIRECTION) {
    for (int i = 0; i < 3; ++i)
      kdir[i] = kcart[i] / kcart_len;
    if (gv.dim == D2) {
      kdir[2] = 0; // beta is fixed
      kmatch = sqrt(kcart[0] * kcart[0] + kcart[1] * kcart[1]);
    }
    else
      kmatch = kcart_len;
  }
  else {
    kmatch = G[d - X][d - X] * k[d - X]; // k[d] in cartesian
    kdir[d - X] = 1;                     // kdir = unit vector in d direction
  }

  // if match_frequency is true, we need at least a crude guess for kmatch;
  // which we automatically pick if kmatch == 0.
  if (match_frequency && kmatch == 0) {
    vec cen = eig_vol.center();
    kmatch = copysign(frequency * sqrt(real(get_eps(cen, frequency)) * real(get_mu(cen, frequency))), _kpoint.in_direction(d));
    if (d == NO_DIRECTION) {
      for (int i = 0; i < 3; ++i)
        k[i] = dot_product(R[i], kdir) * kmatch; // kdir*kmatch in reciprocal basis
      if (gv.dim == D2) k[2] = beta;
    }
    else {
      k[d - X] = kmatch * R[d - X][d - X]; // convert to reciprocal basis
      if (eig_vol.in_direction(d) > 0 &&
          fabs(k[d - X]) > 0.4) // ensure k is well inside the Brillouin zone
        k[d - X] = k[d - X] > 0 ? 0.4 : -0.4;
    }
    if (verbosity > 1) master_printf("NEW KPOINT: %g, %g, %g\n", k[0], k[1], k[2]);
  }

  set_maxwell_data_parity(mdata, parity);
  update_maxwell_data_k(mdata, k, G[0], G[1], G[2]);

  if (k[0] == 0 && k[1] == 0 && k[2] == 0) {
    evectmatrix H;
    H.p = band_num;
    H.c = 2;
    band_num -= maxwell_zero_k_num_const_bands(H, mdata);
    if (band_num == 0) meep::abort("zero-frequency bands at k=0 are ill-defined");
  }

  evectmatrix H = create_evectmatrix(n[0] * n[1] * n[2], 2, band_num, local_N, N_start, alloc_N);
  /* initialize H to pseudorandom values on the master process; on other
     processes we get the value via broadcast() below */
  if (am_master()) {
    set_random_seed(314159);
    for (int i = 0; i < H.n * H.p; ++i) {
      ASSIGN_SCALAR(H.data[i], uniform_random(-1,1), uniform_random(-1,1));
    }
    restore_random_seed();
  }

  mpb_real *eigvals = new mpb_real[band_num];
  int num_iters;
  evectmatrix W[3];
  for (int i = 0; i < 3; ++i)
    W[i] = create_evectmatrix(n[0] * n[1] * n[2], 2, band_num, local_N, N_start, alloc_N);

  evectconstraint_chain *constraints = NULL;
  constraints = evect_add_constraint(constraints, maxwell_parity_constraint, (void *)mdata);
  if (k[0] == 0 && k[1] == 0 && k[2] == 0)
    constraints = evect_add_constraint(constraints, maxwell_zero_k_constraint, (void *)mdata);

  mpb_real vgrp; // Re( W[0]* (dTheta/dk) W[0] ) = group velocity

  // track #times change in kmatch increases to detect non-convergence
  double dkmatch_prev = kmatch;
  int count_dkmatch_increase = 0;

  /*--------------------------------------------------------------*/
  /*- part 2: newton iteration loop with call to MPB on each step */
  /*-         until eigenmode converged to requested tolerance    */
  /*--------------------------------------------------------------*/
  if (am_master() && !dp) do {
      eigensolver(H, eigvals, maxwell_operator, (void *)mdata,
#if MPB_VERSION_MAJOR > 1 || (MPB_VERSION_MAJOR == 1 && MPB_VERSION_MINOR >= 6)
                  NULL, NULL, /* eventually, we can support mu here */
#endif
                  maxwell_preconditioner2, (void *)mdata, evectconstraint_chain_func,
                  (void *)constraints, W, 3, eigensolver_tol, &num_iters,
                  EIGS_DEFAULT_FLAGS | (am_master() && verbosity > 1 ? EIGS_VERBOSE : 0));
      if (verbosity > 0)
        master_printf("MPB solved for frequency_%d(%g,%g,%g) = %g after %d iters\n", band_num,
                      G[0][0] * k[0], G[1][1] * k[1], G[2][2] * k[2], sqrt(eigvals[band_num - 1]),
                      num_iters);

      // copy desired single eigenvector into scratch arrays
      evectmatrix_resize(&W[0], 1, 0);
      evectmatrix_resize(&W[1], 1, 0);
      for (int i = 0; i < H.n; ++i)
        W[0].data[i] = H.data[H.p - 1 + i * H.p];

      // compute the group velocity in the kdir direction
      maxwell_ucross_op(W[0], W[1], mdata, kdir); // W[1] = (dTheta/dk) W[0]
      mpb_real vscratch;
      evectmatrix_XtY_diag_real(W[0], W[1], &vgrp, &vscratch);
      vgrp /= sqrt(eigvals[band_num - 1]);

      // return to original size
      evectmatrix_resize(&W[0], band_num, 0);
      evectmatrix_resize(&W[1], band_num, 0);

      if (match_frequency) {
        // update k via Newton step
        double dkmatch = (sqrt(eigvals[band_num - 1]) - frequency) / vgrp;
        kmatch = kmatch - dkmatch;
        if (verbosity > 1)
          master_printf("Newton step: group velocity v=%g, kmatch=%g\n", vgrp, kmatch);
        count_dkmatch_increase += fabs(dkmatch) > fabs(dkmatch_prev);
        if (count_dkmatch_increase > 4) {
          eigvals[band_num - 1] = -1;
          break;
        }
        if (d == NO_DIRECTION) {
          for (int i = 0; i < 3; ++i)
            k[i] = dot_product(R[i], kdir) * kmatch; // kdir*kmatch in reciprocal basis
          if (gv.dim == D2) k[2] = beta;
        }
        else {
          k[d - X] = kmatch * R[d - X][d - X];
        }
        update_maxwell_data_k(mdata, k, G[0], G[1], G[2]);
      }
    } while (match_frequency &&
             fabs(sqrt(eigvals[band_num - 1]) - frequency) > frequency * match_tol);

  if (dp) {
    scalar_complex s = {real(dp->get_s()),imag(dp->get_s())};
    scalar_complex p = {real(dp->get_p()),imag(dp->get_p())};

    // compute sum of (kparallel+G)^2 in all the periodic directions
    double k2sum = 0, ktmp = 0;
    int m = 0;
    LOOP_OVER_DIRECTIONS(v.dim, dd) {
      m = dp->get_g()[dd - X];
      ktmp = kpoint.in_direction(dd);
      if (eig_vol.in_direction(dd) != 0)
        ktmp += m/eig_vol.in_direction(dd);
      k2sum += ktmp*ktmp;
    }

    // compute kperp (if it is non evanescent) OR
    // frequency from kperp^2 and sum of (kparallel+G)^2
    {
      direction dd = eig_vol.normal_direction();
      if (match_frequency) {
        vec cen = eig_vol.center();
        double nn = sqrt(real(get_eps(cen, frequency)) * real(get_mu(cen, frequency)));
        double k2 = frequency*frequency*nn*nn - k2sum;
        if (k2 < 0) {
          master_printf("WARNING: diffraction order for g=(%d,%d,%d) is evanescent!\n",
                        dp->get_g()[0],dp->get_g()[1],dp->get_g()[2]);
          return NULL;
        }
        else if (k2 > 0)
          k[dd - X] = sqrt(k2);
      }
      else
        frequency = sqrt(kpoint.in_direction(dd)*kpoint.in_direction(dd) + k2sum);
    }

    if (am_master()) {
      update_maxwell_data_k(mdata, k, G[0], G[1], G[2]);
      maxwell_set_planewave(mdata, H, band_num, dp->get_g(), s, p, dp->get_axis());
      eigvals[band_num - 1] = frequency*frequency;
      evectmatrix_resize(&W[0], 1, 0);
      evectmatrix_resize(&W[1], 1, 0);
      for (int i = 0; i < H.n; ++i)
        W[0].data[i] = H.data[H.p - 1 + i * H.p];
      maxwell_ucross_op(W[0], W[1], mdata, kdir); // W[1] = (dTheta/dk) W[0]
      mpb_real vscratch;
      evectmatrix_XtY_diag_real(W[0], W[1], &vgrp, &vscratch);
      vgrp /= sqrt(eigvals[band_num - 1]);
    }
  }

  double eigval = eigvals[band_num - 1];

  // cleanup temporary storage
  delete[] eigvals;
  evect_destroy_constraints(constraints);
  for (int i = 0; i < 3; ++i)
    destroy_evectmatrix(W[i]);

  am_now_working_on(MpiAllTime);
  /* We only run MPB eigensolver on the master process to avoid
     any possibility of inconsistent mode solutions (#568) */
  eigval = broadcast(0, eigval);
  broadcast(0, k, 3);
  vgrp = broadcast(0, vgrp);
  if (eigval < 0) { // no mode found
    destroy_evectmatrix(H);
    if (!user_mdata) destroy_maxwell_data(mdata);
    return NULL;
  }
  if (!am_master()) update_maxwell_data_k(mdata, k, G[0], G[1], G[2]);
  broadcast(0, (double *)H.data, 2 * H.n * H.p);
  finished_working();

  if (!match_frequency) frequency = sqrt(eigval);

  /*--------------------------------------------------------------*/
  /*- part 3: do one stage of postprocessing to tabulate H-field  */
  /*-         components on the internal storage buffer in mdata  */
  /*--------------------------------------------------------------*/
  complex<mpb_real> *cdata = (complex<mpb_real> *)mdata->fft_data;

  maxwell_compute_h_from_H(mdata, H, (scalar_complex *)cdata, band_num - 1, 1);
  /* choose deterministic phase, maximizing power in real part;
     see fix_field_phase routine in MPB.*/
  {
    int i, N = mdata->fft_output_size * 3;
    double sq_sum0 = 0, sq_sum1 = 0, maxabs = 0.0;
    double theta;
    for (i = 0; i < N; ++i) {
      double a = real(cdata[i]), b = imag(cdata[i]);
      sq_sum0 += a * a - b * b;
      sq_sum1 += 2 * a * b;
    }
    theta = 0.5 * atan2(-sq_sum1, sq_sum0);
    complex<mpb_real> phase(cos(theta), sin(theta));
    phase /= sqrt(fabs(R[0][0] * R[1][1] * R[2][2]));
    for (i = 0; i < N; ++i) {
      double r = fabs(real(cdata[i] * phase));
      if (r > maxabs) maxabs = r;
    }
    for (i = N - 1; i >= 0 && fabs(real(cdata[i] * phase)) < 0.5 * maxabs; --i)
      ;
    if (real(cdata[i] * phase) < 0) phase = -phase;
    for (i = 0; i < N; ++i)
      cdata[i] *= phase;
    complex<mpb_real> *hdata = (complex<mpb_real> *)H.data;
    for (i = 0; i < H.n; ++i)
      hdata[i * H.p + (band_num - 1)] *= phase;
  }

  /*--------------------------------------------------------------*/
  /* do a second round of post-processing to tabulate E-fields   -*/
  /* on a (separate) internal storage buffer.  (Previously       -*/
  /* there was only one internal buffer which held either E-field */
  /* or H-field data, but this is inconvenient for cases in which */
  /* you want the E and H fields of an eigenmode simultaneously.) */
  /*--------------------------------------------------------------*/
  int NFFT = 3 * mdata->fft_output_size;
  scalar_complex *fft_data_E = (scalar_complex *)malloc(NFFT * sizeof(scalar_complex));

  maxwell_compute_d_from_H(mdata, H, fft_data_E, band_num - 1, 1);

  // d_from_H actually computes -frequency*D (see mpb/src/maxwell/maxwell_op.c),
  // so we need to divide the E-field amplitudes by -frequency; we also take this
  // opportunity to rescale the overall E and H amplitudes to yield unit power flux.
  double scale = -1.0 / frequency, factor = 2.0 / sqrt(fabs(vgrp));
  complex<double> *efield = (complex<double> *)fft_data_E, *hfield = (complex<double> *)(mdata->fft_data);
  for (int n = 0; n < NFFT; ++n) {
    efield[n] *= factor * scale;
    hfield[n] *= factor;
  }

  maxwell_compute_e_from_d(mdata, fft_data_E, 1);

  /*--------------------------------------------------------------*/
  /*- part 4: initialize and return output data structures.       */
  /*--------------------------------------------------------------*/
  eigenmode_data *edata = new eigenmode_data;
  edata->mdata = mdata;
  edata->fft_data_H = mdata->fft_data;
  edata->fft_data_E = fft_data_E;
  edata->H = H;
  edata->n[0] = n[0];
  edata->n[1] = n[1];
  edata->n[2] = n[2];
  edata->s[0] = s[0];
  edata->s[1] = s[1];
  edata->s[2] = s[2];
  edata->Gk[0] = G[0][0] * k[0] + G[1][0] * k[1] + G[2][0] * k[2];
  edata->Gk[1] = G[0][1] * k[0] + G[1][1] * k[1] + G[2][1] * k[2];
  edata->Gk[2] = G[0][2] * k[0] + G[1][2] * k[1] + G[2][2] * k[2];
  edata->center = eig_vol.center();
  edata->amp_func = default_amp_func;
  edata->band_num = band_num;
  edata->frequency = frequency;
  edata->group_velocity = (double)vgrp;

  if (kdom) {
#if MPB_VERSION_MAJOR > 1 || (MPB_VERSION_MAJOR == 1 && MPB_VERSION_MINOR >= 7)
    maxwell_dominant_planewave(mdata, H, band_num, kdom);
    if (verbosity > 0)
      master_printf("Dominant planewave for band %d: (%f,%f,%f)\n", band_num, kdom[0], kdom[1],
                    kdom[2]);
#else
    kdom[0] = kdom[1] = kdom[2] = 0;
#endif
  }

  return (void *)edata;
}

void destroy_eigenmode_data(void *vedata, bool destroy_mdata) {
  adjust_mpb_verbosity amv;
  eigenmode_data *edata = (eigenmode_data *)vedata;
  destroy_evectmatrix(edata->H);
  if (destroy_mdata) destroy_maxwell_data(edata->mdata);
  free(edata->fft_data_E);
  delete edata;
}

double get_group_velocity(void *vedata) {
  eigenmode_data *edata = (eigenmode_data *)vedata;
  return edata->group_velocity;
}

vec get_k(void *vedata) {
  eigenmode_data *edata = (eigenmode_data *)vedata;
  return vec(edata->Gk[0], edata->Gk[1], edata->Gk[2]);
}

/***************************************************************/
/* call get_eigenmode() to solve for the specified eigenmode,  */
/* then call add_volume_source() to add current sources whose  */
/* radiated fields reproduce the eigenmode fields              */
/***************************************************************/
void fields::add_eigenmode_source(component c0, const src_time &src, direction d,
                                  const volume &where, const volume &eig_vol, int band_num,
                                  const vec &kpoint, bool match_frequency, int parity,
                                  double resolution, double eigensolver_tol, complex<double> amp,
                                  complex<double> A(const vec &),
                                  diffractedplanewave *dp) {
  /*--------------------------------------------------------------*/
  /* step 1: call MPB to compute the eigenmode                    */
  /*--------------------------------------------------------------*/
  adjust_mpb_verbosity amv;
  double frequency = real(src.frequency());

  am_now_working_on(MPBTime);
  global_eigenmode_data =
      (eigenmode_data *)get_eigenmode(frequency, d, where, eig_vol, band_num, kpoint,
                                      match_frequency, parity, resolution, eigensolver_tol,
                                      NULL, NULL, dp);
  finished_working();

  /* add_volume_source amp_fun coordinates are relative to where.center();
     this is not the default in get_eigenmode because where-relative coordinates
     are not used elsewhere, e.g. in getting mode coefficients in dft.cpp. */
  global_eigenmode_data->center -= where.center();

  if (!global_eigenmode_data)
    meep::abort("eigenmode solver failed to find the requested mode; you may need to supply a better "
          "guess for k");

  global_eigenmode_data->amp_func = A ? A : default_amp_func;

  src_time *src_mpb = src.clone();
  if (!match_frequency) src_mpb->set_frequency(global_eigenmode_data->frequency);

  /*--------------------------------------------------------------*/
  // step 2: add sources whose radiated field reproduces the      */
  //         the eigenmode                                        */
  //         electric current K = nHat \times H                   */
  //         magnetic current N = -nHat \times E                  */
  /*--------------------------------------------------------------*/
  if (global_eigenmode_data->group_velocity < 0)
    master_printf("vg is negative!\n");
    amp *= -1; // equivalent to flipping the direction of nhat.
  if (is_D(c0)) c0 = direction_component(Ex, component_direction(c0));
  if (is_B(c0)) c0 = direction_component(Hx, component_direction(c0));
  component cE[3] = {Ex, Ey, Ez}, cH[3] = {Hx, Hy, Hz};
  int n = (d == X ? 0 : (d == Y ? 1 : 2));
  if (d == NO_DIRECTION) {
    n = where.in_direction(X) == 0
            ? 0
            : where.in_direction(Y) == 0 ? 1 : where.in_direction(Z) == 0 ? 2 : -1;
    if (n == -1)
      meep::abort(
          "can't determine source direction for non-empty source volume with NO_DIRECTION source");
  }
  int np1 = (n + 1) % 3;
  int np2 = (n + 2) % 3;
  // Kx = -Hy, Ky = Hx   (for d==Z)
  global_eigenmode_component = cH[np1];
  add_volume_source_check(cE[np2], *src_mpb, where, meep_mpb_A, +1.0 * amp, c0, d,
                          parity & ODD_Z_PARITY, parity & EVEN_Z_PARITY);
  global_eigenmode_component = cH[np2];
  add_volume_source_check(cE[np1], *src_mpb, where, meep_mpb_A, -1.0 * amp, c0, d,
                          parity & ODD_Z_PARITY, parity & EVEN_Z_PARITY);
  // Nx = +Ey, Ny = -Ex  (for d==Z)
  global_eigenmode_component = cE[np1];
  add_volume_source_check(cH[np2], *src_mpb, where, meep_mpb_A, -1.0 * amp, c0, d,
                          parity & ODD_Z_PARITY, parity & EVEN_Z_PARITY);
  global_eigenmode_component = cE[np2];
  add_volume_source_check(cH[np1], *src_mpb, where, meep_mpb_A, +1.0 * amp, c0, d,
                          parity & ODD_Z_PARITY, parity & EVEN_Z_PARITY);

  delete src_mpb;
  destroy_eigenmode_data((void *)global_eigenmode_data);
}

/***************************************************************/
/* get eigenmode coefficients for all frequencies in flux      */
/* and all band indices in the caller-populated bands array.   */
/*                                                             */
/* on input, coeffs must point to a user-allocated array of    */
/* length 2*num_freqs*num_bands (where num_freqs=flux.Nfreq).  */
/* on return, the coefficients of the forward/backward traveling*/
/* eigenmodes for frequency #nf and band index bands[nb] are   */
/*  coeffs[ 2*nb*num_freqs + 2*nf + 0/1 ].                     */
/*                                                             */
/* if vgrp is non-null, it should point to a caller-allocated  */
/* array of size num_bands*num_freqs. then on return the group */
/* velocity for the mode with frequency #nf and band index     */
/* bands[nb] is stored in vgrp[nb*num_freqs + nf].             */
/*                                                             */
/* similarly, if kpoints is non-null it should point to a      */
/* caller-allocated array of size num_bands*num_freqs, which on*/
/* return will be populated by the k-vectors for the modes.    */
/***************************************************************/
void fields::get_eigenmode_coefficients(dft_flux flux, const volume &eig_vol, int *bands,
                                        int num_bands, int parity, double eig_resolution,
                                        double eigensolver_tol, std::complex<double> *coeffs,
                                        double *vgrp, kpoint_func user_kpoint_func,
                                        void *user_kpoint_data, vec *kpoints, vec *kdom_list,
                                        double *cscale, direction d, diffractedplanewave *dp) {
  adjust_mpb_verbosity amv;
  int num_freqs = flux.freq.size();
  bool match_frequency = true;
  if (flux.use_symmetry && S.multiplicity() > 1 && parity == 0)
    meep::abort("flux regions for eigenmode projection with symmetry should be created by "
          "add_mode_monitor()");

  vec kpoint(0.0, 0.0, 0.0); // default guess

  // get_eigenmode will create mdata only once and then reuse it on each iteration of the loop
  maxwell_data *mdata = NULL;

  // loop over all bands and all frequencies
  for (int nb = 0; nb < num_bands; nb++) {
    for (int nf = 0; nf < num_freqs; nf++) {
      /*--------------------------------------------------------------*/
      /*- call mpb to compute the eigenmode --------------------------*/
      /*--------------------------------------------------------------*/
      int band_num = bands ? bands[nb] : 1;
      double kdom[3];
      if (user_kpoint_func) kpoint = user_kpoint_func(flux.freq[nf], band_num, user_kpoint_data);
      am_now_working_on(MPBTime);
      void *mode_data =
          get_eigenmode(flux.freq[nf], d, flux.where, eig_vol, band_num, kpoint, match_frequency,
                        parity, eig_resolution, eigensolver_tol, kdom, (void **)&mdata, dp);
      finished_working();
      if (!mode_data) { // mode not found, assume evanescent
        coeffs[2 * nb * num_freqs + 2 * nf] = coeffs[2 * nb * num_freqs + 2 * nf + 1] = 0;
        if (vgrp) vgrp[nb * num_freqs + nf] = 0;
        if (kpoints) kpoints[nb * num_freqs + nf] = vec(0.0, 0.0, 0.0);
        if (kdom_list) kdom_list[nb * num_freqs + nf] = vec(0.0, 0.0, 0.0);
        continue;
      }

      double vg = get_group_velocity(mode_data);
      vec kfound = get_k(mode_data);
      if (vgrp) vgrp[nb * num_freqs + nf] = vg;
      if (kpoints) kpoints[nb * num_freqs + nf] = kfound;
      if (kdom_list) kdom_list[nb * num_freqs + nf] = vec(kdom[0], kdom[1], kdom[2]);

      /* in the common case of k aligned along a cartesian direction, update
         our k-point guess based on the current k point plus a correction from the group velocity */
      if (match_frequency && nf + 1 < num_freqs && abs(kfound) > 0 &&
          ((kfound.x() == 0 && (kfound.y() == 0 || kfound.z() == 0)) ||
           (kfound.y() == 0 && kfound.z() == 0)))
        kpoint = kfound + kfound * ((flux.freq[nf + 1] - flux.freq[nf]) / (vg * abs(kfound)));

      /*--------------------------------------------------------------*/
      /*--------------------------------------------------------------*/
      /*--------------------------------------------------------------*/
      complex<double> mode_flux[2], mode_mode[2];
      get_mode_flux_overlap(mode_data, flux, nf, mode_flux);
      get_mode_mode_overlap(mode_data, mode_data, flux, mode_mode);
      complex<double> cplus = 0.5 * (mode_flux[0] + mode_flux[1]);
      complex<double> cminus = 0.5 * (mode_flux[0] - mode_flux[1]);
      /* MPB modes are normalized to unit power above, but we need to re-normalize here to have
         unit power as integrated on Meep's Yee grid and not on MPB's grid.  Thus, normfac differs
         from a constant factor only because of discretization effects. */
      complex<double> normfac = 0.5 * (mode_mode[0] + mode_mode[1]);
      if (normfac == 0.0) normfac = 1.0;
      double csc = sqrt((flux.use_symmetry ? S.multiplicity() : 1.0) / abs(normfac));
      if (cscale)
        cscale[nb * num_freqs + nf] =
            csc; // return real part of coefficient scalar for adjoint calculations
      coeffs[2 * nb * num_freqs + 2 * nf + (vg > 0.0 ? 0 : 1)] = cplus * csc;
      coeffs[2 * nb * num_freqs + 2 * nf + (vg > 0.0 ? 1 : 0)] = cminus * csc;
      destroy_eigenmode_data((void *)mode_data, false);
    }
  }
  destroy_maxwell_data(mdata);
}

/**************************************************************/
/* dummy versions of class methods for compiling without MPB  */
/**************************************************************/
#else // #ifdef HAVE_MPB
void *fields::get_eigenmode(double frequency, direction d, const volume where, const volume eig_vol,
                            int band_num, const vec &kpoint, bool match_frequency, int parity,
                            double resolution, double eigensolver_tol, double *kdom,
                            void **user_mdata, diffractedplanewave *dp) {

  (void)frequency;
  (void)d;
  (void)where;
  (void)eig_vol;
  (void)band_num;
  (void)kpoint;
  (void)match_frequency;
  (void)parity;
  (void)resolution;
  (void)eigensolver_tol;
  (void)kdom;
  (void)user_mdata;
  (void)dp;
  meep::abort("Meep must be configured/compiled with MPB for get_eigenmode");
}

void fields::add_eigenmode_source(component c0, const src_time &src, direction d,
                                  const volume &where, const volume &eig_vol, int band_num,
                                  const vec &kpoint, bool match_frequency, int parity,
                                  double resolution, double eigensolver_tol, complex<double> amp,
                                  complex<double> A(const vec &), diffractedplanewave *dp) {
  (void)c0;
  (void)src;
  (void)d;
  (void)where;
  (void)eig_vol;
  (void)band_num;
  (void)kpoint;
  (void)match_frequency;
  (void)parity;
  (void)resolution;
  (void)eigensolver_tol;
  (void)amp;
  (void)A;
  (void)dp;
  meep::abort("Meep must be configured/compiled with MPB for add_eigenmode_source");
}

void fields::get_eigenmode_coefficients(dft_flux flux, const volume &eig_vol, int *bands,
                                        int num_bands, int parity, double eig_resolution,
                                        double eigensolver_tol, std::complex<double> *coeffs,
                                        double *vgrp, kpoint_func user_kpoint_func,
                                        void *user_kpoint_data, vec *kpoints, vec *kdom,
                                        double *cscale, direction d, diffractedplanewave *dp) {
  (void)flux;
  (void)eig_vol;
  (void)bands;
  (void)num_bands;
  (void)parity;
  (void)eig_resolution;
  (void)eigensolver_tol;
  (void)coeffs;
  (void)vgrp;
  (void)kpoints;
  (void)user_kpoint_func;
  (void)user_kpoint_data;
  (void)kdom;
  (void)cscale;
  (void)d;
  (void)dp;
  meep::abort("Meep must be configured/compiled with MPB for get_eigenmode_coefficients");
}

void destroy_eigenmode_data(void *vedata, bool destroy_mdata) {
  (void)vedata;
  (void)destroy_mdata;
}

std::complex<double> eigenmode_amplitude(void *vedata, const vec &p, component c) {
  (void)vedata;
  (void)p;
  (void)c;
  return 0.0;
}

double get_group_velocity(void *vedata) {
  (void)vedata;
  return 0.0;
}

vec get_k(void *vedata) {
  (void)vedata;
  return vec(0.0, 0.0, 0.0);
}

#endif // HAVE_MPB

/* compatibility wrapper routine that passes the default flux.normal_direction to the eigensolver
   (we pass NO_DIRECTION to use the kpoint direction instead, for oblique sources). */
void fields::get_eigenmode_coefficients(dft_flux flux, const volume &eig_vol, int *bands,
                                        int num_bands, int parity, double eig_resolution,
                                        double eigensolver_tol, std::complex<double> *coeffs,
                                        double *vgrp, kpoint_func user_kpoint_func,
                                        void *user_kpoint_data, vec *kpoints, vec *kdom,
                                        double *cscale, diffractedplanewave *dp) {
  get_eigenmode_coefficients(flux, eig_vol, bands, num_bands, parity, eig_resolution,
                             eigensolver_tol, coeffs, vgrp, user_kpoint_func, user_kpoint_data,
                             kpoints, kdom, cscale, flux.normal_direction, dp);
}

} // namespace meep
