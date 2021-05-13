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

/* This file implements dispersive materials for Meep via a
   polarization P = \chi(\omega) W, where W is e.g. E or H.  Each
   subclass of the susceptibility class should implement a different
   type of \chi(\omega).  The subclass knows how to timestep P given W
   at the current (and possibly previous) timestep, and any additional
   internal data that needs to be allocated along with P.

   Each \chi(\omega) is spatially multiplied by a (scalar) sigma
   array.  The meep::fields class is responsible for allocating P and
   sigma and passing them to susceptibility::update_P. */

#include <stdlib.h>
#include <string.h>
#include "meep.hpp"
#include "meep_internals.hpp"

using namespace std;

namespace meep {

int susceptibility::cur_id = 0;

template <class T>
susceptibility *susceptibility::clone() const {
  susceptibility *sus = new susceptibility(*this);
  sus->next = 0;
  sus->ntot = ntot;
  sus->id = id;
  FOR_COMPONENTS(c) FOR_DIRECTIONS(d) {
    if (sigma[c][d]) {
      sus->sigma[c][d] = new T[ntot];
      memcpy(sus->sigma[c][d], sigma[c][d], sizeof(T) * ntot);
    }
    else
      sus->sigma[c][d] = NULL;
    sus->trivial_sigma[c][d] = trivial_sigma[c][d];
  }
  return sus;
}

// generic base class definition.
template <class T>
std::complex<T> susceptibility::chi1(T freq, T sigma) {
  (void)freq;
  (void)sigma;
  return std::complex<T>(0, 0);
}

void susceptibility::delete_internal_data(void *data) const { free(data); }

/* Return whether or not we need to allocate P[c][cmp].  (We don't need to
   allocate P[c] if we can be sure it will be zero.)

   We are a bit wasteful because if sigma is nontrivial in *any* chunk,
   we allocate the corresponding P on *every* owned chunk.  This greatly
   simplifies communication in boundaries.cpp, because we can be sure that
   one chunk has a P then any chunk it borders has the same P, so we don't
   have to worry about communicating with something that doesn't exist.
   TODO: reduce memory usage (bookkeeping seem much harder, though).
*/
template <class T>
bool susceptibility::needs_P(component c, int cmp, T *W[NUM_FIELD_COMPONENTS][2]) const {
  if (!is_electric(c) && !is_magnetic(c)) return false;
  FOR_DIRECTIONS(d) {
    if (!trivial_sigma[c][d] && W[direction_component(c, d)][cmp]) return true;
  }
  return false;
}

/* return whether we need the notowned parts of the W field --
   by default, this is only the case if sigma has offdiagonal components
   coupling P to W.   (See needs_P: again, this is true if the notowned
   W is needed in *any* chunk.) */
template <class T>
bool susceptibility::needs_W_notowned(component c, T *W[NUM_FIELD_COMPONENTS][2]) const {
  FOR_DIRECTIONS(d) {
    if (d != component_direction(c)) {
      component cP = direction_component(c, d);
      if (needs_P(cP, 0, W) && !trivial_sigma[cP][component_direction(c)]) return true;
    }
  }
  return false;
}

template <class T>
typedef struct {
  size_t sz_data;
  size_t ntot;
  T *P[NUM_FIELD_COMPONENTS][2];
  T *P_prev[NUM_FIELD_COMPONENTS][2];
  T data[1];
} lorentzian_data;

// for Lorentzian susc. the internal data is just a backup of P from
// the previous timestep.
template <class T>
void *lorentzian_susceptibility::new_internal_data(T *W[NUM_FIELD_COMPONENTS][2],
                                                   const grid_volume &gv) const {
  int num = 0;
  FOR_COMPONENTS(c) DOCMP2 {
    if (needs_P(c, cmp, W)) num += 2 * gv.ntot();
  }
  size_t sz = sizeof(lorentzian_data<T>) + sizeof(T) * (num - 1);
  lorentzian_data<T> *d = (lorentzian_data<T> *)malloc(sz);
  if (d == NULL) abort("%s:%i:out of memory(%lu)", __FILE__, __LINE__, sz);
  d->sz_data = sz;
  return (void *)d;
}

template <class T>
void lorentzian_susceptibility::init_internal_data(T *W[NUM_FIELD_COMPONENTS][2], T dt,
                                                   const grid_volume &gv, void *data) const {
  (void)dt; // unused
  lorentzian_data<T> *d = (lorentzian_data<T> *)data;
  size_t sz_data = d->sz_data;
  memset(d, 0, sz_data);
  d->sz_data = sz_data;
  size_t ntot = d->ntot = gv.ntot();
  T *P = d->data;
  T *P_prev = d->data + ntot;
  FOR_COMPONENTS(c) DOCMP2 {
    if (needs_P(c, cmp, W)) {
      d->P[c][cmp] = P;
      d->P_prev[c][cmp] = P_prev;
      P += 2 * ntot;
      P_prev += 2 * ntot;
    }
  }
}

template <class T>
void *lorentzian_susceptibility::copy_internal_data(void *data) const {
  lorentzian_data<T> *d = (lorentzian_data<T> *)data;
  if (!d) return 0;
  lorentzian_data<T> *dnew = (lorentzian_data<T> *)malloc(d->sz_data);
  memcpy(dnew, d, d->sz_data);
  size_t ntot = d->ntot;
  T *P = dnew->data;
  T *P_prev = dnew->data + ntot;
  FOR_COMPONENTS(c) DOCMP2 {
    if (d->P[c][cmp]) {
      dnew->P[c][cmp] = P;
      dnew->P_prev[c][cmp] = P_prev;
      P += 2 * ntot;
      P_prev += 2 * ntot;
    }
  }
  return (void *)dnew;
}

#if 0
/* Return true if the discretized Lorentzian ODE is intrinsically unstable,
   i.e. if it corresponds to a filter with a pole z outside the unit circle.
   Note that the pole satisfies the quadratic equation:
            (z + 1/z - 2)/dt^2 + g*(z - 1/z)/(2*dt) + w^2 = 0
   where w = 2*pi*omega_0 and g = 2*pi*gamma.   It is just a little
   algebra from this to get the condition for a root with |z| > 1.

   FIXME: this test seems to be too conservative (issue #12) */
template <class T>
static bool lorentzian_unstable(T omega_0, T gamma, T dt) {
  T w = 2 * pi * omega_0, g = 2 * pi * gamma;
  T g2 = g * dt / 2, w2 = (w * dt) * (w * dt);
  T b = (1 - w2 / 2) / (1 + g2), c = (1 - g2) / (1 + g2);
  return b * b > c && 2 * b * b - c + 2 * fabs(b) * sqrt(b * b - c) > 1;
}
#endif

#define SWAP(t, a, b)                                                                              \
  {                                                                                                \
    t SWAP_temp = a;                                                                               \
    a = b;                                                                                         \
    b = SWAP_temp;                                                                                 \
  }

// stable averaging of offdiagonal components
#define OFFDIAG(u, g, sx, s)                                                                       \
  (0.25 * ((g[i] + g[i - sx]) * u[i] + (g[i + s] + g[(i + s) - sx]) * u[i + s]))

template <class T>
void lorentzian_susceptibility::update_P(T *W[NUM_FIELD_COMPONENTS][2],
                                         T *W_prev[NUM_FIELD_COMPONENTS][2], T dt,
                                         const grid_volume &gv, void *P_internal_data) const {
  lorentzian_data<T> *d = (lorentzian_data<T> *)P_internal_data;
  const T omega2pi = 2 * pi * omega_0, g2pi = gamma * 2 * pi;
  const T omega0dtsqr = omega2pi * omega2pi * dt * dt;
  const T gamma1inv = 1 / (1 + g2pi * dt / 2), gamma1 = (1 - g2pi * dt / 2);
  const T omega0dtsqr_denom = no_omega_0_denominator ? 0 : omega0dtsqr;
  (void)W_prev; // unused;

  // TODO: add back lorentzian_unstable(omega_0, gamma, dt) if we can improve the stability test

  FOR_COMPONENTS(c) DOCMP2 {
    if (d->P[c][cmp]) {
      const T *w = W[c][cmp], *s = sigma[c][component_direction(c)];
      if (w && s) {
        T *p = d->P[c][cmp], *pp = d->P_prev[c][cmp];

        // directions/strides for offdiagonal terms, similar to update_eh
        const direction d = component_direction(c);
        const ptrdiff_t is = gv.stride(d) * (is_magnetic(c) ? -1 : +1);
        direction d1 = cycle_direction(gv.dim, d, 1);
        component c1 = direction_component(c, d1);
        ptrdiff_t is1 = gv.stride(d1) * (is_magnetic(c) ? -1 : +1);
        const T *w1 = W[c1][cmp];
        const T *s1 = w1 ? sigma[c][d1] : NULL;
        direction d2 = cycle_direction(gv.dim, d, 2);
        component c2 = direction_component(c, d2);
        ptrdiff_t is2 = gv.stride(d2) * (is_magnetic(c) ? -1 : +1);
        const T *w2 = W[c2][cmp];
        const T *s2 = w2 ? sigma[c][d2] : NULL;

        if (s2 && !s1) { // make s1 the non-NULL one if possible
          SWAP(direction, d1, d2);
          SWAP(component, c1, c2);
          SWAP(ptrdiff_t, is1, is2);
          SWAP(const T *, w1, w2);
          SWAP(const T *, s1, s2);
        }
        if (s1 && s2) { // 3x3 anisotropic
          LOOP_OVER_VOL_OWNED(gv, c, i) {
            // s[i] != 0 check is a bit of a hack to work around
            // some instabilities that occur near the boundaries
            // of materials; see PR #666
            if (s[i] != 0) {
              T pcur = p[i];
              p[i] = gamma1inv * (pcur * (2 - omega0dtsqr_denom) - gamma1 * pp[i] +
                                  omega0dtsqr * (s[i] * w[i] + OFFDIAG(s1, w1, is1, is) +
                                                 OFFDIAG(s2, w2, is2, is)));
              pp[i] = pcur;
            }
          }
        }
        else if (s1) { // 2x2 anisotropic
          LOOP_OVER_VOL_OWNED(gv, c, i) {
            if (s[i] != 0) { // see above
              T pcur = p[i];
              p[i] = gamma1inv * (pcur * (2 - omega0dtsqr_denom) - gamma1 * pp[i] +
                                  omega0dtsqr * (s[i] * w[i] + OFFDIAG(s1, w1, is1, is)));
              pp[i] = pcur;
            }
          }
        }
        else { // isotropic
          LOOP_OVER_VOL_OWNED(gv, c, i) {
            T pcur = p[i];
            p[i] = gamma1inv *
                   (pcur * (2 - omega0dtsqr_denom) - gamma1 * pp[i] + omega0dtsqr * (s[i] * w[i]));
            pp[i] = pcur;
          }
        }
      }
    }
  }
}

template <class T>
void lorentzian_susceptibility::subtract_P(field_type ft,
                                           T *f_minus_p[NUM_FIELD_COMPONENTS][2],
                                           void *P_internal_data) const {
  lorentzian_data<T> *d = (lorentzian_data<T> *)P_internal_data;
  field_type ft2 = ft == E_stuff ? D_stuff : B_stuff; // for sources etc.
  size_t ntot = d->ntot;
  FOR_FT_COMPONENTS(ft, ec) DOCMP2 {
    if (d->P[ec][cmp]) {
      component dc = field_type_component(ft2, ec);
      if (f_minus_p[dc][cmp]) {
        T *p = d->P[ec][cmp];
        T *fmp = f_minus_p[dc][cmp];
        for (size_t i = 0; i < ntot; ++i)
          fmp[i] -= p[i];
      }
    }
  }
}

template <class T>
int lorentzian_susceptibility::num_cinternal_notowned_needed(component c,
                                                             void *P_internal_data) const {
  lorentzian_data<T> *d = (lorentzian_data<T> *)P_internal_data;
  return d->P[c][0] ? 1 : 0;
}

template <class T>
T *lorentzian_susceptibility::cinternal_notowned_ptr(int inotowned, component c, int cmp,
                                                     int n, void *P_internal_data) const {
  lorentzian_data<T> *d = (lorentzian_data<T> *)P_internal_data;
  (void)inotowned; // always = 0
  if (!d || !d->P[c][cmp]) return NULL;
  return d->P[c][cmp] + n;
}

template <class T>
std::complex<T> lorentzian_susceptibility::chi1(T freq, T sigma) {
  if (no_omega_0_denominator) {
    // Drude model
    return sigma * omega_0 * omega_0 / std::complex<T>(-freq * freq, -gamma * freq);
  }
  else {
    // Standard Lorentzian model
    return sigma * omega_0 * omega_0 /
           std::complex<T>(omega_0 * omega_0 - freq * freq, -gamma * freq);
  }
}

template <class T>
void lorentzian_susceptibility::dump_params(h5file *h5f, size_t *start) {
  size_t num_params = 5;
  size_t params_dims[1] = {num_params};
  T params_data[] = {4, (T)get_id(), omega_0, gamma, (T)no_omega_0_denominator};
  h5f->write_chunk(1, start, params_dims, params_data);
  *start += num_params;
}

template <class T>
void noisy_lorentzian_susceptibility::update_P(T *W[NUM_FIELD_COMPONENTS][2],
                                               T *W_prev[NUM_FIELD_COMPONENTS][2], T dt,
                                               const grid_volume &gv, void *P_internal_data) const {
  lorentzian_susceptibility::update_P(W, W_prev, dt, gv, P_internal_data);
  lorentzian_data<T> *d = (lorentzian_data<T> *)P_internal_data;

  const T g2pi = gamma * 2 * pi;
  const T w2pi = omega_0 * 2 * pi;
  const T amp = w2pi * noise_amp * sqrt(g2pi) * dt * dt / (1 + g2pi * dt / 2);
  /* for uniform random numbers in [-amp,amp] below, multiply amp by sqrt(3) */

  FOR_COMPONENTS(c) DOCMP2 {
    if (d->P[c][cmp]) {
      const T *s = sigma[c][component_direction(c)];
      if (s) {
        T *p = d->P[c][cmp];
        LOOP_OVER_VOL_OWNED(gv, c, i) { p[i] += gaussian_random(0, amp * sqrt(s[i])); }
        // for uniform random numbers, use uniform_random(-1,1) * amp * sqrt(s[i])
        // for gaussian random numbers, use gaussian_random(0, amp * sqrt(s[i]))
      }
    }
  }
}

template <class T>
void noisy_lorentzian_susceptibility::dump_params(h5file *h5f, size_t *start) {
  size_t num_params = 6;
  size_t params_dims[1] = {num_params};
  T params_data[] = {
      5, (T)get_id(), noise_amp, omega_0, gamma, (T)no_omega_0_denominator};
  h5f->write_chunk(1, start, params_dims, params_data);
  *start += num_params;
}

template <class T>
gyrotropic_susceptibility::gyrotropic_susceptibility(const vec &bias, T omega_0, T gamma,
                                                     T alpha, gyrotropy_model model)
    : omega_0(omega_0), gamma(gamma), alpha(alpha), model(model) {
  // Precalculate g_{ij} = sum_k epsilon_{ijk} b_k, used in update_P.
  // Ignore |b| for Landau-Lifshitz-Gilbert gyrotropy model.
  const vec b = (model == GYROTROPIC_SATURATED) ? bias / abs(bias) : bias;
  memset(gyro_tensor, 0, 9 * sizeof(T));
  gyro_tensor[X][Y] = b.z();
  gyro_tensor[Y][X] = -b.z();
  gyro_tensor[Y][Z] = b.x();
  gyro_tensor[Z][Y] = -b.x();
  gyro_tensor[Z][X] = b.y();
  gyro_tensor[X][Z] = -b.y();
}

/* To implement gyrotropic susceptibilities, we track three
   polarization components (e.g. Px, Py, Pz) on EACH of the Yee cell's
   three driving field positions (e.g., Ex, Ey, and Ez), i.e. 9
   numbers per cell.  This takes 3X the memory and runtime compared to
   Lorentzian susceptibility.  The advantage is that during update_P,
   we can directly access the value of P at each update point without
   averaging.  */
template <class T>
typedef struct {
  size_t sz_data;
  size_t ntot;
  T *P[NUM_FIELD_COMPONENTS][2][3];
  T *P_prev[NUM_FIELD_COMPONENTS][2][3];
  T data[1];
} gyrotropy_data;

template <class T>
void *gyrotropic_susceptibility::new_internal_data(T *W[NUM_FIELD_COMPONENTS][2],
                                                   const grid_volume &gv) const {
  int num = 0;
  FOR_COMPONENTS(c) DOCMP2 {
    if (needs_P(c, cmp, W)) num += 6 * gv.ntot();
  }
  size_t sz = sizeof(gyrotropy_data<T>) + sizeof(T) * (num - 1);
  gyrotropy_data<T> *d = (gyrotropy_data<T> *)malloc(sz);
  if (d == NULL) abort("%s:%i:out of memory(%lu)", __FILE__, __LINE__, sz);
  d->sz_data = sz;
  return (void *)d;
}

template <class T>
void gyrotropic_susceptibility::init_internal_data(T *W[NUM_FIELD_COMPONENTS][2], T dt,
                                                   const grid_volume &gv, void *data) const {
  (void)dt; // unused
  gyrotropy_data<T> *d = (gyrotropy_data<T> *)data;
  size_t sz_data = d->sz_data;
  memset(d, 0, sz_data);
  d->sz_data = sz_data;
  d->ntot = gv.ntot();
  T *p = d->data;
  FOR_COMPONENTS(c) DOCMP2 {
    if (needs_P(c, cmp, W)) {
      for (int dd = X; dd < R; dd++) {
        d->P[c][cmp][dd] = p;
        p += d->ntot;
        d->P_prev[c][cmp][dd] = p;
        p += d->ntot;
      }
    }
  }
}

void *gyrotropic_susceptibility::copy_internal_data(void *data) const {
  gyrotropy_data<T> *d = (gyrotropy_data<T> *)data;
  if (!d) return 0;
  gyrotropy_data<T> *dnew = (gyrotropy_data<T> *)malloc(d->sz_data);
  memcpy(dnew, d, d->sz_data);
  T *p = dnew->data;
  FOR_COMPONENTS(c) DOCMP2 {
    if (d->P[c][cmp][0]) {
      for (int dd = X; dd < R; dd++) {
        dnew->P[c][cmp][dd] = p;
        p += d->ntot;
        dnew->P_prev[c][cmp][dd] = p;
        p += d->ntot;
      }
    }
  }
  return (void *)dnew;
}

template <class T>
bool gyrotropic_susceptibility::needs_P(component c, int cmp,
                                        T *W[NUM_FIELD_COMPONENTS][2]) const {
  if (!is_electric(c) && !is_magnetic(c)) return false;
  direction d0 = component_direction(c);
  return (d0 == X || d0 == Y || d0 == Z) && sigma[c][d0] && W[c][cmp];
}

// Similar to the OFFDIAG macro, but without averaging sigma.
#define OFFDIAGW(g, sx, s) (0.25 * (g[i] + g[i - sx] + g[i + s] + g[i + s - sx]))

template <class T>
void gyrotropic_susceptibility::update_P(T *W[NUM_FIELD_COMPONENTS][2],
                                         T *W_prev[NUM_FIELD_COMPONENTS][2], T dt,
                                         const grid_volume &gv, void *P_internal_data) const {
  gyrotropy_data<T> *d = (gyrotropy_data<T> *)P_internal_data;
  const T omega2pidt = 2 * pi * omega_0 * dt;
  const T g2pidt = 2 * pi * gamma * dt;
  (void)W_prev; // unused;

  switch (model) {
    case GYROTROPIC_LORENTZIAN:
    case GYROTROPIC_DRUDE: {
      const T omega0dtsqr = omega2pidt * omega2pidt;
      const T gamma1 = (1 - g2pidt / 2);
      const T diag = 2 - (model == GYROTROPIC_DRUDE ? 0 : omega0dtsqr);
      const T pt = pi * dt;

      // Precalculate 3x3 matrix inverse, exploiting skew symmetry
      const T gd = (1 + g2pidt / 2);
      const T gx = pt * gyro_tensor[Y][Z];
      const T gy = pt * gyro_tensor[Z][X];
      const T gz = pt * gyro_tensor[X][Y];
      const T invdet = 1.0 / gd / (gd * gd + gx * gx + gy * gy + gz * gz);
      const T inv[3][3] = {{invdet * (gd * gd + gx * gx), invdet * (gx * gy + gd * gz),
                                  invdet * (gx * gz - gd * gy)},
                                 {invdet * (gy * gx - gd * gz), invdet * (gd * gd + gy * gy),
                                  invdet * (gy * gz + gd * gx)},
                                 {invdet * (gz * gx + gd * gy), invdet * (gz * gy - gd * gx),
                                  invdet * (gd * gd + gz * gz)}};

      FOR_COMPONENTS(c) DOCMP2 {
        if (d->P[c][cmp][0]) {
          const direction d0 = component_direction(c);
          const T *w0 = W[c][cmp], *s = sigma[c][d0];

          if (!w0 || !s || (d0 != X && d0 != Y && d0 != Z))
            abort("gyrotropic media require 3D Cartesian fields\n");

          const direction d1 = cycle_direction(gv.dim, d0, 1);
          const direction d2 = cycle_direction(gv.dim, d0, 2);
          const T *w1 = W[direction_component(c, d1)][cmp];
          const T *w2 = W[direction_component(c, d2)][cmp];
          T *p0 = d->P[c][cmp][d0], *pp0 = d->P_prev[c][cmp][d0];
          T *p1 = d->P[c][cmp][d1], *pp1 = d->P_prev[c][cmp][d1];
          T *p2 = d->P[c][cmp][d2], *pp2 = d->P_prev[c][cmp][d2];
          const ptrdiff_t is = gv.stride(d0) * (is_magnetic(c) ? -1 : +1);
          const ptrdiff_t is1 = gv.stride(d1) * (is_magnetic(c) ? -1 : +1);
          const ptrdiff_t is2 = gv.stride(d2) * (is_magnetic(c) ? -1 : +1);
          T r0, r1, r2;

          if (!pp1 || !pp2) abort("gyrotropic media require 3D Cartesian fields\n");
          if (sigma[c][d1] || sigma[c][d2])
            abort("gyrotropic media do not support anisotropic sigma\n");

          LOOP_OVER_VOL_OWNED(gv, c, i) {
            r0 = diag * p0[i] - gamma1 * pp0[i] + omega0dtsqr * s[i] * w0[i] -
                 pt * gyro_tensor[d0][d1] * pp1[i] - pt * gyro_tensor[d0][d2] * pp2[i];
            r1 = diag * p1[i] - gamma1 * pp1[i] +
                 (w1 ? omega0dtsqr * s[i] * OFFDIAGW(w1, is1, is) : 0) -
                 pt * gyro_tensor[d1][d0] * pp0[i] - pt * gyro_tensor[d1][d2] * pp2[i];
            r2 = diag * p2[i] - gamma1 * pp2[i] +
                 (w2 ? omega0dtsqr * s[i] * OFFDIAGW(w2, is2, is) : 0) -
                 pt * gyro_tensor[d2][d1] * pp1[i] - pt * gyro_tensor[d2][d0] * pp0[i];

            pp0[i] = p0[i];
            pp1[i] = p1[i];
            pp2[i] = p2[i];
            p0[i] = inv[d0][d0] * r0 + inv[d0][d1] * r1 + inv[d0][d2] * r2;
            p1[i] = inv[d1][d0] * r0 + inv[d1][d1] * r1 + inv[d1][d2] * r2;
            p2[i] = inv[d2][d0] * r0 + inv[d2][d1] * r1 + inv[d2][d2] * r2;
          }
        }
      }
    } break;

    case GYROTROPIC_SATURATED: {
      const T dt2pi = 2 * pi * dt;

      // Precalculate 3x3 matrix inverse, exploiting skew symmetry
      const T gd = 0.5;
      const T gx = -0.5 * alpha * gyro_tensor[Y][Z];
      const T gy = -0.5 * alpha * gyro_tensor[Z][X];
      const T gz = -0.5 * alpha * gyro_tensor[X][Y];
      const T invdet = 1.0 / gd / (gd * gd + gx * gx + gy * gy + gz * gz);
      const T inv[3][3] = {{invdet * (gd * gd + gx * gx), invdet * (gx * gy + gd * gz),
                                  invdet * (gx * gz - gd * gy)},
                                 {invdet * (gy * gx - gd * gz), invdet * (gd * gd + gy * gy),
                                  invdet * (gy * gz + gd * gx)},
                                 {invdet * (gz * gx + gd * gy), invdet * (gz * gy - gd * gx),
                                  invdet * (gd * gd + gz * gz)}};

      FOR_COMPONENTS(c) DOCMP2 {
        if (d->P[c][cmp][0]) {
          const direction d0 = component_direction(c);
          const T *w0 = W[c][cmp], *s = sigma[c][d0];

          if (!w0 || !s || (d0 != X && d0 != Y && d0 != Z))
            abort("gyrotropic media require 3D Cartesian fields\n");

          const direction d1 = cycle_direction(gv.dim, d0, 1);
          const direction d2 = cycle_direction(gv.dim, d0, 2);
          const T *w1 = W[direction_component(c, d1)][cmp];
          const T *w2 = W[direction_component(c, d2)][cmp];
          T *p0 = d->P[c][cmp][d0], *pp0 = d->P_prev[c][cmp][d0];
          T *p1 = d->P[c][cmp][d1], *pp1 = d->P_prev[c][cmp][d1];
          T *p2 = d->P[c][cmp][d2], *pp2 = d->P_prev[c][cmp][d2];
          const ptrdiff_t is = gv.stride(d0) * (is_magnetic(c) ? -1 : +1);
          const ptrdiff_t is1 = gv.stride(d1) * (is_magnetic(c) ? -1 : +1);
          const ptrdiff_t is2 = gv.stride(d2) * (is_magnetic(c) ? -1 : +1);
          T r0, r1, r2, q0, q1, q2;

          if (!pp1 || !pp2) abort("gyrotropic media require 3D Cartesian fields\n");
          if (sigma[c][d1] || sigma[c][d2])
            abort("gyrotropic media do not support anisotropic sigma\n");

          LOOP_OVER_VOL_OWNED(gv, c, i) {
            q0 = -omega2pidt * p0[i] + 0.5 * alpha * pp0[i] + dt2pi * s[i] * w0[i];
            q1 = -omega2pidt * p1[i] + 0.5 * alpha * pp1[i] +
                 dt2pi * s[i] * (w1 ? OFFDIAGW(w1, is1, is) : 0);
            q2 = -omega2pidt * p2[i] + 0.5 * alpha * pp2[i] +
                 dt2pi * s[i] * (w2 ? OFFDIAGW(w2, is2, is) : 0);

            r0 =
                0.5 * pp0[i] - g2pidt * p0[i] + gyro_tensor[d0][d1] * q1 + gyro_tensor[d0][d2] * q2;
            r1 =
                0.5 * pp1[i] - g2pidt * p1[i] + gyro_tensor[d1][d2] * q2 + gyro_tensor[d1][d0] * q0;
            r2 =
                0.5 * pp2[i] - g2pidt * p2[i] + gyro_tensor[d2][d0] * q0 + gyro_tensor[d2][d1] * q1;

            pp0[i] = p0[i];
            pp1[i] = p1[i];
            pp2[i] = p2[i];
            p0[i] = inv[d0][d0] * r0 + inv[d0][d1] * r1 + inv[d0][d2] * r2;
            p1[i] = inv[d1][d0] * r0 + inv[d1][d1] * r1 + inv[d1][d2] * r2;
            p2[i] = inv[d2][d0] * r0 + inv[d2][d1] * r1 + inv[d2][d2] * r2;
          }
        }
      }
    } break;
  }
}

template <class T>
void gyrotropic_susceptibility::subtract_P(field_type ft,
                                           T *f_minus_p[NUM_FIELD_COMPONENTS][2],
                                           void *P_internal_data) const {
  gyrotropy_data<T> *d = (gyrotropy_data<T> *)P_internal_data;
  field_type ft2 = ft == E_stuff ? D_stuff : B_stuff; // for sources etc.
  size_t ntot = d->ntot;
  FOR_FT_COMPONENTS(ft, ec) DOCMP2 {
    if (d->P[ec][cmp][0]) {
      component dc = field_type_component(ft2, ec);
      if (f_minus_p[dc][cmp]) {
        T *p = d->P[ec][cmp][component_direction(ec)];
        T *fmp = f_minus_p[dc][cmp];
        for (size_t i = 0; i < ntot; ++i)
          fmp[i] -= p[i];
      }
    }
  }
}

int gyrotropic_susceptibility::num_cinternal_notowned_needed(component c,
                                                             void *P_internal_data) const {
  (void)c;
  (void)P_internal_data;
  return 0;
}

template <class T>
T *gyrotropic_susceptibility::cinternal_notowned_ptr(int inotowned, component c, int cmp,
                                                           int n, void *P_internal_data) const {
  gyrotropy_data<T> *d = (gyrotropy_data<T> *)P_internal_data;
  if (!d || !d->P[c][cmp][inotowned]) return NULL;
  return d->P[c][cmp][inotowned] + n;
}

template <class T>
void gyrotropic_susceptibility::dump_params(h5file *h5f, size_t *start) {
  size_t num_params = 9;
  size_t params_dims[1] = {num_params};
  T bias[] = {gyro_tensor[Y][Z], gyro_tensor[Z][X], gyro_tensor[X][Y]};
  T params_data[] = {8, (T)get_id(), bias[X], bias[Y], bias[Z], omega_0, gamma,
                          alpha, (T)model};
  h5f->write_chunk(1, start, params_dims, params_data);
  *start += num_params;
}

} // namespace meep
