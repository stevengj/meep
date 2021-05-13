/* Copyright (C) 2005-2021 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
%
%  This program is distributed in the hope that it will be useful,
%  but WITHOUT ANY WARRANTY; without even the implied warranty of
%  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%  GNU General Public License for more details.
%
%  You should have received a copy of the GNU General Public License
%  along with this program; if not, write to the Free Software Foundation,
%  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "meep.hpp"
#include "meep_internals.hpp"

#include "config.h"
#if defined(HAVE_LIBFFTW3)
#include <fftw3.h>
#elif defined(HAVE_LIBDFFTW)
#include <dfftw.h>
#elif defined(HAVE_LIBFFTW)
#include <fftw.h>
#endif
#if defined(HAVE_LIBFFTW3) || defined(HAVE_LIBFFTW) || defined(HAVE_LIBDFFTW)
#define HAVE_SOME_FFTW 1
#else
#define HAVE_SOME_FFTW 0
#endif

/* Below are the monitor point routines. */

using namespace std;

namespace meep {

monitor_point::monitor_point() { next = NULL; }

monitor_point::~monitor_point() {
  if (next) delete next;
}

template <class T>
inline complex<double> getcm(const T *const f[2], size_t i) {
  return complex<double>(f[0][i], f[1][i]);
}

void fields::get_point(monitor_point *pt, const vec &loc) const {
  if (pt == NULL) abort("Error:  get_point passed a null pointer!\n");
  for (int i = 0; i < 10; i++)
    pt->f[i] = 0.0;
  pt->loc = loc;
  pt->t = time();
  FOR_COMPONENTS(c) {
    if (gv.has_field(c)) pt->f[c] = get_field(c, loc);
  }
}

complex<double> fields::get_field(int c, const vec &loc, bool parallel) const {
  return (is_derived(c) ? get_field(derived_component(c), loc, parallel)
                        : get_field(component(c), loc, parallel));
}

double fields::get_field(derived_component c, const vec &loc, bool parallel) const {
  component c1 = Ex, c2 = Ex;
  double sum = 0;
  switch (c) {
    case Sx:
    case Sy:
    case Sz:
    case Sr:
    case Sp:
      switch (c) {
        case Sx:
          c1 = Ey;
          c2 = Hz;
          break;
        case Sy:
          c1 = Ez;
          c2 = Hx;
          break;
        case Sz:
          c1 = Ex;
          c2 = Hy;
          break;
        case Sr:
          c1 = Ep;
          c2 = Hz;
          break;
        case Sp:
          c1 = Ez;
          c2 = Hr;
          break;
        default: break; // never
      }
      sum += real(conj(get_field(c1, loc, parallel)) * get_field(c2, loc, parallel));
      sum -= real(conj(get_field(direction_component(Ex, component_direction(c2)), loc, parallel)) *
                  get_field(direction_component(Hx, component_direction(c1)), loc, parallel));
      return sum;
    case EnergyDensity:
    case D_EnergyDensity:
    case H_EnergyDensity:
      if (c != H_EnergyDensity) FOR_ELECTRIC_COMPONENTS(c1) {
          if (gv.has_field(c1)) {
            c2 = direction_component(Dx, component_direction(c1));
            sum += real(conj(get_field(c1, loc, parallel)) * get_field(c2, loc, parallel));
          }
        }
      if (c != D_EnergyDensity) FOR_MAGNETIC_COMPONENTS(c1) {
          if (gv.has_field(c1)) {
            c2 = direction_component(Bx, component_direction(c1));
            sum += real(conj(get_field(c1, loc, parallel)) * get_field(c2, loc, parallel));
          }
        }
      return sum * 0.5;
    default: abort("unknown derived_component in get_field");
  }
}

complex<double> fields::get_field(component c, const vec &loc, bool parallel) const {
  switch (c) {
    case Dielectric: return get_eps(loc);
    case Permeability: return get_mu(loc);
    case NO_COMPONENT: return 1.0;
    default:
      ivec ilocs[8];
      double w[8];
      complex<double> res = 0.0;
      gv.interpolate(c, loc, ilocs, w);
      for (int argh = 0; argh < 8 && w[argh]; argh++)
        res += w[argh] * get_field(c, ilocs[argh], false);
      if (gv.dim == D2 && loc.in_direction(Z) != 0) // special_kz handling
        res *= std::polar(1.0, 2 * pi * beta * loc.in_direction(Z));
      return parallel ? sum_to_all(res) : res;
  }
}

complex<double> fields::get_field(component c, const ivec &origloc, bool parallel) const {
  ivec iloc = origloc;
  complex<double> kphase = 1.0;
  locate_point_in_user_volume(&iloc, &kphase);
  for (int sn = 0; sn < S.multiplicity(); sn++)
    for (int i = 0; i < num_chunks; i++)
      if (chunks[i]->gv.owns(S.transform(iloc, sn))) {
        complex<double> val = S.phase_shift(c, sn) * kphase *
                              chunks[i]->get_field(S.transform(c, sn), S.transform(iloc, sn));
        return parallel ? sum_to_all(val) : val;
      }
  return 0.0;
}

complex<double> fields_chunk::get_field(component c, const ivec &iloc) const {
  if (is_mine())
    return f[c][0] ? (f[c][1] ? getcm(f[c], gv.index(c, iloc)) : f[c][0][gv.index(c, iloc)]) : 0.0;
  else
    return 0.0;
}

complex<double> fields::get_chi1inv(component c, direction d, const ivec &origloc, double frequency,
                                    bool parallel) const {
  ivec iloc = origloc;
  complex<double> aaack = 1.0;
  locate_point_in_user_volume(&iloc, &aaack);
  for (int sn = 0; sn < S.multiplicity(); sn++)
    for (int i = 0; i < num_chunks; i++)
      if (chunks[i]->gv.owns(S.transform(iloc, sn))) {
        signed_direction ds = S.transform(d, sn);
        complex<double> val =
            chunks[i]->get_chi1inv(S.transform(c, sn), ds.d, S.transform(iloc, sn), frequency) *
            complex<double>(ds.flipped ^ S.transform(component_direction(c), sn).flipped ? -1 : 1,
                            0);
        return parallel ? sum_to_all(val) : val;
      }
  return d == component_direction(c) ? 1.0 : 0; // default to vacuum outside computational cell
}

complex<double> fields_chunk::get_chi1inv(component c, direction d, const ivec &iloc,
                                          double frequency) const {
  return s->get_chi1inv(c, d, iloc, frequency);
}

complex<double> fields::get_chi1inv(component c, direction d, const vec &loc, double frequency,
                                    bool parallel) const {
  ivec ilocs[8];
  double w[8];
  complex<double> res(0.0, 0.0);
  gv.interpolate(c, loc, ilocs, w);
  for (int argh = 0; argh < 8 && w[argh] != 0; argh++)
    res += w[argh] * get_chi1inv(c, d, ilocs[argh], frequency, false);
  return parallel ? sum_to_all(res) : res;
}

complex<double> fields::get_eps(const vec &loc, double frequency) const {
  complex<double> tr(0.0, 0.0);
  int nc = 0;
  FOR_ELECTRIC_COMPONENTS(c) {
    if (gv.has_field(c)) {
      tr += get_chi1inv(c, component_direction(c), loc, frequency, false);
      ++nc;
    }
  }
  return complex<double>(nc, 0) / sum_to_all(tr);
}

complex<double> fields::get_mu(const vec &loc, double frequency) const {
  complex<double> tr(0.0, 0.0);
  int nc = 0;
  FOR_MAGNETIC_COMPONENTS(c) {
    if (gv.has_field(c)) {
      tr += get_chi1inv(c, component_direction(c), loc, frequency, false);
      ++nc;
    }
  }
  return complex<double>(nc, 0) / sum_to_all(tr);
}

complex<double> structure::get_chi1inv(component c, direction d, const ivec &origloc,
                                       double frequency, bool parallel) const {
  ivec iloc = origloc;
  for (int sn = 0; sn < S.multiplicity(); sn++)
    for (int i = 0; i < num_chunks; i++)
      if (chunks[i]->gv.owns(S.transform(iloc, sn))) {
        signed_direction ds = S.transform(d, sn);
        complex<double> val =
            chunks[i]->get_chi1inv(S.transform(c, sn), ds.d, S.transform(iloc, sn), frequency) *
            complex<double>((ds.flipped ^ S.transform(component_direction(c), sn).flipped ? -1 : 1),
                            0);
        return parallel ? sum_to_all(val) : val;
      }
  return 0.0;
}

/* Set Vinv = inverse of V, where both V and Vinv are complex matrices.*/
void matrix_invert(std::complex<double> (&Vinv)[9], std::complex<double> (&V)[9]) {

  std::complex<double> det =
      (V[0 + 3 * 0] * (V[1 + 3 * 1] * V[2 + 3 * 2] - V[1 + 3 * 2] * V[2 + 3 * 1]) -
       V[0 + 3 * 1] * (V[0 + 3 * 1] * V[2 + 3 * 2] - V[1 + 3 * 2] * V[0 + 3 * 2]) +
       V[0 + 3 * 2] * (V[0 + 3 * 1] * V[1 + 3 * 2] - V[1 + 3 * 1] * V[0 + 3 * 2]));

  if (det == 0.0) abort("meep: Matrix is singular, aborting.\n");

  Vinv[0 + 3 * 0] = 1.0 / det * (V[1 + 3 * 1] * V[2 + 3 * 2] - V[1 + 3 * 2] * V[2 + 3 * 1]);
  Vinv[0 + 3 * 1] = 1.0 / det * (V[0 + 3 * 2] * V[2 + 3 * 1] - V[0 + 3 * 1] * V[2 + 3 * 2]);
  Vinv[0 + 3 * 2] = 1.0 / det * (V[0 + 3 * 1] * V[1 + 3 * 2] - V[0 + 3 * 2] * V[1 + 3 * 1]);
  Vinv[1 + 3 * 0] = 1.0 / det * (V[1 + 3 * 2] * V[2 + 3 * 0] - V[1 + 3 * 0] * V[2 + 3 * 2]);
  Vinv[1 + 3 * 1] = 1.0 / det * (V[0 + 3 * 0] * V[2 + 3 * 2] - V[0 + 3 * 2] * V[2 + 3 * 0]);
  Vinv[1 + 3 * 2] = 1.0 / det * (V[0 + 3 * 2] * V[1 + 3 * 0] - V[0 + 3 * 0] * V[1 + 3 * 2]);
  Vinv[2 + 3 * 0] = 1.0 / det * (V[1 + 3 * 0] * V[2 + 3 * 1] - V[1 + 3 * 1] * V[2 + 3 * 0]);
  Vinv[2 + 3 * 1] = 1.0 / det * (V[0 + 3 * 1] * V[2 + 3 * 0] - V[0 + 3 * 0] * V[2 + 3 * 1]);
  Vinv[2 + 3 * 2] = 1.0 / det * (V[0 + 3 * 0] * V[1 + 3 * 1] - V[0 + 3 * 1] * V[1 + 3 * 0]);
}

complex<double> structure_chunk::get_chi1inv_at_pt(component c, direction d, int idx,
                                                   double frequency) const {
  complex<double> res(0.0, 0.0);
  if (is_mine()) {
    if (frequency == 0)
      return chi1inv[c][d] ? chi1inv[c][d][idx] : (d == component_direction(c) ? 1.0 : 0);
    // ----------------------------------------------------------------- //
    // ---- Step 1: Get instantaneous chi1 tensor ----------------------
    // ----------------------------------------------------------------- //

    int my_stuff = E_stuff;
    component comp_list[3];
    if (is_electric(c)) {
      comp_list[0] = Ex;
      comp_list[1] = Ey;
      comp_list[2] = Ez;
      my_stuff = E_stuff;
    }
    else if (is_magnetic(c)) {
      comp_list[0] = Hx;
      comp_list[1] = Hy;
      comp_list[2] = Hz;
      my_stuff = H_stuff;
    }
    else if (is_D(c)) {
      comp_list[0] = Dx;
      comp_list[1] = Dy;
      comp_list[2] = Dz;
      my_stuff = D_stuff;
    }
    else if (is_B(c)) {
      comp_list[0] = Bx;
      comp_list[1] = By;
      comp_list[2] = Bz;
      my_stuff = B_stuff;
    }

    std::complex<double> chi1_inv_tensor[9] = {
        std::complex<double>(1, 0), std::complex<double>(0, 0), std::complex<double>(0, 0),
        std::complex<double>(0, 0), std::complex<double>(1, 0), std::complex<double>(0, 0),
        std::complex<double>(0, 0), std::complex<double>(0, 0), std::complex<double>(1, 0)};
    std::complex<double> chi1_tensor[9] = {
        std::complex<double>(1, 0), std::complex<double>(0, 0), std::complex<double>(0, 0),
        std::complex<double>(0, 0), std::complex<double>(1, 0), std::complex<double>(0, 0),
        std::complex<double>(0, 0), std::complex<double>(0, 0), std::complex<double>(1, 0)};

    // Set up the chi1inv tensor with the DC components
    for (int com_it = 0; com_it < 3; com_it++) {
      for (int dir_int = 0; dir_int < 3; dir_int++) {
        if (chi1inv[comp_list[com_it]][dir_int])
          chi1_inv_tensor[com_it + 3 * dir_int] = chi1inv[comp_list[com_it]][dir_int][idx];
      }
    }

    matrix_invert(chi1_tensor, chi1_inv_tensor); // We have the inverse, so let's invert it.

    // ----------------------------------------------------------------- //
    // ---- Step 2: Evaluate susceptibilities of each tensor element ---
    // ----------------------------------------------------------------- //

    // loop over tensor elements
    for (int com_it = 0; com_it < 3; com_it++) {
      for (int dir_int = 0; dir_int < 3; dir_int++) {
        std::complex<double> eps = chi1_tensor[com_it + 3 * dir_int];
        component cc = comp_list[com_it];
        direction dd = (direction)dir_int;
        // Loop through and add up susceptibility contributions
        // locate correct susceptibility list
        susceptibility *my_sus = chiP[my_stuff];
        while (my_sus) {
          if (my_sus->sigma[cc][dd]) {
            double sigma = my_sus->sigma[cc][dd][idx];
            eps += my_sus->chi1(frequency, sigma);
          }
          my_sus = my_sus->next;
        }

        // Account for conductivity term
        if (conductivity[cc][dd]) {
          double conductivityCur = conductivity[cc][dd][idx];
          eps = std::complex<double>(1.0, (conductivityCur / frequency)) * eps;
        }
        chi1_tensor[com_it + 3 * dir_int] = eps;
      }
    }

    // ----------------------------------------------------------------- //
    // ---- Step 3: Invert chi1 matrix to get chi1inv matrix -----------
    // ----------------------------------------------------------------- //

    matrix_invert(chi1_inv_tensor, chi1_tensor); // We have the inverse, so let's invert it.
    res = chi1_inv_tensor[component_index(c) + 3 * d];
  }
  return res;
}

complex<double> structure_chunk::get_chi1inv(component c, direction d, const ivec &iloc,
                                             double frequency) const {
  return get_chi1inv_at_pt(c, d, gv.index(c, iloc), frequency);
}

complex<double> structure::get_chi1inv(component c, direction d, const vec &loc, double frequency,
                                       bool parallel) const {
  ivec ilocs[8];
  double w[8];
  complex<double> res(0.0, 0.0);
  gv.interpolate(c, loc, ilocs, w);
  for (int argh = 0; argh < 8 && w[argh] != 0; argh++)
    res += w[argh] * get_chi1inv(c, d, ilocs[argh], frequency, false);
  return parallel ? sum_to_all(res) : res;
}

complex<double> structure::get_eps(const vec &loc, double frequency) const {
  complex<double> tr(0.0, 0.0);
  int nc = 0;
  FOR_ELECTRIC_COMPONENTS(c) {
    if (gv.has_field(c)) {
      tr += get_chi1inv(c, component_direction(c), loc, frequency, false);
      ++nc;
    }
  }
  return complex<double>(nc, 0) / sum_to_all(tr);
}

complex<double> structure::get_mu(const vec &loc, double frequency) const {
  complex<double> tr(0.0, 0.0);
  int nc = 0;
  FOR_MAGNETIC_COMPONENTS(c) {
    if (gv.has_field(c)) {
      tr += get_chi1inv(c, component_direction(c), loc, frequency, false);
      ++nc;
    }
  }
  return complex<double>(nc, 0) / sum_to_all(tr);
}

monitor_point *fields::get_new_point(const vec &loc, monitor_point *the_list) const {
  monitor_point *p = new monitor_point();
  get_point(p, loc);
  p->next = the_list;
  return p;
}

complex<double> monitor_point::get_component(component w) { return f[w]; }

double monitor_point::poynting_in_direction(direction d) {
  direction d1 = cycle_direction(loc.dim, d, 1);
  direction d2 = cycle_direction(loc.dim, d, 2);

  // below Ex and Hx are used just to say that we want electric or magnetic component
  complex<double> E1 = get_component(direction_component(Ex, d1));
  complex<double> E2 = get_component(direction_component(Ex, d2));
  complex<double> H1 = get_component(direction_component(Hx, d1));
  complex<double> H2 = get_component(direction_component(Hx, d2));

  return (real(E1) * real(H2) - real(E2) * real(H1)) + (imag(E1) * imag(H2) - imag(E2) * imag(H1));
}

double monitor_point::poynting_in_direction(vec dir) {
  if (dir.dim != loc.dim) abort("poynting_in_direction: dir.dim != loc.dim\n");
  dir = dir / abs(dir);
  double result = 0.0;
  LOOP_OVER_DIRECTIONS(dir.dim, d) { result += dir.in_direction(d) * poynting_in_direction(d); }
  return result;
}

void monitor_point::fourier_transform(component w, complex<double> **a, complex<double> **f,
                                      int *numout, double fmin, double fmax, int maxbands) {
  int n = 1;
  monitor_point *p = next;
  double tmax = t, tmin = t;
  while (p) {
    n++;
    if (p->t > tmax) tmax = p->t;
    if (p->t < tmin) tmin = p->t;
    p = p->next;
  }
  p = this;
  complex<double> *d = new complex<double>[n];
  for (int i = 0; i < n; i++, p = p->next) {
    d[i] = p->get_component(w);
  }
  if (fmin > 0.0) { // Get rid of any static fields_chunk!
    complex<double> mean = 0.0;
    for (int i = 0; i < n; i++)
      mean += d[i];
    mean /= n;
    for (int i = 0; i < n; i++)
      d[i] -= mean;
  }
#if HAVE_SOME_FFTW
  if ((fmin > 0.0 || fmax > 0.0) && maxbands > 0) {
#else
  if ((fmin <= 0.0 && fmax <= 0.0) || maxbands <= 0) {
    maxbands = n;
    fmin = 0;
    fmax = (n - 1) * (1.0 / (tmax - tmin));
  }
#endif
    *a = new complex<double>[maxbands];
    *f = new complex<double>[maxbands];
    *numout = maxbands;
    delete[] d;
    for (int i = 0; i < maxbands; i++) {
      double df = (maxbands == 1) ? 0.0 : (fmax - fmin) / (maxbands - 1);
      (*f)[i] = fmin + i * df;
      (*a)[i] = 0.0;
      p = this;
      while (p) {
        double inside = 2 * pi * real((*f)[i]) * p->t;
        (*a)[i] += p->get_component(w) * complex<double>(cos(inside), sin(inside));
        p = p->next;
      }
      (*a)[i] /= (tmax - tmin);
    }
#if HAVE_SOME_FFTW
  }
  else {
    *numout = n;
    *a = new complex<double>[n];
    *f = d;
    fftw_complex *in = (fftw_complex *)d, *out = (fftw_complex *)*a;
    fftw_plan p;
#ifdef HAVE_LIBFFTW3
    p = fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
#else
    p = fftw_create_plan(n, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_one(p, in, out);
    fftw_destroy_plan(p);
#endif
    for (int i = 0; i < n; i++) {
      (*f)[i] = i * (1.0 / (tmax - tmin));
      if (real((*f)[i]) > 0.5 * n / (tmax - tmin)) (*f)[i] -= n / (tmax - tmin);
      (*a)[i] *= (tmax - tmin) / n;
    }
  }
#endif
}

void monitor_point::harminv(component w, complex<double> **a, complex<double> **f, int *numout,
                            double fmin, double fmax, int maxbands) {
  int n = 1;
  monitor_point *p = next;
  double tmax = t, tmin = t;
  while (p) {
    n++;
    if (p->t > tmax) tmax = p->t;
    if (p->t < tmin) tmin = p->t;
    p = p->next;
  }
  p = this;
  complex<double> *d = new complex<double>[n];
  for (int i = 0; i < n; i++, p = p->next) {
    d[i] = p->get_component(w);
  }
  *a = new complex<double>[n];
  double *f_re = new double[n];
  double *f_im = new double[n];
  *numout = do_harminv(d, n, (tmax - tmin) / (n - 1), fmin, fmax, maxbands, *a, f_re, f_im, NULL);
  *f = new complex<double>[*numout];
  for (int i = 0; i < *numout; i++)
    (*f)[i] = complex<double>(f_re[i], f_im[i]);
  delete[] f_re;
  delete[] f_im;
  delete[] d;
}

} // namespace meep
