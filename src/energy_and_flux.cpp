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
#include <string.h>

#include "meep.hpp"
#include "meep_internals.hpp"

using namespace std;

namespace meep {

/* Energy calculation */

double fields::count_volume(component c) {
  double vol = 0;
  for (int i = 0; i < num_chunks; i++)
    if (chunks[i]->is_mine()) vol += chunks[i]->count_volume(c);
  return sum_to_all(vol);
}

double fields_chunk::count_volume(component c) {
  double vol = 0;
  for (size_t i = 0; i < gv.ntot(); i++)
    vol += gv.dV(c, i).full_volume();
  return vol;
}

double fields::total_energy() { return energy_in_box(user_volume.surroundings()); }

double fields::field_energy() { return field_energy_in_box(user_volume.surroundings()); }

double fields::energy_in_box(const volume &where) {
  return thermo_energy_in_box(where) + field_energy_in_box(where);
}

double fields::field_energy_in_box(const volume &where) {
  synchronize_magnetic_fields();
  double cur_step_magnetic_energy = magnetic_energy_in_box(where);
  restore_magnetic_fields();
  return electric_energy_in_box(where) + cur_step_magnetic_energy;
}

static complex<double> dot_integrand(const complex<double> *fields, const vec &loc, void *data_) {
  (void)loc;
  (void)data_; // unused;
  return real(conj(fields[0]) * fields[1]);
}

double fields::field_energy_in_box(component c, const volume &where) {
  if (coordinate_mismatch(gv.dim, c)) return 0.0;

  component cs[2];
  if (is_electric(c) || is_D(c)) {
    cs[0] = direction_component(Ex, component_direction(c));
    cs[1] = direction_component(Dx, component_direction(c));
  }
  else if (is_magnetic(c) || is_B(c)) {
    cs[0] = direction_component(Hx, component_direction(c));
    cs[1] = direction_component(Bx, component_direction(c));
  }
  else
    abort("invalid field component in field_energy_in_box");

  return real(integrate(2, cs, dot_integrand, 0, where)) * 0.5;
}

double fields::electric_energy_in_box(const volume &where) {
  long double sum = 0.0;
  FOR_ELECTRIC_COMPONENTS(c) { sum += field_energy_in_box(c, where); }
  return sum;
}

double fields::magnetic_energy_in_box(const volume &where) {
  long double sum = 0.0;
  FOR_MAGNETIC_COMPONENTS(c) { sum += field_energy_in_box(c, where); }
  return sum;
}

template <class T>
void fields_chunk::backup_component(component c) {
  DOCMP {
    if (c < NUM_FIELD_COMPONENTS && f[c][cmp] &&
        // in mu=1 regions where H==B, don't bother to backup H
        !(is_magnetic(c) && f[c][cmp] == f[direction_component(Bx, component_direction(c))][cmp])) {

#define BACKUP(f)                                                                                  \
  if (f[c][cmp]) {                                                                                 \
    if (!f##_backup[c][cmp]) f##_backup[c][cmp] = new T[gv.ntot()];                                \
    memcpy(f##_backup[c][cmp], f[c][cmp], gv.ntot() * sizeof(T));                                  \
  }

      BACKUP(f);
      BACKUP(f_u);
      BACKUP(f_w);
      BACKUP(f_cond);

#undef BACKUP
    }
  }
}

void fields_chunk::restore_component(component c) {
  DOCMP {
#define RESTORE(f)                                                                                 \
  if (f##_backup[c][cmp] && f[c][cmp])                                                             \
    memcpy(f[c][cmp], f##_backup[c][cmp], gv.ntot() * sizeof(T));

    RESTORE(f);
    RESTORE(f_u);
    RESTORE(f_w);
    RESTORE(f_cond);

#undef RESTORE
  }
}

void fields_chunk::average_with_backup(component c) {
  DOCMP {
    T *fc = f[c][cmp];
    T *backup = f_backup[c][cmp];
    if (fc && backup)
      for (size_t i = 0; i < gv.ntot(); i++)
        fc[i] = 0.5 * (fc[i] + backup[i]);
  }
}

void fields::synchronize_magnetic_fields() {
  if (synchronized_magnetic_fields++) return; // already synched
  for (int i = 0; i < num_chunks; i++)
    if (chunks[i]->is_mine()) {
      FOR_B_COMPONENTS(c) { chunks[i]->backup_component(c); }
      FOR_MAGNETIC_COMPONENTS(c) { chunks[i]->backup_component(c); }
    }
  am_now_working_on(Stepping);
  calc_sources(time()); // for B sources
  step_db(B_stuff);
  step_source(B_stuff);
  step_boundaries(B_stuff);
  calc_sources(time() + 0.5 * dt); // for integrated H sources
  update_eh(H_stuff);
  step_boundaries(H_stuff);
  finished_working();
  for (int i = 0; i < num_chunks; i++)
    if (chunks[i]->is_mine()) {
      FOR_B_COMPONENTS(c) { chunks[i]->average_with_backup(c); }
      FOR_MAGNETIC_COMPONENTS(c) { chunks[i]->average_with_backup(c); }
    }
}

void fields::restore_magnetic_fields() {
  if (!synchronized_magnetic_fields      // already restored
      || --synchronized_magnetic_fields) // not ready to restore yet
    return;
  for (int i = 0; i < num_chunks; i++)
    if (chunks[i]->is_mine()) {
      FOR_B_COMPONENTS(c) { chunks[i]->restore_component(c); }
      FOR_MAGNETIC_COMPONENTS(c) { chunks[i]->restore_component(c); }
    }
}

double fields::thermo_energy_in_box(const volume &where) {
  long double sum = 0.0;
  (void)where; // unused
  abort("thermo_energy_in_box no longer supported");
  return sum_to_all(sum);
}

/* Compute ExH integral in box using current fields, ignoring fact
   that this E and H correspond to different times. */
double fields::flux_in_box_wrongH(direction d, const volume &where) {
  if (coordinate_mismatch(gv.dim, d)) return 0.0;

  component cE[2] = {Ey, Ez}, cH[2] = {Hz, Hy};
  switch (d) {
    case X: cE[0] = Ey, cE[1] = Ez, cH[0] = Hz, cH[1] = Hy; break;
    case Y: cE[0] = Ez, cE[1] = Ex, cH[0] = Hx, cH[1] = Hz; break;
    case R: cE[0] = Ep, cE[1] = Ez, cH[0] = Hz, cH[1] = Hp; break;
    case P: cE[0] = Ez, cE[1] = Er, cH[0] = Hr, cH[1] = Hz; break;
    case Z:
      if (gv.dim == Dcyl)
        cE[0] = Er, cE[1] = Ep, cH[0] = Hp, cH[1] = Hr;
      else
        cE[0] = Ex, cE[1] = Ey, cH[0] = Hy, cH[1] = Hx;
      break;
    case NO_DIRECTION: abort("cannot get flux in NO_DIRECTION");
  }

  long double sum = 0.0;
  for (int i = 0; i < 2; ++i) {
    component cs[2];
    cs[0] = cE[i];
    cs[1] = cH[i];
    sum += real(integrate(2, cs, dot_integrand, 0, where)) * (1 - 2 * i);
  }
  return sum;
}

double fields::flux_in_box(direction d, const volume &where) {
  synchronize_magnetic_fields();
  double cur_step_flux = flux_in_box_wrongH(d, where);
  restore_magnetic_fields();
  return cur_step_flux;
}

flux_vol *fields::add_flux_vol(direction d, const volume &where) {
  if (where.dim != gv.dim) abort("invalid dimensionality in add_flux_vol");
  if (d == NO_DIRECTION || coordinate_mismatch(gv.dim, d))
    abort("invalid direction in add_flux_vol");
  return new flux_vol(this, d, where);
}

// As add_flux_vol, but infer direction from where (if possible)
flux_vol *fields::add_flux_plane(const volume &where) {
  return add_flux_vol(where.normal_direction(), where);
}

flux_vol *fields::add_flux_plane(const vec &p1, const vec &p2) {
  return add_flux_plane(volume(p1, p2));
}

/************************************************************************/

/* Note that computation of modal grid_volume by this definition is
   somewhat problematic computationally, because we need to compute
   max|D*E|, which requires averaging discontinuous functions.  Hence,
   except for the special case of 2d TM polarization, the computed
   value tends to have a large error bar if the maximum lies on a
   dielectric boundary as it commonly does.

   A better method would be to average only continuous quantities in
   order to compute the fields on the Centered grid, but this
   is more expensive and requires us to know the boundary orientation, and
   does not seem worth the trouble at this point. */

static complex<double> dot3_max_integrand(const complex<double> *fields, const vec &loc,
                                          void *data_) {
  (void)loc;
  (void)data_; // unused;
  return (real(conj(fields[0]) * fields[3]) + real(conj(fields[1]) * fields[4]) +
          real(conj(fields[2]) * fields[5]));
}

double fields::electric_energy_max_in_box(const volume &where) {
  component cs[6];
  if (gv.dim == Dcyl) {
    cs[0] = Er;
    cs[1] = Ep;
    cs[2] = Ez;
    cs[3 + 0] = Dr;
    cs[3 + 1] = Dp;
    cs[3 + 2] = Dz;
  }
  else {
    cs[0] = Ex;
    cs[1] = Ey;
    cs[2] = Ez;
    cs[3 + 0] = Dx;
    cs[3 + 1] = Dy;
    cs[3 + 2] = Dz;
  }

  return max_abs(6, cs, dot3_max_integrand, 0, where) * 0.5;
}

/* "modal" grid_volume according to definition in:
      E. M. Purcell, Phys. Rev. B 69, 681 (1946).
    (based on spontaneous emission enhancement). */
double fields::modal_volume_in_box(const volume &where) {
  return electric_energy_in_box(where) / electric_energy_max_in_box(where);
}

/************************************************************************/

/* compute integral f(x) * Re[conj(f1)*f2] * 0.5, which is useful for
   perturbation theory, etcetera, where f1 and f2 are two field components
   on the same Yee lattice (e.g. Hx and Hx or Ex and Dx). */

typedef double (*fx_func)(const vec &);

static complex<double> dot_fx_integrand(const complex<double> *fields, const vec &loc,
                                        void *data_) {
  fx_func fx = (fx_func)data_;
  return (real(conj(fields[0]) * fields[1]) * fx(loc));
}

/* computes integral of f(x) * |E|^2 / integral epsilon*|E|^2 */
double fields::electric_sqr_weighted_integral(double (*f)(const vec &), const volume &where) {
  double sum = 0.0;
  FOR_ELECTRIC_COMPONENTS(c) {
    if (!coordinate_mismatch(gv.dim, component_direction(c))) {
      component cs[2];
      cs[0] = cs[1] = direction_component(Ex, component_direction(c));
      sum += real(integrate(2, cs, dot_fx_integrand, (void *)f, where));
    }
  }
  return sum * 0.5 / electric_energy_in_box(where);
}

/* computes integral of f(x) * epsilon*|E|^2 / integral epsilon*|E|^2 */
double fields::electric_energy_weighted_integral(double (*f)(const vec &), const volume &where) {
  double sum = 0.0;
  FOR_ELECTRIC_COMPONENTS(c) {
    if (!coordinate_mismatch(gv.dim, component_direction(c))) {
      component cs[2];
      cs[0] = direction_component(Ex, component_direction(c));
      cs[1] = direction_component(Dx, component_direction(c));
      sum += real(integrate(2, cs, dot_fx_integrand, (void *)f, where));
    }
  }
  return sum * 0.5 / electric_energy_in_box(where);
}

} // namespace meep
