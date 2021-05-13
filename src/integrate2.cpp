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

#include "meep.hpp"
#include "meep_internals.hpp"

/* integration routine similar to those in integrate.cpp, but
   integrating a combination of two fields from two different
   simulations (assumed to have identical grids etcetera), based on
   fields::loop_in_chunk */

using namespace std;

namespace meep {

struct integrate_data {
  int num_fvals;
  const component *components;
  const fields *fields2;
  int num_fvals2;
  const component *components2;
  component *cS;
  complex<double> *ph;
  complex<double> *fvals;
  ptrdiff_t *offsets;
  int ninveps;
  component inveps_cs[3];
  direction inveps_ds[3];
  int ninvmu;
  component invmu_cs[3];
  direction invmu_ds[3];
  complex<long double> sum;
  double maxabs;
  field_function integrand;
  void *integrand_data_;
};

template <class T>
static void integrate_chunkloop(fields_chunk *fc, int ichunk, component cgrid, ivec is, ivec ie,
                                vec s0, vec s1, vec e0, vec e1, double dV0, double dV1, ivec shift,
                                complex<double> shift_phase, const symmetry &S, int sn,
                                void *data_) {
  (void)ichunk; // unused
  integrate_data *data = (integrate_data *)data_;
  ptrdiff_t *off = data->offsets;
  component *cS = data->cS;
  complex<double> *fvals = data->fvals, *ph = data->ph;
  complex<long double> sum = 0.0;
  double maxabs = 0;
  const component *iecs = data->inveps_cs;
  const direction *ieds = data->inveps_ds;
  ptrdiff_t ieos[6];
  const component *imcs = data->invmu_cs;
  const direction *imds = data->invmu_ds;
  int num_fvals1 = data->num_fvals;
  int num_fvals2 = data->num_fvals2;
  ptrdiff_t imos[6];
  const fields_chunk *fc2 = data->fields2->chunks[ichunk];

  for (int i = 0; i < num_fvals1; ++i) {
    cS[i] = S.transform(data->components[i], -sn);
    if (cS[i] == Dielectric || cS[i] == Permeability)
      ph[i] = 1.0;
    else {
      if (cgrid == Centered) fc->gv.yee2cent_offsets(cS[i], off[2 * i], off[2 * i + 1]);
      ph[i] = shift_phase * S.phase_shift(cS[i], sn);
    }
  }
  for (int i = 0; i < num_fvals2; ++i) {
    int j = i + num_fvals1;
    cS[j] = S.transform(data->components2[i], -sn);
    if (cS[j] == Dielectric || cS[j] == Permeability)
      ph[j] = 1.0;
    else {
      if (cgrid == Centered) fc->gv.yee2cent_offsets(cS[j], off[2 * j], off[2 * j + 1]);
      ph[j] = shift_phase * S.phase_shift(cS[j], sn);
    }
  }
  for (int k = 0; k < data->ninveps; ++k)
    fc->gv.yee2cent_offsets(iecs[k], ieos[2 * k], ieos[2 * k + 1]);
  for (int k = 0; k < data->ninvmu; ++k)
    fc->gv.yee2cent_offsets(imcs[k], imos[2 * k], imos[2 * k + 1]);

  vec rshift(shift * (0.5 * fc->gv.inva));
  LOOP_OVER_IVECS(fc->gv, is, ie, idx) {
    IVEC_LOOP_LOC(fc->gv, loc);
    loc = S.transform(loc, sn) + rshift;

    for (int i = 0; i < data->num_fvals; ++i) {
      if (cS[i] == Dielectric) {
        double tr = 0.0;
        for (int k = 0; k < data->ninveps; ++k) {
          const T *ie = fc->s->chi1inv[iecs[k]][ieds[k]];
          if (ie)
            tr += (ie[idx] + ie[idx + ieos[2 * k]] + ie[idx + ieos[1 + 2 * k]] +
                   ie[idx + ieos[2 * k] + ieos[1 + 2 * k]]);
          else
            tr += 4; // default inveps == 1
        }
        fvals[i] = (4 * data->ninveps) / tr;
      }
      else if (cS[i] == Permeability) {
        double tr = 0.0;
        for (int k = 0; k < data->ninvmu; ++k) {
          const T *im = fc->s->chi1inv[imcs[k]][imds[k]];
          if (im)
            tr += (im[idx] + im[idx + imos[2 * k]] + im[idx + imos[1 + 2 * k]] +
                   im[idx + imos[2 * k] + imos[1 + 2 * k]]);
          else
            tr += 4; // default invmu == 1
        }
        fvals[i] = (4 * data->ninvmu) / tr;
      }
      else {
        double f[2];
        for (int k = 0; k < 2; ++k)
          if (fc->f[cS[i]][k])
            f[k] = 0.25 * (fc->f[cS[i]][k][idx] + fc->f[cS[i]][k][idx + off[2 * i]] +
                           fc->f[cS[i]][k][idx + off[2 * i + 1]] +
                           fc->f[cS[i]][k][idx + off[2 * i] + off[2 * i + 1]]);
          else
            f[k] = 0;
        fvals[i] = complex<double>(f[0], f[1]) * ph[i];
      }
    }

    for (int j = 0; j < num_fvals2; ++j) {
      int i = j + num_fvals1;
      if (cS[i] == Dielectric) {
        double tr = 0.0;
        for (int k = 0; k < data->ninveps; ++k) {
          const T *ie = fc2->s->chi1inv[iecs[k]][ieds[k]];
          if (ie)
            tr += (ie[idx] + ie[idx + ieos[2 * k]] + ie[idx + ieos[1 + 2 * k]] +
                   ie[idx + ieos[2 * k] + ieos[1 + 2 * k]]);
          else
            tr += 4; // default inveps == 1
        }
        fvals[i] = (4 * data->ninveps) / tr;
      }
      else if (cS[i] == Permeability) {
        double tr = 0.0;
        for (int k = 0; k < data->ninvmu; ++k) {
          const T *im = fc2->s->chi1inv[imcs[k]][imds[k]];
          if (im)
            tr += (im[idx] + im[idx + imos[2 * k]] + im[idx + imos[1 + 2 * k]] +
                   im[idx + imos[2 * k] + imos[1 + 2 * k]]);
          else
            tr += 4; // default invmu == 1
        }
        fvals[i] = (4 * data->ninvmu) / tr;
      }
      else {
        double f[2];
        for (int k = 0; k < 2; ++k)
          if (fc2->f[cS[i]][k])
            f[k] = 0.25 * (fc2->f[cS[i]][k][idx] + fc2->f[cS[i]][k][idx + off[2 * i]] +
                           fc2->f[cS[i]][k][idx + off[2 * i + 1]] +
                           fc2->f[cS[i]][k][idx + off[2 * i] + off[2 * i + 1]]);
          else
            f[k] = 0;
        fvals[i] = complex<double>(f[0], f[1]) * ph[i];
      }
    }

    complex<double> integrand = data->integrand(fvals, loc, data->integrand_data_);
    maxabs = max(maxabs, abs(integrand));
    sum += integrand * IVEC_LOOP_WEIGHT(s0, s1, e0, e1, dV0 + dV1 * loop_i2);
  }

  data->maxabs = max(data->maxabs, maxabs);
  data->sum += sum;
}

complex<double> fields::integrate2(const fields &fields2, int num_fvals1,
                                   const component *components1, int num_fvals2,
                                   const component *components2, field_function integrand,
                                   void *integrand_data_, const volume &where, double *maxabs) {
  if (!equal_layout(fields2))
    abort("invalid call to integrate2: fields must have equal grid layout");

  if (num_fvals2 == 0)
    return integrate(num_fvals1, components1, integrand, integrand_data_, where, maxabs);
  if (num_fvals1 == 0)
    return const_cast<fields &>(fields2).integrate(num_fvals2, components2, integrand,
                                                   integrand_data_, where, maxabs);

  // check if components are all on the same grid:
  bool same_grid = true;
  for (int i = 1; i < num_fvals1; ++i)
    if (gv.iyee_shift(components1[i]) != gv.iyee_shift(components1[0])) {
      same_grid = false;
      break;
    }
  if (same_grid)
    for (int i = 0; i < num_fvals2; ++i)
      if (gv.iyee_shift(components2[i]) != gv.iyee_shift(components1[0])) {
        same_grid = false;
        break;
      }

  component cgrid = Centered;
  if (same_grid) cgrid = components1[0];

  integrate_data data;
  data.num_fvals = num_fvals1;
  data.components = components1;
  data.fields2 = &fields2;
  data.num_fvals2 = num_fvals2;
  data.components2 = components2;
  data.cS = new component[num_fvals1 + num_fvals2];
  data.ph = new complex<double>[num_fvals1 + num_fvals2];
  data.fvals = new complex<double>[num_fvals1 + num_fvals2];
  data.sum = 0;
  data.maxabs = 0;
  data.integrand = integrand;
  data.integrand_data_ = integrand_data_;

  /* compute inverse-epsilon directions for computing Dielectric fields */
  data.ninveps = 0;
  bool needs_dielectric = false;
  for (int i = 0; i < num_fvals1; ++i)
    if (components1[i] == Dielectric) {
      needs_dielectric = true;
      break;
    }
  if (!needs_dielectric)
    for (int i = 0; i < num_fvals2; ++i)
      if (components2[i] == Dielectric) {
        needs_dielectric = true;
        break;
      }
  if (needs_dielectric) FOR_ELECTRIC_COMPONENTS(c) if (gv.has_field(c)) {
      if (data.ninveps == 3) abort("more than 3 field components??");
      data.inveps_cs[data.ninveps] = c;
      data.inveps_ds[data.ninveps] = component_direction(c);
      ++data.ninveps;
    }

  /* compute inverse-mu directions for computing Permeability fields */
  data.ninvmu = 0;
  bool needs_permeability = false;
  for (int i = 0; i < num_fvals1; ++i)
    if (components1[i] == Permeability) {
      needs_permeability = true;
      break;
    }
  if (!needs_permeability)
    for (int i = 0; i < num_fvals2; ++i)
      if (components2[i] == Permeability) {
        needs_permeability = true;
        break;
      }
  if (needs_permeability) FOR_MAGNETIC_COMPONENTS(c) if (gv.has_field(c)) {
      if (data.ninvmu == 3) abort("more than 3 field components??");
      data.invmu_cs[data.ninvmu] = c;
      data.invmu_ds[data.ninvmu] = component_direction(c);
      ++data.ninvmu;
    }

  data.offsets = new ptrdiff_t[2 * (num_fvals1 + num_fvals2)];
  for (int i = 0; i < 2 * (num_fvals1 + num_fvals2); ++i)
    data.offsets[i] = 0;

  loop_in_chunks(integrate_chunkloop, (void *)&data, where, cgrid);

  delete[] data.offsets;
  delete[] data.fvals;
  delete[] data.ph;
  delete[] data.cS;

  if (maxabs) *maxabs = max_to_all(data.maxabs);
  data.sum = sum_to_all(data.sum);

  return complex<double>(real(data.sum), imag(data.sum));
}

typedef struct {
  field_rfunction integrand;
  void *integrand_data;
} rfun_wrap_data;
static complex<double> rfun_wrap(const complex<double> *fields, const vec &loc, void *data_) {
  rfun_wrap_data *data = (rfun_wrap_data *)data_;
  return data->integrand(fields, loc, data->integrand_data);
}

double fields::integrate2(const fields &fields2, int num_fvals1, const component *components1,
                          int num_fvals2, const component *components2, field_rfunction integrand,
                          void *integrand_data_, const volume &where, double *maxabs) {
  rfun_wrap_data data;
  data.integrand = integrand;
  data.integrand_data = integrand_data_;
  return real(integrate2(fields2, num_fvals1, components1, num_fvals2, components2, rfun_wrap,
                         &data, where, maxabs));
}

} // namespace meep
