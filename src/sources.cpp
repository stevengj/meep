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
#include <complex>

#include "meep.hpp"
#include "meep_internals.hpp"

using namespace std;

namespace meep {

/*********************************************************************/

// this function is necessary to make equality commutative ... ugh
bool src_times_equal(const src_time &t1, const src_time &t2) {
  return t1.is_equal(t2) && t2.is_equal(t1) && t1.is_integrated == t2.is_integrated;
}

src_time *src_time::add_to(src_time *others, src_time **added) const {
  if (others) {
    if (src_times_equal(*this, *others))
      *added = others;
    else
      others->next = add_to(others->next, added);
    return others;
  }
  else {
    src_time *t = clone();
    t->next = others;
    *added = t;
    return t;
  }
}

double src_time::last_time_max(double after) {
  after = max(last_time(), after);
  if (next)
    return next->last_time_max(after);
  else
    return after;
}

gaussian_src_time::gaussian_src_time(double f, double fwidth, double s) {
  freq = f;
  width = 1.0 / fwidth;
  peak_time = width * s;
  cutoff = width * s;

  // this is to make last_source_time as small as possible
  while (exp(-cutoff * cutoff / (2 * width * width)) < 1e-100)
    cutoff *= 0.9;
  cutoff = float(cutoff); // don't make cutoff sensitive to roundoff error
}

gaussian_src_time::gaussian_src_time(double f, double w, double st, double et) {
  freq = f;
  width = w;
  peak_time = 0.5 * (st + et);
  cutoff = (et - st) * 0.5;

  // this is to make last_source_time as small as possible
  while (exp(-cutoff * cutoff / (2 * width * width)) < 1e-100)
    cutoff *= 0.9;
  cutoff = float(cutoff); // don't make cutoff sensitive to roundoff error
}

complex<double> gaussian_src_time::dipole(double time) const {
  double tt = time - peak_time;
  if (float(fabs(tt)) > cutoff) return 0.0;

  // correction factor so that current amplitude (= d(dipole)/dt) is
  // ~ 1 near the peak of the Gaussian.
  complex<double> amp = 1.0 / complex<double>(0, -2 * pi * freq);

  return exp(-tt * tt / (2 * width * width)) * polar(1.0, -2 * pi * freq * tt) * amp;
}

// (1/\sqrt{2*pi}) \int e^{i\omega t} G(t) dt
// where G(t) is the *current* envelope, i.e. the time derivative
// of the dipole envelope
std::complex<double> gaussian_src_time::fourier_transform(const double f) {
  double omega = 2.0 * pi * f;
  double omega0 = 2.0 * pi * freq;
  double delta = (omega - omega0) * width;
  return width * polar(1.0, omega * peak_time) * exp(-0.5 * delta * delta);
}

bool gaussian_src_time::is_equal(const src_time &t) const {
  const gaussian_src_time *tp = dynamic_cast<const gaussian_src_time *>(&t);
  if (tp)
    return (tp->freq == freq && tp->width == width && tp->peak_time == peak_time &&
            tp->cutoff == cutoff);
  else
    return 0;
}

complex<double> continuous_src_time::dipole(double time) const {
  float rtime = float(time);
  if (rtime < start_time || rtime > end_time) return 0.0;

  // correction factor so that current amplitude (= d(dipole)/dt) is 1.
  complex<double> amp = 1.0 / (complex<double>(0, -1.0) * (2 * pi) * freq);

  if (width == 0.0)
    return exp(complex<double>(0, -1.0) * (2 * pi) * freq * time) * amp;
  else {
    double ts = (time - start_time) / width - slowness;
    double te = (end_time - time) / width - slowness;

    return exp(complex<double>(0, -1.0) * (2 * pi) * freq * time) * amp *
           (1.0 + tanh(ts))   // goes from 0 to 2
           * (1.0 + tanh(te)) // goes from 2 to 0
           * 0.25;
  }
}

bool continuous_src_time::is_equal(const src_time &t) const {
  const continuous_src_time *tp = dynamic_cast<const continuous_src_time *>(&t);
  if (tp)
    return (tp->freq == freq && tp->width == width && tp->start_time == start_time &&
            tp->end_time == end_time && tp->slowness == slowness);
  else
    return 0;
}

bool custom_src_time::is_equal(const src_time &t) const {
  const custom_src_time *tp = dynamic_cast<const custom_src_time *>(&t);
  if (tp)
    return (tp->start_time == start_time && tp->end_time == end_time && tp->func == func &&
            tp->data == data && tp->freq == freq);
  else
    return 0;
}

/*********************************************************************/

src_vol::src_vol(component cc, src_time *st, size_t n, ptrdiff_t *ind, complex<double> *amps) {
  c = cc;
  if (is_D(c)) c = direction_component(Ex, component_direction(c));
  if (is_B(c)) c = direction_component(Hx, component_direction(c));
  t = st;
  next = NULL;
  npts = n;
  index = ind;
  A = amps;
}

src_vol::src_vol(const src_vol &sv) {
  c = sv.c;
  t = sv.t;
  npts = sv.npts;
  index = new ptrdiff_t[npts];
  A = new complex<double>[npts];
  for (size_t j = 0; j < npts; j++) {
    index[j] = sv.index[j];
    A[j] = sv.A[j];
  }
  if (sv.next)
    next = new src_vol(*sv.next);
  else
    next = NULL;
}

src_vol *src_vol::add_to(src_vol *others) {
  if (others) {
    if (*this == *others) {
      if (npts != others->npts)
        abort("Cannot add grid_volume sources with different number of points\n");
      /* Compare all of the indices...if this ever becomes too slow,
         we can just compare the first and last indices. */
      for (size_t j = 0; j < npts; j++) {
        if (others->index[j] != index[j]) abort("Different indices\n");
        others->A[j] += A[j];
      }
    }
    else
      others->next = add_to(others->next);
    return others;
  }
  else {
    next = others;
    return this;
  }
}

/*********************************************************************/

// THIS VARIANT IS FOR BACKWARDS COMPATIBILITY, and is DEPRECATED:
void fields::add_point_source(component c, double freq, double width, double peaktime,
                              double cutoff, const vec &p, complex<double> amp, int is_c) {
  width /= freq;

  if (is_c) { // TODO: don't ignore peaktime?
    continuous_src_time src(freq, width, time(), infinity, cutoff);
    if (is_magnetic(c)) src.is_integrated = false;
    add_point_source(c, src, p, amp);
  }
  else {
    cutoff = gv.inva + cutoff * width;
    if (peaktime <= 0.0) peaktime = time() + cutoff;

    // backward compatibility (slight phase shift in old Meep version)
    peaktime += is_magnetic(c) ? -dt * 0.5 : dt;

    gaussian_src_time src(freq, width, peaktime - cutoff, peaktime + cutoff);
    if (is_magnetic(c)) src.is_integrated = false;
    add_point_source(c, src, p, amp);
  }
}

void fields::add_point_source(component c, const src_time &src, const vec &p, complex<double> amp) {
  add_volume_source(c, src, volume(p, p), amp);
}

static complex<double> one(const vec &pt) {
  (void)pt;
  return 1.0;
}
void fields::add_volume_source(component c, const src_time &src, const volume &where,
                               complex<double> amp) {
  add_volume_source(c, src, where, one, amp);
}

struct src_vol_chunkloop_data {
  complex<double> (*A)(const vec &);
  complex<double> amp;
  src_time *src;
  vec center;
};

/* Adding source volumes can be treated as a kind of "integration"
   problem, since we need to loop over all the chunks that intersect
   the source grid_volume, with appropriate interpolation weights at the
   boundaries so that the integral of the current is fixed regardless
   of resolution.  Unlike most uses of fields::loop_in_chunks, however, we
   set use_symmetry=false: we only find the intersection of the grid_volume
   with the untransformed chunks (since the transformed versions are
   implicit). */
static void src_vol_chunkloop(fields_chunk *fc, int ichunk, component c, ivec is, ivec ie, vec s0,
                              vec s1, vec e0, vec e1, double dV0, double dV1, ivec shift,
                              complex<double> shift_phase, const symmetry &S, int sn, void *data_) {
  src_vol_chunkloop_data *data = (src_vol_chunkloop_data *)data_;

  (void)S;
  (void)sn; // these should be the identity
  (void)dV0;
  (void)dV1; // grid_volume weighting is included in data->amp
  (void)ichunk;

  size_t npts = 1;
  LOOP_OVER_DIRECTIONS(is.dim, d) { npts *= (ie.in_direction(d) - is.in_direction(d)) / 2 + 1; }
  ptrdiff_t *index_array = new ptrdiff_t[npts];
  complex<double> *amps_array = new complex<double>[npts];

  complex<double> amp = data->amp * conj(shift_phase);

  direction cd = component_direction(c);

  double inva = fc->gv.inva;
  size_t idx_vol = 0;
  LOOP_OVER_IVECS(fc->gv, is, ie, idx) {
    IVEC_LOOP_ILOC(fc->gv, iloc);
    if (!fc->gv.owns(iloc)) continue;

    IVEC_LOOP_LOC(fc->gv, loc);
    loc += shift * (0.5 * inva);

    vec rel_loc = loc - data->center;
    amps_array[idx_vol] = IVEC_LOOP_WEIGHT(s0, s1, e0, e1, 1) * amp * data->A(rel_loc);

    /* for "D" sources, multiply by epsilon.  FIXME: this is not quite
       right because it doesn't handle non-diagonal chi1inv!
       similarly, for "B" sources, multiply by mu. */
    if (is_D(c) && fc->s->chi1inv[c - Dx + Ex][cd])
      amps_array[idx_vol] /= fc->s->chi1inv[c - Dx + Ex][cd][idx];
    if (is_B(c) && fc->s->chi1inv[c - Bx + Hx][cd])
      amps_array[idx_vol] /= fc->s->chi1inv[c - Bx + Hx][cd][idx];

    index_array[idx_vol++] = idx;
  }

  if (idx_vol > npts) abort("add_volume_source: computed wrong npts (%zd vs. %zd)", npts, idx_vol);

  src_vol *tmp = new src_vol(c, data->src, idx_vol, index_array, amps_array);
  field_type ft = is_magnetic(c) ? B_stuff : D_stuff;
  fc->sources[ft] = tmp->add_to(fc->sources[ft]);
}

void fields::add_srcdata(struct sourcedata cur_data, src_time *src, size_t n, std::complex<double>* amp_arr){
  sources = src->add_to(sources, &src);
  ptrdiff_t* index_arr = new ptrdiff_t[n];
  std::complex<double>* amp_arr_copy = new std::complex<double>[n];
  for (size_t i = 0; i < n; i++){
    index_arr[i] = cur_data.idx_arr[i];
    amp_arr_copy[i] = amp_arr[i];
  }
  component c = cur_data.near_fd_comp;
  src_vol *tmp = new src_vol(c, src, n, index_arr, amp_arr_copy);
  field_type ft = is_magnetic(c) ? B_stuff : D_stuff;
  if (0 > cur_data.fc_idx or cur_data.fc_idx >= num_chunks) abort("fields chunk index out of range");
  fields_chunk *fc = chunks[cur_data.fc_idx];
  if (!fc->is_mine()) abort("wrong fields chunk");
  fc->sources[ft] = tmp->add_to(fc->sources[ft]);
  // We can't do require_component(c) since that only works if all processes are adding
  // srcdata for the same components in the same order, which may not be true.
  // ... instead, the caller should call fields::require_source_components()
  //     after all add_srcdata calls are complete.
}

static double *amp_func_data_re = NULL;
static double *amp_func_data_im = NULL;
static const volume *amp_func_vol = NULL;
static size_t amp_file_dims[3];

complex<double> amp_file_func(const vec &p) {
  double x_size = 0, y_size = 0, z_size = 0;

  switch (amp_func_vol->dim) {
    case D1: z_size = amp_func_vol->in_direction(Z); break;
    case D2:
      x_size = amp_func_vol->in_direction(X);
      y_size = amp_func_vol->in_direction(Y);
      break;
    case D3:
      x_size = amp_func_vol->in_direction(X);
      y_size = amp_func_vol->in_direction(Y);
      z_size = amp_func_vol->in_direction(Z);
      break;
    case Dcyl:
      x_size = amp_func_vol->in_direction(X);
      z_size = amp_func_vol->in_direction(Z);
      break;
  }

  double rx = x_size == 0 ? 0 : 0.5 + p.x() / x_size;
  double ry = y_size == 0 ? 0 : 0.5 + p.y() / y_size;
  double rz = z_size == 0 ? 0 : 0.5 + p.z() / z_size;

  complex<double> res;
  res.real(linear_interpolate(rx, ry, rz, amp_func_data_re, amp_file_dims[0], amp_file_dims[1],
                              amp_file_dims[2], 1));
  res.imag(linear_interpolate(rx, ry, rz, amp_func_data_im, amp_file_dims[0], amp_file_dims[1],
                              amp_file_dims[2], 1));
  return res;
}

void fields::add_volume_source(component c, const src_time &src, const volume &where_,
                               complex<double> *arr, size_t dim1, size_t dim2, size_t dim3,
                               complex<double> amp) {

  amp_func_vol = &where_;

  amp_file_dims[0] = dim1;
  amp_file_dims[1] = dim2;
  amp_file_dims[2] = dim3;

  size_t total_size = dim1 * dim2 * dim3;
  amp_func_data_re = new double[total_size];
  amp_func_data_im = new double[total_size];

  for (size_t i = 0; i < total_size; ++i) {
    amp_func_data_re[i] = real(arr[i]);
    amp_func_data_im[i] = imag(arr[i]);
  }

  add_volume_source(c, src, where_, amp_file_func, amp);

  delete[] amp_func_data_re;
  delete[] amp_func_data_im;
}

// Reads amplitude function data from h5file 'filename.' Assumes real and imaginary components
// of 'dataset' exist with '.re' and '.im' extensions.
template <class T>
void fields::add_volume_source(component c, const src_time &src, const volume &where_,
                               const char *filename, const char *dataset, complex<double> amp) {

  meep::h5file eps_file(filename, meep::h5file::READONLY, false);
  int rank;
  std::string dataset_re = std::string(dataset) + ".re";
  std::string dataset_im = std::string(dataset) + ".im";

  size_t re_dims[] = {1, 1, 1};
  T *real_data = (T *)eps_file.read(dataset_re.c_str(), &rank, re_dims, 3, sizeof(T) == sizeof(float));
  if (verbosity > 0)
    master_printf("read in %zdx%zdx%zd amplitude function file \"%s:%s\"\n", re_dims[0], re_dims[1],
                  re_dims[2], filename, dataset_re.c_str());

  size_t im_dims[] = {1, 1, 1};
  T *imag_data = (T *)eps_file.read(dataset_im.c_str(), &rank, im_dims, 3, sizeof(T) == sizeof(float));
  if (verbosity > 0)
    master_printf("read in %zdx%zdx%zd amplitude function file \"%s:%s\"\n", im_dims[0], im_dims[1],
                  im_dims[2], filename, dataset_im.c_str());

  for (int i = 0; i < 3; ++i) {
    if (re_dims[i] != im_dims[i]) {
      abort("Imaginary and real datasets have different dimensions");
    }
  }

  size_t total_size = re_dims[0] * re_dims[1] * re_dims[2];
  complex<double> *amp_data = new complex<double>[total_size];
  for (size_t i = 0; i < total_size; ++i) {
    amp_data[i] = complex<double>(real_data[i], imag_data[i]);
  }

  add_volume_source(c, src, where_, amp_data, re_dims[0], re_dims[1], re_dims[2], amp);

  delete[] real_data;
  delete[] imag_data;
  delete[] amp_data;
}

void fields::add_volume_source(component c, const src_time &src, const volume &where_,
                               complex<double> A(const vec &), complex<double> amp) {
  volume where(where_); // make a copy to adjust size if necessary
  if (gv.dim != where.dim)
    abort("incorrect source grid_volume dimensionality in add_volume_source");
  LOOP_OVER_DIRECTIONS(gv.dim, d) {
    double w = user_volume.boundary_location(High, d) - user_volume.boundary_location(Low, d);
    if (where.in_direction(d) > w + gv.inva)
      abort("Source width > cell width in %s direction!\n", direction_name(d));
    else if (where.in_direction(d) > w) { // difference is less than 1 pixel
      double dw = where.in_direction(d) - w;
      where.set_direction_min(d, where.in_direction_min(d) - dw * 0.5);
      where.set_direction_max(d, where.in_direction_min(d) + w);
    }
  }

  src_vol_chunkloop_data data;
  data.A = A ? A : one;
  data.amp = amp;
  LOOP_OVER_DIRECTIONS(gv.dim, d) {
    if (where.in_direction(d) == 0.0 && !nosize_direction(d)) // delta-fun
      data.amp *= gv.a; // correct units for J delta-function amplitude
  }
  sources = src.add_to(sources, &data.src);
  data.center = (where.get_min_corner() + where.get_max_corner()) * 0.5;
  loop_in_chunks(src_vol_chunkloop, (void *)&data, where, c, false);
  require_component(c);
}

/***************************************************************/
/* helper routine for add_eigenmode_source that calls          */
/* add_volume_source only if certain conditions are met        */
/***************************************************************/
void fields::add_volume_source_check(component c, const src_time &src, const volume &where,
                                     complex<double> A(const vec &), complex<double> amp, component c0,
                                     direction d, int has_tm, int has_te) {
  if (!gv.has_field(c)) return;
  if (c0 != Centered && c0 != c) return;
  if (component_direction(c) == d) return;
  if (gv.dim == D2) // parity checks
  {
    if (has_te && is_tm(c)) return;
    if (has_tm && !is_tm(c)) return;
  };
  add_volume_source(c, src, where, A, amp);
}

static gaussianbeam *global_gaussianbeam_object = 0;
static component global_gaussianbeam_component;

static std::complex<double> gaussianbeam_ampfunc(const vec &p) {
  std::complex<double> EH[6];
  global_gaussianbeam_object->get_fields(EH, p);
  switch (global_gaussianbeam_component) {
  case Ex: return EH[0];
  case Ey: return EH[1];
  case Ez: return EH[2];
  case Hx: return EH[3];
  case Hy: return EH[4];
  case Hz: return EH[5];
  default: abort("invalid component in gaussianbeam_ampfunc");
  }
}

void fields::add_volume_source(const src_time &src, const volume &where, gaussianbeam beam) {
  global_gaussianbeam_object = &beam;
  direction d = normal_direction(where);
  component cE[3] = {Ex, Ey, Ez}, cH[3] = {Hx, Hy, Hz};
  int n = (d == X ? 0 : (d == Y ? 1 : 2));
  if (d == NO_DIRECTION) {
    n = where.in_direction(X) == 0
            ? 0
            : where.in_direction(Y) == 0 ? 1 : where.in_direction(Z) == 0 ? 2 : -1;
    if (n == -1)
      abort(
          "can't determine source direction for non-empty source volume with NO_DIRECTION source");
  }
  bool has_tm = abs(beam.get_E0(2)) > 0;
  bool has_te = abs(beam.get_E0(0)) > 0 || abs(beam.get_E0(1)) > 0;
  int np1 = (n + 1) % 3;
  int np2 = (n + 2) % 3;
  // Kx = -Hy, Ky = Hx   (for d==Z)
  global_gaussianbeam_component = cH[np1];
  add_volume_source_check(cE[np2], src, where, gaussianbeam_ampfunc, +1.0, cE[np2], d,
                          has_tm, has_te);
  global_gaussianbeam_component = cH[np2];
  add_volume_source_check(cE[np1], src, where, gaussianbeam_ampfunc, -1.0, cE[np1], d,
                          has_tm, has_te);
  // Nx = +Ey, Ny = -Ex  (for d==Z)
  global_gaussianbeam_component = cE[np1];
  add_volume_source_check(cH[np2], src, where, gaussianbeam_ampfunc, -1.0, cH[np2], d,
                          has_tm, has_te);
  global_gaussianbeam_component = cE[np2];
  add_volume_source_check(cH[np1], src, where, gaussianbeam_ampfunc, +1.0, cH[np1], d,
                          has_tm, has_te);
}

gaussianbeam::gaussianbeam(const vec &x0_, const vec &kdir_, double w0_, double freq_,
                           double eps_, double mu_, std::complex<double> E0_[3]) {
  if (x0_.dim == Dcyl) abort("wrong dimensionality in gaussianbeam");

  x0 = x0_;
  kdir = kdir_;
  w0 = w0_;
  freq = freq_;
  eps = eps_;
  mu = mu_;
  for (int j = 0; j < 3; ++j)
    E0[j] = E0_[j];
}

/* Compute the E/H fields EH[6] (6 components) at point x in 3d.

   Adapted from code by M. T. Homer Reid in his SCUFF-EM package:
   https://github.com/HomerReid/scuff-em/blob/master/libs/libIncField/GaussianBeam.cc
*/

/* Copyright (C) 2005-2011 M. T. Homer Reid
 *
 * This file is part of SCUFF-EM.
 *
 * SCUFF-EM is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * SCUFF-EM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

/*
 * GaussianBeam.cc -- routine for computing the electric and magnetic
 *                    fields of a focused gaussian beam
 *
 * homer reid        -- 4/2011
 *
 * new approach:
 * johannes feist    -- 11/2011
 *
 * note: this calculation follows CJR Sheppard & S Saghafi, J Opt Soc Am A 16, 1381,
 * approximating a Gaussian beam as the field of the sum of an
 * x-polarized electric and y-polarized magnetic dipole, but located at
 * the _complex_ source point X = (0,0,i z0).
 * This produces a field that looks like a Gaussian beam in paraxial approximation
 * and exactly solves the field-free Maxwell equations everywhere in (real) space.
 *
 * note that the core of the calculation is carried out in a coordinate
 * system in which the beam propagates in the +z direction and is
 * polarized such that the E-field is (mostly) in the +x direction. after
 * carrying out the computation described in the memo to obtain the field
 * components in this coordinate system, we then rotate the field components
 * as appropriate for the user's specified propagation and polarization
 * vectors.
 */

void gaussianbeam::get_fields(std::complex<double> *EH, const vec &x) const {
  double n = sqrt(eps * mu);
  double k = 2 * pi * freq * n;
  double ZR = sqrt(mu/eps);
  double z0 = k * w0 * w0 / 2;
  double kz0 = k*z0;
  vec xrel = x - x0;

  vec zhat = kdir / abs(kdir);
  double rho = abs(vec(zhat.y()*xrel.z()-zhat.z()*xrel.y(),
                       zhat.z()*xrel.x()-zhat.x()*xrel.z(),
                       zhat.x()*xrel.y()-zhat.y()*xrel.x()));
  double zhatdotxrel = zhat & xrel;

  bool UseRescaledFG = false;

  complex<double> zc = complex<double>(zhatdotxrel,-z0);
  complex<double> Rsq = rho*rho + zc*zc;
  complex<double> R = sqrt(Rsq);
  complex<double> kR = k*R, kR2 = kR*kR, kR3 = kR2*kR;
  complex<double> f,g,fmgbRsq;
  if (abs(kR) > 1e-4) {
    complex<double> coskR, sinkR;
    if (fabs(imag(kR)) > 30.0) {
      UseRescaledFG = true;
      complex<double> ExpI = exp(complex<double>(0,1.0)*real(kR));
      complex<double> ExpPlus = exp(imag(kR)-kz0);
      complex<double> ExpMinus = exp(-(imag(kR)+kz0));
      coskR = 0.5*(ExpI*ExpMinus + conj(ExpI)*ExpPlus);
      sinkR = -0.5*complex<double>(0,1.0)*(ExpI*ExpMinus - conj(ExpI)*ExpPlus);
    } else {
      coskR = cos(kR);
      sinkR = sin(kR);
    }
    f = -3.0 * (coskR/kR2 - sinkR/kR3);
    g = 1.5 * (sinkR/kR + coskR/kR2 - sinkR/kR3);
    fmgbRsq = (f-g)/Rsq;
  } else {
    complex<double> kR4 = kR2*kR2;
    // Taylor series expansion for small R
    f = kR4/280.0 - kR2/10.0 + 1.0;
    g = 3.0*kR4/280.0 - kR2/5.0 + 1.0;
    fmgbRsq = (kR4/5040.0 - kR2/140.0 + 0.1)*(k*k);
  }
  complex<double> i2fk = 0.5*complex<double>(0,1)*f*k;

  complex<double> E[3], H[3];
  for (int j=0; j<3; ++j) {
    E[j] = complex<double>(0,0);
    H[j] = complex<double>(0,0);
  }

  double rnorm = sqrt(real(E0[0])*real(E0[0])+real(E0[1])*real(E0[1])+real(E0[2])*real(E0[2]));
  if (rnorm > 1e-13) {
    vec xhat = vec(real(E0[0]),real(E0[1]),real(E0[2])) / rnorm;
    vec yhat = vec(zhat.y()*xhat.z()-zhat.z()*xhat.y(),
                   zhat.z()*xhat.x()-zhat.x()*xhat.z(),
                   zhat.x()*xhat.y()-zhat.y()*xhat.x());
    double xhatdotxrel = xhat & xrel;
    double yhatdotxrel = yhat & xrel;

    complex<double> gb_Ex = g + fmgbRsq * xhatdotxrel * xhatdotxrel + i2fk * zc;
    complex<double> gb_Ey = fmgbRsq * xhatdotxrel * yhatdotxrel;
    complex<double> gb_Ez = fmgbRsq * xhatdotxrel * zc - i2fk * xhatdotxrel;
    complex<double> gb_Hx = EH[1];
    complex<double> gb_Hy = g + fmgbRsq * yhatdotxrel * yhatdotxrel + i2fk * zc;
    complex<double> gb_Hz = fmgbRsq * yhatdotxrel * zc - i2fk * yhatdotxrel;

    E[0] += rnorm * (gb_Ex * xhat.x() + gb_Ey * yhat.x() + gb_Ez * zhat.x());
    E[1] += rnorm * (gb_Ex * xhat.y() + gb_Ey * yhat.y() + gb_Ez * zhat.y());
    E[2] += rnorm * (gb_Ex * xhat.z() + gb_Ey * yhat.z() + gb_Ez * zhat.z());
    H[0] += rnorm * (gb_Hx * xhat.x() + gb_Hy * yhat.x() + gb_Hz * zhat.x());
    H[1] += rnorm * (gb_Hx * xhat.y() + gb_Hy * yhat.y() + gb_Hz * zhat.y());
    H[2] += rnorm * (gb_Hx * xhat.z() + gb_Hy * yhat.z() + gb_Hz * zhat.z());
  }

  double inorm = sqrt(imag(E0[0])*imag(E0[0])+imag(E0[1])*imag(E0[1])+imag(E0[2])*imag(E0[2]));
  if (inorm > 1e-13) {
    vec xhat = vec(imag(E0[0]),imag(E0[1]),imag(E0[2])) / inorm;
    vec yhat = vec(zhat.y()*xhat.z()-zhat.z()*xhat.y(),
                   zhat.z()*xhat.x()-zhat.x()*xhat.z(),
                   zhat.x()*xhat.y()-zhat.y()*xhat.x());
    double xhatdotxrel = xhat & xrel;
    double yhatdotxrel = yhat & xrel;

    complex<double> gb_Ex = g + fmgbRsq * xhatdotxrel * xhatdotxrel + i2fk * zc;
    complex<double> gb_Ey = fmgbRsq * xhatdotxrel * yhatdotxrel;
    complex<double> gb_Ez = fmgbRsq * xhatdotxrel * zc - i2fk * xhatdotxrel;
    complex<double> gb_Hx = EH[1];
    complex<double> gb_Hy = g + fmgbRsq * yhatdotxrel * yhatdotxrel + i2fk * zc;
    complex<double> gb_Hz = fmgbRsq * yhatdotxrel * zc - i2fk * yhatdotxrel;

    E[0] += complex<double>(0,1.0) * inorm * (gb_Ex * xhat.x() + gb_Ey * yhat.x() + gb_Ez * zhat.x());
    E[1] += complex<double>(0,1.0) * inorm * (gb_Ex * xhat.y() + gb_Ey * yhat.y() + gb_Ez * zhat.y());
    E[2] += complex<double>(0,1.0) * inorm * (gb_Ex * xhat.z() + gb_Ey * yhat.z() + gb_Ez * zhat.z());
    H[0] += complex<double>(0,1.0) * inorm * (gb_Hx * xhat.x() + gb_Hy * yhat.x() + gb_Hz * zhat.x());
    H[1] += complex<double>(0,1.0) * inorm * (gb_Hx * xhat.y() + gb_Hy * yhat.y() + gb_Hz * zhat.y());
    H[2] += complex<double>(0,1.0) * inorm * (gb_Hx * xhat.z() + gb_Hy * yhat.z() + gb_Hz * zhat.z());
  }

  double Eorig;
  if (UseRescaledFG)
    Eorig = 3.0/(2*kz0*kz0*kz0) * (kz0*(kz0-1) + 0.5*(1.0-exp(-2.0*kz0)));
  else
    Eorig = 3.0/(2*kz0*kz0*kz0) * (exp(kz0)*kz0*(kz0-1) + sinh(kz0));

  EH[0] = E[0] / Eorig;
  EH[1] = E[1] / Eorig;
  EH[2] = E[2] / Eorig;
  EH[3] = H[0] / (Eorig*ZR);
  EH[4] = H[1] / (Eorig*ZR);
  EH[5] = H[2] / (Eorig*ZR);
}

diffractedplanewave::diffractedplanewave(int g_[3], double axis_[3], std::complex<double> s_, std::complex<double> p_) {
  for (int j = 0; j < 3; ++j) {
    g[j] = g_[j];
    axis[j] = axis_[j];
  }
  s = s_;
  p = p_;
};

} // namespace meep
