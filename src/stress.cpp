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

/* Computation of the force spectrum via integration of the Maxwell
   stress tensor of the Fourier-transformed fields */

#include <meep.hpp>

using namespace std;

namespace meep {

dft_force::dft_force(dft_chunk *offdiag1_, dft_chunk *offdiag2_, dft_chunk *diag_, double fmin,
                     double fmax, int Nf, const volume &where_)
    : where(where_) {
  freq = meep::linspace(fmin, fmax, Nf);
  offdiag1 = offdiag1_;
  offdiag2 = offdiag2_;
  diag = diag_;
  // where = new volume(where_.get_min_corner(), where_.get_max_corner());
}

dft_force::dft_force(dft_chunk *offdiag1_, dft_chunk *offdiag2_, dft_chunk *diag_,
                     const std::vector<double> freq_, const volume &where_)
    : where(where_) {
  freq = freq_;
  offdiag1 = offdiag1_;
  offdiag2 = offdiag2_;
  diag = diag_;
  // where = new volume(where_.get_min_corner(), where_.get_max_corner());
}

dft_force::dft_force(dft_chunk *offdiag1_, dft_chunk *offdiag2_, dft_chunk *diag_,
                     const double *freq_, size_t Nfreq, const volume &where_)
    : freq(Nfreq), where(where_) {
  for (size_t i = 0; i < Nfreq; ++i)
    freq[i] = freq_[i];
  offdiag1 = offdiag1_;
  offdiag2 = offdiag2_;
  diag = diag_;
  // where = new volume(where_.get_min_corner(), where_.get_max_corner());
}

dft_force::dft_force(const dft_force &f) : where(f.where) {
  freq = f.freq;
  offdiag1 = f.offdiag1;
  offdiag2 = f.offdiag2;
  diag = f.diag;
  // where = new volume(f.where->get_min_corner(), f.where->get_max_corner());
}

void dft_force::remove() {
  while (offdiag1) {
    dft_chunk *nxt = offdiag1->next_in_dft;
    delete offdiag1;
    offdiag1 = nxt;
  }
  while (offdiag2) {
    dft_chunk *nxt = offdiag2->next_in_dft;
    delete offdiag2;
    offdiag2 = nxt;
  }
  while (diag) {
    dft_chunk *nxt = diag->next_in_dft;
    delete diag;
    diag = nxt;
  }
}

void dft_force::operator-=(const dft_force &st) {
  if (offdiag1 && st.offdiag1) *offdiag1 -= *st.offdiag1;
  if (offdiag2 && st.offdiag2) *offdiag2 -= *st.offdiag2;
  if (diag && st.diag) *diag -= *st.diag;
}

static void stress_sum(size_t Nfreq, double *F, const dft_chunk *F1, const dft_chunk *F2) {
  for (const dft_chunk *curF1 = F1, *curF2 = F2; curF1 && curF2;
       curF1 = curF1->next_in_dft, curF2 = curF2->next_in_dft) {
    complex<double> extra_weight(real(curF1->extra_weight), imag(curF1->extra_weight));
    for (size_t k = 0; k < curF1->N; ++k)
      for (size_t i = 0; i < Nfreq; ++i)
        F[i] += real(extra_weight *
                     complex<double>(curF1->dft[k * Nfreq + i]) *
                     conj(complex<double>(curF2->dft[k * Nfreq + i])));
  }
}

double *dft_force::force() {
  const size_t Nfreq = freq.size();
  double *F = new double[Nfreq];
  for (size_t i = 0; i < Nfreq; ++i)
    F[i] = 0;

  stress_sum(Nfreq, F, offdiag1, offdiag2);
  stress_sum(Nfreq, F, diag, diag);

  double *Fsum = new double[Nfreq];
  sum_to_all(F, Fsum, int(Nfreq));
  delete[] F;
  return Fsum;
}

void dft_force::save_hdf5(h5file *file, const char *dprefix) {
  save_dft_hdf5(offdiag1, "offdiag1", file, dprefix);
  file->prevent_deadlock(); // hackery
  save_dft_hdf5(offdiag2, "offdiag2", file, dprefix);
  file->prevent_deadlock(); // hackery
  save_dft_hdf5(diag, "diag", file, dprefix);
}

void dft_force::load_hdf5(h5file *file, const char *dprefix) {
  load_dft_hdf5(offdiag1, "offdiag1", file, dprefix);
  file->prevent_deadlock(); // hackery
  load_dft_hdf5(offdiag2, "offdiag2", file, dprefix);
  file->prevent_deadlock(); // hackery
  load_dft_hdf5(diag, "diag", file, dprefix);
}

void dft_force::save_hdf5(fields &f, const char *fname, const char *dprefix, const char *prefix) {
  h5file *ff = f.open_h5file(fname, h5file::WRITE, prefix);
  save_hdf5(ff, dprefix);
  delete ff;
}

void dft_force::load_hdf5(fields &f, const char *fname, const char *dprefix, const char *prefix) {
  h5file *ff = f.open_h5file(fname, h5file::READONLY, prefix);
  load_hdf5(ff, dprefix);
  delete ff;
}

void dft_force::scale_dfts(complex<double> scale) {
  if (offdiag1) offdiag1->scale_dft(scale);
  if (offdiag2) offdiag2->scale_dft(scale);
  if (diag) diag->scale_dft(scale);
}

/* note that the components where->c indicate the direction of the
   force to be computed, so they should be vector components (such as
   Ex, Ey, ... or Sx, ...)  rather than pseudovectors (like Hx, ...). */
dft_force fields::add_dft_force(const volume_list *where_, const double *freq, size_t Nfreq) {
  dft_chunk *offdiag1 = 0, *offdiag2 = 0, *diag = 0;

  volume_list *where = S.reduce(where_);
  volume_list *where_save = where;
  volume everywhere = where->v;

  for (; where; where = where->next) {
    direction nd = normal_direction(where->v);
    if (nd == NO_DIRECTION) meep::abort("cannot determine dft_force normal");
    direction fd = component_direction(where->c); // force direction
    if (fd == NO_DIRECTION) meep::abort("NO_DIRECTION dft_force is invalid");
    if (coordinate_mismatch(gv.dim, fd)) meep::abort("coordinate-type mismatch in add_dft_force");

    if (fd != nd) { // off-diagaonal stress-tensor terms
      offdiag1 = add_dft(direction_component(Ex, fd), where->v, freq, Nfreq, true, where->weight,
                         offdiag1);
      offdiag2 = add_dft(direction_component(Ex, nd), where->v, freq, Nfreq, false, 1.0, offdiag2);
      offdiag1 = add_dft(direction_component(Hx, fd), where->v, freq, Nfreq, true, where->weight,
                         offdiag1);
      offdiag2 = add_dft(direction_component(Hx, nd), where->v, freq, Nfreq, false, 1.0, offdiag2);
    }
    else // diagonal stress-tensor terms
      LOOP_OVER_FIELD_DIRECTIONS(gv.dim, d) {
        complex<double> weight1 = where->weight * (d == fd ? +0.5 : -0.5);
        diag = add_dft(direction_component(Ex, d), where->v, freq, Nfreq, true, 1.0, diag, true,
                       weight1, false);
        diag = add_dft(direction_component(Hx, d), where->v, freq, Nfreq, true, 1.0, diag, true,
                       weight1, false);
      }
    everywhere = everywhere | where->v;
  }

  delete where_save;
  return dft_force(offdiag1, offdiag2, diag, freq, Nfreq, everywhere);
}

} // namespace meep
