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

#include <string.h>

#include "meep.hpp"
#include "meep_internals.hpp"

using namespace std;

namespace meep {

void fields::update_eh(field_type ft, bool skip_w_components) {
  if (ft != E_stuff && ft != H_stuff) abort("update_eh only works with E/H");
  for (int i = 0; i < num_chunks; i++)
    if (chunks[i]->is_mine())
      if (chunks[i]->update_eh(ft, skip_w_components))
        chunk_connections_valid = false; // E/H allocated - reconnect chunks
}

bool fields_chunk::needs_W_prev(component c) const {
  for (susceptibility *chiP = s->chiP[type(c)]; chiP; chiP = chiP->next)
    if (chiP->needs_W_prev()) return true;
  return false;
}

template <class T>
bool fields_chunk::update_eh(field_type ft, bool skip_w_components) {
  field_type ft2 = ft == E_stuff ? D_stuff : B_stuff; // for sources etc.
  bool allocated_eh = false;

  bool have_int_sources = false;
  if (!doing_solve_cw) {
    for (src_vol *sv = sources[ft2]; sv; sv = sv->next)
      if (sv->t->is_integrated) {
        have_int_sources = true;
        break;
      }
  }

  FOR_FT_COMPONENTS(ft, ec) {
    component dc = field_type_component(ft2, ec);
    DOCMP {
      bool need_fmp = false;
      if (f[ec][cmp]) {
        need_fmp = have_int_sources;
        for (polarization_state *p = pol[ft]; p && !need_fmp; p = p->next)
          need_fmp = need_fmp || p->s->needs_P(ec, cmp, f);
      }
      if (need_fmp) {
        if (!f_minus_p[dc][cmp]) f_minus_p[dc][cmp] = new T[gv.ntot()];
      }
      else if (f_minus_p[dc][cmp]) { // remove unneeded f_minus_p
        delete[] f_minus_p[dc][cmp];
        f_minus_p[dc][cmp] = 0;
      }
    }
  }
  bool have_f_minus_p = false;
  FOR_FT_COMPONENTS(ft2, dc) {
    if (f_minus_p[dc][0]) {
      have_f_minus_p = true;
      break;
    }
  }

  const size_t ntot = s->gv.ntot();

  if (have_f_minus_p && doing_solve_cw)
    abort("dispersive materials are not yet implemented for solve_cw");

  //////////////////////////////////////////////////////////////////////////
  // First, initialize f_minus_p to D - P, if necessary

  FOR_FT_COMPONENTS(ft, ec) if (f[ec][0]) {
    component dc = field_type_component(ft2, ec);
    DOCMP if (f_minus_p[dc][cmp]) {
      T *fmp = f_minus_p[dc][cmp];
      memcpy(fmp, f[dc][cmp], sizeof(T) * ntot);
    }
  }

  for (polarization_state *p = pol[ft]; p; p = p->next)
    if (p->data) p->s->subtract_P(ft, f_minus_p, p->data);

  //////////////////////////////////////////////////////////////////////////
  // Next, subtract time-integrated sources (i.e. polarizations, not currents)

  if (have_f_minus_p && !doing_solve_cw) {
    for (src_vol *sv = sources[ft2]; sv; sv = sv->next) {
      if (sv->t->is_integrated && f[sv->c][0] && ft == type(sv->c)) {
        component c = field_type_component(ft2, sv->c);
        for (size_t j = 0; j < sv->npts; ++j) {
          const complex<double> A = sv->dipole(j);
          DOCMP { f_minus_p[c][cmp][sv->index[j]] -= (cmp) ? imag(A) : real(A); }
        }
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////
  // Finally, compute E = chi1inv * D

  T *dmp[NUM_FIELD_COMPONENTS][2];
  FOR_FT_COMPONENTS(ft2, dc) DOCMP2 {
    dmp[dc][cmp] = f_minus_p[dc][cmp] ? f_minus_p[dc][cmp] : f[dc][cmp];
  }

  DOCMP FOR_FT_COMPONENTS(ft, ec) {
    if (f[ec][cmp]) {
      if (type(ec) != ft) abort("bug in FOR_FT_COMPONENTS");
      component dc = field_type_component(ft2, ec);
      const direction d_ec = component_direction(ec);
      const ptrdiff_t s_ec = gv.stride(d_ec) * (ft == H_stuff ? -1 : +1);
      const direction d_1 = cycle_direction(gv.dim, d_ec, 1);
      const component dc_1 = direction_component(dc, d_1);
      const ptrdiff_t s_1 = gv.stride(d_1) * (ft == H_stuff ? -1 : +1);
      const direction d_2 = cycle_direction(gv.dim, d_ec, 2);
      const component dc_2 = direction_component(dc, d_2);
      const ptrdiff_t s_2 = gv.stride(d_2) * (ft == H_stuff ? -1 : +1);

      direction dsigw0 = d_ec;
      direction dsigw = s->sigsize[dsigw0] > 1 ? dsigw0 : NO_DIRECTION;

      // lazily allocate any E/H fields that are needed (H==B initially)
      if (f[ec][cmp] == f[dc][cmp] &&
          (s->chi1inv[ec][d_ec] || have_f_minus_p || dsigw != NO_DIRECTION)) {
        f[ec][cmp] = new T[gv.ntot()];
        memcpy(f[ec][cmp], f[dc][cmp], gv.ntot() * sizeof(T));
        allocated_eh = true;
      }

      // lazily allocate W auxiliary field
      if (!f_w[ec][cmp] && dsigw != NO_DIRECTION) {
        f_w[ec][cmp] = new T[gv.ntot()];
        memcpy(f_w[ec][cmp], f[ec][cmp], gv.ntot() * sizeof(T));
        if (needs_W_notowned(ec)) allocated_eh = true; // communication needed
      }

      // for solve_cw, when W exists we get W and E from special variables
      if (f_w[ec][cmp] && skip_w_components) continue;

      // save W field from this timestep in f_w_prev if needed by pols
      if (needs_W_prev(ec)) {
        if (!f_w_prev[ec][cmp]) f_w_prev[ec][cmp] = new T[gv.ntot()];
        memcpy(f_w_prev[ec][cmp], f_w[ec][cmp] ? f_w[ec][cmp] : f[ec][cmp],
               sizeof(T) * gv.ntot());
      }

      if (f[ec][cmp] != f[dc][cmp])
        STEP_UPDATE_EDHB(f[ec][cmp], ec, gv, dmp[dc][cmp], dmp[dc_1][cmp], dmp[dc_2][cmp],
                         s->chi1inv[ec][d_ec], dmp[dc_1][cmp] ? s->chi1inv[ec][d_1] : NULL,
                         dmp[dc_2][cmp] ? s->chi1inv[ec][d_2] : NULL, s_ec, s_1, s_2, s->chi2[ec],
                         s->chi3[ec], f_w[ec][cmp], dsigw, s->sig[dsigw], s->kap[dsigw]);
    }
  }

  /* Do annoying special cases for r=0 in cylindrical coords.  Note
     that this only really matters for field output; the Ez and Ep
     components at r=0 don't usually affect the fields elsewhere
     because of the form of Maxwell's equations in cylindrical coords. */
  // (FIXME: handle Kerr case?  Do we care about auxiliary PML fields here?)
  if (gv.dim == Dcyl && gv.origin_r() == 0.0) DOCMP FOR_FT_COMPONENTS(ft, ec) {
      if (f[ec][cmp] && (ec == Ep || ec == Ez || ec == Hr)) {
        component dc = field_type_component(ft2, ec);
        if (f[ec][cmp] == f[dc][cmp]) continue;
        const int yee_idx = gv.yee_index(ec);
        const int d_ec = component_direction(ec);
        const int sR = gv.stride(R), nZ = gv.num_direction(Z);
        T *E = f[ec][cmp];
        const T *D = f_minus_p[dc][cmp] ? f_minus_p[dc][cmp] : f[dc][cmp];
        const T *chi1inv = s->chi1inv[ec][d_ec];
        if (chi1inv)
          for (int iZ = 0; iZ < nZ; iZ++) {
            const int i = yee_idx + iZ - sR;
            E[i] = chi1inv[i] * D[i];
          }
        else
          for (int iZ = 0; iZ < nZ; iZ++) {
            const int i = yee_idx + iZ - sR;
            E[i] = D[i];
          }
      }
    }

  return allocated_eh;
}

} // namespace meep
