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
#include <assert.h>

#include "meep.hpp"
#include "meep_internals.hpp"

#define RESTRICT

using namespace std;

namespace meep {

void fields::step_db(field_type ft) {
  for (int i = 0; i < num_chunks; i++)
    if (chunks[i]->is_mine())
      if (chunks[i]->step_db(ft)) {
       chunk_connections_valid = false;
       assert(changed_materials);
      }
}

bool fields_chunk::step_db(field_type ft) {
  bool allocated_u = false;

  if (ft != B_stuff && ft != D_stuff) meep::abort("bug - step_db should only be called for B or D");

  for (size_t i = 0; i < gvs.size(); ++i) {
    grid_volume gv_sub = gvs[i];

    DOCMP FOR_FT_COMPONENTS(ft, cc) {
      if (f[cc][cmp]) {
        const component c_p = plus_component[cc], c_m = minus_component[cc];
        const direction d_deriv_p = plus_deriv_direction[cc];
        const direction d_deriv_m = minus_deriv_direction[cc];
        const direction d_c = component_direction(cc);
        const bool have_p = have_plus_deriv[cc];
        const bool have_m = have_minus_deriv[cc];
        const direction dsig0 = cycle_direction(gv_sub.dim, d_c, 1);
        const direction dsig = s->sigsize[dsig0] > 1 ? dsig0 : NO_DIRECTION;
        const direction dsigu0 = cycle_direction(gv_sub.dim, d_c, 2);
        const direction dsigu = s->sigsize[dsigu0] > 1 ? dsigu0 : NO_DIRECTION;
        ptrdiff_t stride_p = have_p ? gv_sub.stride(d_deriv_p) : 0;
        ptrdiff_t stride_m = have_m ? gv_sub.stride(d_deriv_m) : 0;
        realnum *f_p = have_p ? f[c_p][cmp] : NULL;
        realnum *f_m = have_m ? f[c_m][cmp] : NULL;
        realnum *the_f = f[cc][cmp];

        if (dsig != NO_DIRECTION && s->conductivity[cc][d_c] && !f_cond[cc][cmp]) {
          f_cond[cc][cmp] = new realnum[gv_sub.ntot()];
          memset(f_cond[cc][cmp], 0, sizeof(realnum) * gv_sub.ntot());
        }
        if (dsigu != NO_DIRECTION && !f_u[cc][cmp]) {
          f_u[cc][cmp] = new realnum[gv_sub.ntot()];
          memcpy(f_u[cc][cmp], the_f, gv_sub.ntot() * sizeof(realnum));
          allocated_u = true;
        }

        if (ft == D_stuff) { // strides are opposite sign for H curl
          stride_p = -stride_p;
          stride_m = -stride_m;
        }

        if (gv_sub.dim == Dcyl) switch (d_c) {
          case R:
            f_p = NULL; // im/r Fz term will be handled separately
            break;
          case P: break; // curl works normally for phi component
          case Z: {
            f_m = NULL; // im/r Fr term will be handled separately

            /* Here we do a somewhat cool hack: the update of the z
               component gives a 1/r d(r Fp)/dr term, rather than
               just the derivative dg/dr expected in step_curl.
               Rather than duplicating all of step_curl to handle
               this bloody derivative, however, we define a new
               array f_rderiv_int which is the integral of 1/r d(r Fp)/dr,
               so that we can pass it to the unmodified step_curl
               and get the correct derivative.  (More precisely,
               the derivative and integral are replaced by differences
               and sums, but you get the idea). */
            if (!f_rderiv_int) f_rderiv_int = new realnum[gv_sub.ntot()];
            realnum ir0 = gv_sub.origin_r() * gv_sub.a + 0.5 * gv_sub.iyee_shift(c_p).in_direction(R);
            for (int iz = 0; iz <= gv_sub.nz(); ++iz)
              f_rderiv_int[iz] = 0;
            int sr = gv_sub.nz() + 1;
            for (int ir = 1; ir <= gv_sub.nr(); ++ir) {
              realnum rinv = 1.0 / ((ir + ir0) - 0.5);
              for (int iz = 0; iz <= gv_sub.nz(); ++iz) {
                ptrdiff_t idx = ir * sr + iz;
                f_rderiv_int[idx] =
                  f_rderiv_int[idx - sr] +
                  rinv * (f_p[idx] * (ir + ir0) - f_p[idx - sr] * ((ir - 1) + ir0));
              }
            }
            f_p = f_rderiv_int;
            break;
          }
          default: meep::abort("bug - non-cylindrical field component in Dcyl");
          }

        STEP_CURL(the_f, cc, f_p, f_m, stride_p, stride_m,
                  gv_sub, gv_sub.little_owned_corner0(cc), gv_sub.big_corner(),
                  Courant, dsig, s->sig[dsig],
                  s->kap[dsig], s->siginv[dsig], f_u[cc][cmp], dsigu, s->sig[dsigu], s->kap[dsigu],
                  s->siginv[dsigu], dt, s->conductivity[cc][d_c], s->condinv[cc][d_c],
                  f_cond[cc][cmp]);
      }
    }
  }

  /* In 2d with beta != 0, add beta terms.  This is a trick to model
     an exp(i beta z) z-dependence but without requiring a "3d"
     calculation and without requiring complex fields.  Looking at the
     z=0 2d cross-section, the exp(i beta z) term adds an i \beta
     \hat{z} \times cross-product to the curls, which couples the TE
     and TM polarizations.  However, to avoid complex fields, in the
     case of real fields we implicitly store i*(TM fields) rather than
     the TM fields, in which case the i's cancel in the update
     equations.  (Mathematically, this is equivalent to looking at the
     superposition of the fields at beta and the timereversed fields
     at -beta.)  The nice thing about this is that most calculations
     of flux, energy, etcetera, are insensitive to this implicit "i"
     factor.   For complex fields, we implement i*beta directly. */
  if (gv.dim == D2 && beta != 0) DOCMP for (direction d_c = X; d_c <= Y; d_c = direction(d_c + 1)) {
      component cc = direction_component(first_field_component(ft), d_c);
      component c_g = direction_component(ft == D_stuff ? Hx : Ex, d_c == X ? Y : X);
      realnum *the_f = f[cc][cmp];
      const realnum *g = f[c_g][1 - cmp] ? f[c_g][1 - cmp] : f[c_g][cmp];
      const direction dsig0 = cycle_direction(gv.dim, d_c, 1);
      const direction dsig = s->sigsize[dsig0] > 1 ? dsig0 : NO_DIRECTION;
      const direction dsigu0 = cycle_direction(gv.dim, d_c, 2);
      const direction dsigu = s->sigsize[dsigu0] > 1 ? dsigu0 : NO_DIRECTION;
      const realnum betadt = 2 * pi * beta * dt * (d_c == X ? +1 : -1) *
                            (f[c_g][1 - cmp] ? (ft == D_stuff ? -1 : +1) * (2 * cmp - 1) : 1);
      STEP_BETA(the_f, cc, g, gv, gv.little_owned_corner0(cc), gv.big_corner(),
                betadt, dsig, s->siginv[dsig], f_u[cc][cmp], dsigu,
                s->siginv[dsigu], s->condinv[cc][d_c], f_cond[cc][cmp]);
    }

  // in cylindrical coordinates, we now have to add the i*m/r terms... */
  if (gv.dim == Dcyl && m != 0) DOCMP FOR_FT_COMPONENTS(ft, cc) {
      const direction d_c = component_direction(cc);
      if (f[cc][cmp] && (d_c == R || d_c == Z)) {
        const component c_g = d_c == R ? plus_component[cc] : minus_component[cc];
        const realnum *g = f[c_g][1 - cmp];
        realnum *the_f = f[cc][cmp];
        const realnum *cndinv = s->condinv[cc][d_c];
        realnum *fcnd = f_cond[cc][cmp];
        realnum *fu = f_u[cc][cmp];
        const direction dsig = cycle_direction(gv.dim, d_c, 1);
        const realnum *siginv = s->sigsize[dsig] > 1 ? s->siginv[dsig] : 0;
        const int dk = gv.iyee_shift(cc).in_direction(dsig);
        const direction dsigu = cycle_direction(gv.dim, d_c, 2);
        const realnum *siginvu = s->sigsize[dsigu] > 1 ? s->siginv[dsigu] : 0;
        const int dku = gv.iyee_shift(cc).in_direction(dsigu);
        const realnum the_m =
            m * (1 - 2 * cmp) * (1 - 2 * (ft == B_stuff)) * (1 - 2 * (d_c == R)) * Courant;
        const realnum ir0 = gv.origin_r() * gv.a + 0.5 * gv.iyee_shift(cc).in_direction(R);
        int sr = gv.nz() + 1;

        // 8 special cases of the same loop (sigh):
        if (siginv) {    // PML in f update
          if (siginvu) { // PML + fu
            if (cndinv)  // PML + fu + conductivity
              //////////////////// MOST GENERAL CASE //////////////////////
              for (int ir = ir0 == 0; ir <= gv.nr(); ++ir) {
                realnum rinv = the_m / (ir + ir0);
                for (int iz = 0; iz <= gv.nz(); ++iz) {
                  ptrdiff_t idx = ir * sr + iz;
                  int k = dk + 2 * (dsig == Z ? iz : ir);
                  int ku = dku + 2 * (dsigu == Z ? iz : ir);
                  realnum df, dfcnd = rinv * g[idx] * cndinv[idx];
                  fcnd[idx] += dfcnd;
                  fu[idx] += (df = dfcnd * siginv[k]);
                  the_f[idx] += siginvu[ku] * df;
                }
              }
            /////////////////////////////////////////////////////////////
            else // PML + fu - conductivity
              for (int ir = ir0 == 0; ir <= gv.nr(); ++ir) {
                realnum rinv = the_m / (ir + ir0);
                for (int iz = 0; iz <= gv.nz(); ++iz) {
                  ptrdiff_t idx = ir * sr + iz;
                  int k = dk + 2 * (dsig == Z ? iz : ir);
                  int ku = dku + 2 * (dsigu == Z ? iz : ir);
                  realnum df, dfcnd = rinv * g[idx];
                  fu[idx] += (df = dfcnd * siginv[k]);
                  the_f[idx] += siginvu[ku] * df;
                }
              }
          }
          else {        // PML - fu
            if (cndinv) // PML - fu + conductivity
              for (int ir = ir0 == 0; ir <= gv.nr(); ++ir) {
                realnum rinv = the_m / (ir + ir0);
                for (int iz = 0; iz <= gv.nz(); ++iz) {
                  ptrdiff_t idx = ir * sr + iz;
                  int k = dk + 2 * (dsig == Z ? iz : ir);
                  realnum dfcnd = rinv * g[idx] * cndinv[idx];
                  fcnd[idx] += dfcnd;
                  the_f[idx] += dfcnd * siginv[k];
                }
              }
            else // PML - fu - conductivity
              for (int ir = ir0 == 0; ir <= gv.nr(); ++ir) {
                realnum rinv = the_m / (ir + ir0);
                for (int iz = 0; iz <= gv.nz(); ++iz) {
                  ptrdiff_t idx = ir * sr + iz;
                  int k = dk + 2 * (dsig == Z ? iz : ir);
                  realnum dfcnd = rinv * g[idx];
                  the_f[idx] += dfcnd * siginv[k];
                }
              }
          }
        }
        else {           // no PML in f update
          if (siginvu) { // no PML + fu
            if (cndinv)  // no PML + fu + conductivity
              for (int ir = ir0 == 0; ir <= gv.nr(); ++ir) {
                realnum rinv = the_m / (ir + ir0);
                for (int iz = 0; iz <= gv.nz(); ++iz) {
                  ptrdiff_t idx = ir * sr + iz;
                  int ku = dku + 2 * (dsigu == Z ? iz : ir);
                  realnum df = rinv * g[idx] * cndinv[idx];
                  fu[idx] += df;
                  the_f[idx] += siginvu[ku] * df;
                }
              }
            else // no PML + fu - conductivity
              for (int ir = ir0 == 0; ir <= gv.nr(); ++ir) {
                realnum rinv = the_m / (ir + ir0);
                for (int iz = 0; iz <= gv.nz(); ++iz) {
                  ptrdiff_t idx = ir * sr + iz;
                  int ku = dku + 2 * (dsigu == Z ? iz : ir);
                  realnum df = rinv * g[idx];
                  fu[idx] += df;
                  the_f[idx] += siginvu[ku] * df;
                }
              }
          }
          else {        // no PML - fu
            if (cndinv) // no PML - fu + conductivity
              for (int ir = ir0 == 0; ir <= gv.nr(); ++ir) {
                realnum rinv = the_m / (ir + ir0);
                for (int iz = 0; iz <= gv.nz(); ++iz) {
                  ptrdiff_t idx = ir * sr + iz;
                  the_f[idx] += rinv * g[idx] * cndinv[idx];
                }
              }
            else // no PML - fu - conductivity
              for (int ir = ir0 == 0; ir <= gv.nr(); ++ir) {
                realnum rinv = the_m / (ir + ir0);
                for (int iz = 0; iz <= gv.nz(); ++iz) {
                  ptrdiff_t idx = ir * sr + iz;
                  the_f[idx] += rinv * g[idx];
                }
              }
          }
        }
      }
    }

#define ZERO_Z(array) memset(array, 0, sizeof(realnum) * (nz + 1));

  // deal with annoying r=0 boundary conditions for m=0 and m=1
  if (gv.dim == Dcyl && gv.origin_r() == 0.0) DOCMP {
      const int nz = gv.nz();
      if (m == 0 && ft == D_stuff && f[Dz][cmp]) {
        // d(Dz)/dt = (1/r) * d(r*Hp)/dr
        const realnum *g = f[Hp][cmp];
        const realnum *cndinv = s->condinv[Dz][Z];
        realnum *fcnd = f_cond[Dz][cmp];
        const direction dsig = cycle_direction(gv.dim, Z, 1);
        const realnum *siginv = s->sigsize[dsig] > 1 ? s->siginv[dsig] : 0;
        const int dk = gv.iyee_shift(Dz).in_direction(dsig);
        const direction dsigu = cycle_direction(gv.dim, Z, 2);
        const realnum *siginvu = s->sigsize[dsigu] > 1 ? s->siginv[dsigu] : 0;
        const int dku = gv.iyee_shift(Dz).in_direction(dsigu);
        realnum *fu = siginvu && f_u[Dz][cmp] ? f[Dz][cmp] : 0;
        realnum *the_f = fu ? f_u[Dz][cmp] : f[Dz][cmp];
        for (int iz = 0; iz < nz; ++iz) {
          // Note: old code (prior to Meep 0.2) was missing factor of 4??
          realnum df, dfcnd = g[iz] * (Courant * 4) * (cndinv ? cndinv[iz] : 1);
          if (fcnd) fcnd[iz] += dfcnd;
          the_f[iz] += (df = dfcnd * (siginv ? siginv[dk + 2 * (dsig == Z) * iz] : 1));
          if (fu) fu[iz] += siginvu[dku + 2 * (dsigu == Z) * iz] * df;
        }
        ZERO_Z(f[Dp][cmp]);
        if (f_cond[Dp][cmp]) ZERO_Z(f_cond[Dp][cmp]);
        if (f_u[Dp][cmp]) ZERO_Z(f_u[Dp][cmp]);
      }
      else if (m == 0 && ft == B_stuff && f[Br][cmp]) {
        ZERO_Z(f[Br][cmp]);
        if (f_cond[Br][cmp]) ZERO_Z(f_cond[Br][cmp]);
        if (f_u[Br][cmp]) ZERO_Z(f_u[Br][cmp]);
      }
      else if (fabs(m) == 1) {
        // D_stuff: d(Dp)/dt = d(Hr)/dz - d(Hz)/dr
        // B_stuff: d(Br)/dt = d(Ep)/dz - i*m*Ez/r
        component cc = ft == D_stuff ? Dp : Br;
        direction d_c = component_direction(cc);
        if (!f[cc][cmp]) continue;
        const realnum *f_p = f[ft == D_stuff ? Hr : Ep][cmp];
        const realnum *f_m = ft == D_stuff ? f[Hz][cmp] : (f[Ez][1 - cmp] + (nz + 1));
        const realnum *cndinv = s->condinv[cc][d_c];
        realnum *fcnd = f_cond[cc][cmp];
        const direction dsig = cycle_direction(gv.dim, d_c, 1);
        const realnum *siginv = s->sigsize[dsig] > 1 ? s->siginv[dsig] : 0;
        const int dk = gv.iyee_shift(cc).in_direction(dsig);
        const direction dsigu = cycle_direction(gv.dim, d_c, 2);
        const realnum *siginvu = s->sigsize[dsigu] > 1 ? s->siginv[dsigu] : 0;
        const int dku = gv.iyee_shift(cc).in_direction(dsigu);
        realnum *fu = siginvu && f_u[cc][cmp] ? f[cc][cmp] : 0;
        realnum *the_f = fu ? f_u[cc][cmp] : f[cc][cmp];
        int sd = ft == D_stuff ? +1 : -1;
        realnum f_m_mult = ft == D_stuff ? 2 : (1 - 2 * cmp);

        for (int iz = (ft == D_stuff); iz < nz + (ft == D_stuff); ++iz) {
          realnum df;
          realnum dfcnd = (sd * Courant) * (f_p[iz] - f_p[iz - sd] - f_m_mult * f_m[iz]) *
                         (cndinv ? cndinv[iz] : 1);
          if (fcnd) fcnd[iz] += dfcnd;
          the_f[iz] += (df = dfcnd * (siginv ? siginv[dk + 2 * (dsig == Z) * iz] : 1));
          if (fu) fu[iz] += siginvu[dku + 2 * (dsigu == Z) * iz] * df;
        }
        if (ft == D_stuff) {
          ZERO_Z(f[Dz][cmp]);
          if (f_cond[Dz][cmp]) ZERO_Z(f_cond[Dz][cmp]);
          if (f_u[Dz][cmp]) ZERO_Z(f_u[Dz][cmp]);
        }
      }
      else if (m != 0) {                  // m != {0,+1,-1}
        if (zero_fields_near_cylorigin) { /* default behavior */
          /* I seem to recall David telling me that this was for numerical
             stability of some sort - the larger m is, the farther from
             the origin we need to be before we can use nonzero fields
             ... note that this is a fixed number of pixels for a given m,
             so it should still converge.  Still, this is weird...

             Update: experimentally, this seems to indeed be important
             for stability.  Setting these fields to zero, it seems to be
             stable with a Courant number < 0.62 or so for all m.  Without
             this, it becomes unstable unless we set the Courant number to
             about 1 / (|m| + 0.5) or less.

             Cons: setting fields near the origin to identically zero is
             somewhat unexpected for users, and probably spoils 2nd-order
             accuracy, and may not fix all stability issues anyway (based
             on anecdotal evidence from Alex M. of having to reduce Courant
             for large m). */
          double rmax = fabs(m) - int(gv.origin_r() * gv.a + 0.5);
          if (ft == D_stuff)
            for (int r = 0; r <= gv.nr() && r < rmax; r++) {
              const int ir = r * (nz + 1);
              ZERO_Z(f[Dp][cmp] + ir);
              ZERO_Z(f[Dz][cmp] + ir);
              if (f_cond[Dp][cmp]) ZERO_Z(f_cond[Dp][cmp] + ir);
              if (f_cond[Dz][cmp]) ZERO_Z(f_cond[Dz][cmp] + ir);
              if (f_u[Dp][cmp]) ZERO_Z(f_u[Dp][cmp] + ir);
              if (f_u[Dz][cmp]) ZERO_Z(f_u[Dz][cmp] + ir);
            }
          else
            for (int r = 0; r <= gv.nr() && r < rmax; r++) {
              const int ir = r * (nz + 1);
              ZERO_Z(f[Br][cmp] + ir);
              if (f_cond[Br][cmp]) ZERO_Z(f_cond[Br][cmp] + ir);
              if (f_u[Br][cmp]) ZERO_Z(f_u[Br][cmp] + ir);
            }
        }
        else {
          /* Without David's hack: just set boundary conditions at r=0.
             This seems to be unstable unless we make the Courant number
             around 1 / (|m| + 0.5) or smaller.  Pros: probably maintains
             2nd-order accuracy, is more sane for r near zero.  Cons:
             1/(|m|+0.5) is purely empirical (no theory yet), and I'm not
             sure how universal it is.  Makes higher m's more expensive. */
          if (ft == D_stuff) {
            ZERO_Z(f[Dp][cmp]);
            ZERO_Z(f[Dz][cmp]);
            if (f_cond[Dp][cmp]) ZERO_Z(f_cond[Dp][cmp]);
            if (f_cond[Dz][cmp]) ZERO_Z(f_cond[Dz][cmp]);
            if (f_u[Dp][cmp]) ZERO_Z(f_u[Dp][cmp]);
            if (f_u[Dz][cmp]) ZERO_Z(f_u[Dz][cmp]);
          }
          else {
            ZERO_Z(f[Br][cmp]);
            if (f_cond[Br][cmp]) ZERO_Z(f_cond[Br][cmp]);
            if (f_u[Br][cmp]) ZERO_Z(f_u[Br][cmp]);
          }
        }
      }
    }

  return allocated_u;
}

} // namespace meep
