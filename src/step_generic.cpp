#include "meep.hpp"
#include "meep_internals.hpp"
#include "config.h"

#define RPR realnum *restrict

/* These macros get into the guts of the PLOOP_OVER_VOL loops to
   efficiently construct the index k into a PML sigma array.
   Basically, k needs to increment by 2 for each increment of one of
   LOOP's for-loops, starting at the appropriate corner of the grid_volume,
   and these macros define the relevant strides etc. for each loop.
   KSTRIDE_DEF defines the relevant strides etc. and goes outside the
   LOOP, wheras KDEF defines the k index and goes inside the LOOP. */
#define KSTRIDE_DEF(dsig, k, corner)                                                               \
  const int k##0 = corner.in_direction(dsig) - gv.little_corner().in_direction(dsig);              \
  const int s##k##1 = gv.yucky_direction(0) == dsig ? 2 : 0;                                       \
  const int s##k##2 = gv.yucky_direction(1) == dsig ? 2 : 0;                                       \
  const int s##k##3 = gv.yucky_direction(2) == dsig ? 2 : 0
#define KDEF(k, dsig)                                                                              \
  const int k = ((k##0 + s##k##1 * loop_i1) + s##k##2 * loop_i2) + s##k##3 * loop_i3
#define DEF_k KDEF(k, dsig)
#define DEF_ku KDEF(ku, dsigu)
#define DEF_kw KDEF(kw, dsigw)

using namespace std;

namespace meep {

#define SWAP(t, a, b)                                                                              \
  {                                                                                                \
    t xxxx = a;                                                                                    \
    a = b;                                                                                         \
    b = xxxx;                                                                                      \
  }

/* update step for df/dt = curl g,
   i.e. f += dt curl g = dt/dx (dg1 - dg2)
   where dgk = gk[i] - gk[i+sk].

   g = (g1,g2), where g1 or g2 may be NULL.  Note that dt/dx and/or s1
   and s2 may be negative to flip signs of derivatives.

   PML: sig[k] = sigma[k]*dt/2, siginv[k] = 1 / (kap[k] + sigma[k]*dt/2).
   Here, k is the index in the dsig direction.  if dsig ==
   NO_DIRECTION, then PML is not used.  (dsig is the sigma direction.)

   if non-NULL, then cnd is an array of conductivity values, changing
   the underlying PDE to:
       df/dt = curl g - cnd f
   which is updated as:
       f = [ dt * curl g + (1 - dt cnd/2) f ] / (1 + dt cnd/2)
   cndinv should be an array of 1 / (1 + dt cnd/2).  In the case
   of PML, cndinv should contain 1 / (1 + dt (cnd + sigma)/2).

   fcnd is an auxiliary field used ONLY when we simultaneously have
   PML (dsig != NO_DIR) and conductivity, in which case fcnd solves
       dfcnd/dt = curl g - cnd*fcnd
   and f satisfies
       df/dt = dfcnd/dt - sigma*f.

   fu is another auxiliary field used only in PML (dsigu != NO_DIR),
   in which case f solves:
       df/dt = dfu/dt - sigma_u * f
   and fu replaces f in the equations above (fu += dt curl g etcetera).
*/
void step_curl(RPR f, component c, const RPR g1, const RPR g2,
               ptrdiff_t s1, ptrdiff_t s2, // strides for g1/g2 shift
               const grid_volume &gv, realnum dtdx, direction dsig, const RPR sig, const RPR kap,
               const RPR siginv, RPR fu, direction dsigu, const RPR sigu, const RPR kapu,
               const RPR siginvu, realnum dt, const RPR cnd, const RPR cndinv, RPR fcnd) {
  if (!g1) { // swap g1 and g2
    SWAP(const RPR, g1, g2);
    SWAP(ptrdiff_t, s1, s2);
    dtdx = -dtdx; // need to flip derivative sign
  }

  /* The following are a bunch of special cases of the "MOST GENERAL CASE"
     loop below.  We make copies of the loop for each special case in
     order to keep the innermost loop efficient.  This is especially
     important because the non-PML cases are actually more common.
     (The "right" way to do this is by partial evaluation of the
      most general case, but that would require a code generator.) */

  if (dsig == NO_DIRECTION) {    // no PML in f update
    if (dsigu == NO_DIRECTION) { // no fu update
      if (cnd) {
        realnum dt2 = dt * 0.5;
        if (g2) {
          PLOOP_OVER_VOL_OWNED0(gv, c, i) {
            f[i] = ((1 - dt2 * cnd[i]) * f[i] - dtdx * (g1[i + s1] - g1[i] + g2[i] - g2[i + s2])) *
                   cndinv[i];
          }
        }
        else {
          PLOOP_OVER_VOL_OWNED0(gv, c, i) {
            f[i] = ((1 - dt2 * cnd[i]) * f[i] - dtdx * (g1[i + s1] - g1[i])) * cndinv[i];
          }
        }
      }
      else { // no conductivity
        if (g2) {
          PLOOP_OVER_VOL_OWNED0(gv, c, i) {
            f[i] -= dtdx * (g1[i + s1] - g1[i] + g2[i] - g2[i + s2]);
          }
        }
        else {
          PLOOP_OVER_VOL_OWNED0(gv, c, i) { f[i] -= dtdx * (g1[i + s1] - g1[i]); }
        }
      }
    }
    else { // fu update, no PML in f update
      KSTRIDE_DEF(dsigu, ku, gv.little_owned_corner0(c));
      if (cnd) {
        realnum dt2 = dt * 0.5;
        if (g2) {
          PLOOP_OVER_VOL_OWNED0(gv, c, i) {
            DEF_ku;
            realnum fprev = fu[i];
            fu[i] =
                ((1 - dt2 * cnd[i]) * fprev - dtdx * (g1[i + s1] - g1[i] + g2[i] - g2[i + s2])) *
                cndinv[i];
            f[i] = siginvu[ku] * ((kapu[ku] - sigu[ku]) * f[i] + fu[i] - fprev);
          }
        }
        else {
          PLOOP_OVER_VOL_OWNED0(gv, c, i) {
            DEF_ku;
            realnum fprev = fu[i];
            fu[i] = ((1 - dt2 * cnd[i]) * fprev - dtdx * (g1[i + s1] - g1[i])) * cndinv[i];
            f[i] = siginvu[ku] * ((kapu[ku] - sigu[ku]) * f[i] + fu[i] - fprev);
          }
        }
      }
      else { // no conductivity
        if (g2) {
          PLOOP_OVER_VOL_OWNED0(gv, c, i) {
            DEF_ku;
            realnum fprev = fu[i];
            fu[i] -= dtdx * (g1[i + s1] - g1[i] + g2[i] - g2[i + s2]);
            f[i] = siginvu[ku] * ((kapu[ku] - sigu[ku]) * f[i] + fu[i] - fprev);
          }
        }
        else {
          PLOOP_OVER_VOL_OWNED0(gv, c, i) {
            DEF_ku;
            realnum fprev = fu[i];
            fu[i] -= dtdx * (g1[i + s1] - g1[i]);
            f[i] = siginvu[ku] * ((kapu[ku] - sigu[ku]) * f[i] + fu[i] - fprev);
          }
        }
      }
    }
  }
  else { /* PML in f update */
    KSTRIDE_DEF(dsig, k, gv.little_owned_corner0(c));
    if (dsigu == NO_DIRECTION) { // no fu update
      if (cnd) {
        realnum dt2 = dt * 0.5;
        if (g2) {
          PLOOP_OVER_VOL_OWNED0(gv, c, i) {
            DEF_k;
            realnum fcnd_prev = fcnd[i];
            fcnd[i] =
                ((1 - dt2 * cnd[i]) * fcnd[i] - dtdx * (g1[i + s1] - g1[i] + g2[i] - g2[i + s2])) *
                cndinv[i];
            f[i] = ((kap[k] - sig[k]) * f[i] + (fcnd[i] - fcnd_prev)) * siginv[k];
          }
        }
        else {
          PLOOP_OVER_VOL_OWNED0(gv, c, i) {
            DEF_k;
            realnum fcnd_prev = fcnd[i];
            fcnd[i] = ((1 - dt2 * cnd[i]) * fcnd[i] - dtdx * (g1[i + s1] - g1[i])) * cndinv[i];
            f[i] = ((kap[k] - sig[k]) * f[i] + (fcnd[i] - fcnd_prev)) * siginv[k];
          }
        }
      }
      else { // no conductivity (other than PML conductivity)
        if (g2) {
          PLOOP_OVER_VOL_OWNED0(gv, c, i) {
            DEF_k;
            f[i] = ((kap[k] - sig[k]) * f[i] - dtdx * (g1[i + s1] - g1[i] + g2[i] - g2[i + s2])) *
                   siginv[k];
          }
        }
        else {
          PLOOP_OVER_VOL_OWNED0(gv, c, i) {
            DEF_k;
            f[i] = ((kap[k] - sig[k]) * f[i] - dtdx * (g1[i + s1] - g1[i])) * siginv[k];
          }
        }
      }
    }
    else { // fu update + PML in f update
      KSTRIDE_DEF(dsigu, ku, gv.little_owned_corner0(c));
      if (cnd) {
        realnum dt2 = dt * 0.5;
        if (g2) {
          //////////////////// MOST GENERAL CASE //////////////////////
          PLOOP_OVER_VOL_OWNED0(gv, c, i) {
            DEF_k;
            DEF_ku;
            realnum fprev = fu[i];
            realnum fcnd_prev = fcnd[i];
            fcnd[i] =
                ((1 - dt2 * cnd[i]) * fcnd[i] - dtdx * (g1[i + s1] - g1[i] + g2[i] - g2[i + s2])) *
                cndinv[i];
            fu[i] = ((kap[k] - sig[k]) * fu[i] + (fcnd[i] - fcnd_prev)) * siginv[k];
            f[i] = siginvu[ku] * ((kapu[ku] - sigu[ku]) * f[i] + fu[i] - fprev);
          }
          /////////////////////////////////////////////////////////////
        }
        else {
          PLOOP_OVER_VOL_OWNED0(gv, c, i) {
            DEF_k;
            DEF_ku;
            realnum fprev = fu[i];
            realnum fcnd_prev = fcnd[i];
            fcnd[i] = ((1 - dt2 * cnd[i]) * fcnd[i] - dtdx * (g1[i + s1] - g1[i])) * cndinv[i];
            fu[i] = ((kap[k] - sig[k]) * fu[i] + (fcnd[i] - fcnd_prev)) * siginv[k];
            f[i] = siginvu[ku] * ((kapu[ku] - sigu[ku]) * f[i] + fu[i] - fprev);
          }
        }
      }
      else { // no conductivity (other than PML conductivity)
        if (g2) {
          PLOOP_OVER_VOL_OWNED0(gv, c, i) {
            DEF_k;
            DEF_ku;
            realnum fprev = fu[i];
            fu[i] = ((kap[k] - sig[k]) * fu[i] - dtdx * (g1[i + s1] - g1[i] + g2[i] - g2[i + s2])) *
                    siginv[k];
            f[i] = siginvu[ku] * ((kapu[ku] - sigu[ku]) * f[i] + fu[i] - fprev);
          }
        }
        else {
          PLOOP_OVER_VOL_OWNED0(gv, c, i) {
            DEF_k;
            DEF_ku;
            realnum fprev = fu[i];
            fu[i] = ((kap[k] - sig[k]) * fu[i] - dtdx * (g1[i + s1] - g1[i])) * siginv[k];
            f[i] = siginvu[ku] * ((kapu[ku] - sigu[ku]) * f[i] + fu[i] - fprev);
          }
        }
      }
    }
  }
}

/* field-update equation f += betadt * g (plus variants for conductivity
   and/or PML).  This is used in 2d calculations to add an exp(i beta z)
   time dependence, which gives an additional i \beta \hat{z} \times
   cross-product in the curl equations. */
void step_beta(RPR f, component c, const RPR g, const grid_volume &gv, realnum betadt,
               direction dsig, const RPR siginv, RPR fu, direction dsigu, const RPR siginvu,
               const RPR cndinv, RPR fcnd) {
  if (!g) return;
  if (dsig != NO_DIRECTION) { // PML in f update
    KSTRIDE_DEF(dsig, k, gv.little_owned_corner0(c));
    if (dsigu != NO_DIRECTION) { // PML in f + fu
      KSTRIDE_DEF(dsigu, ku, gv.little_owned_corner0(c));
      if (cndinv) { // conductivity + PML
        //////////////////// MOST GENERAL CASE //////////////////////
        PLOOP_OVER_VOL_OWNED0(gv, c, i) {
          DEF_k;
          DEF_ku;
          realnum df;
          realnum dfcnd = betadt * g[i] * cndinv[i];
          fcnd[i] += dfcnd;
          fu[i] += (df = dfcnd * siginv[k]);
          f[i] += siginvu[ku] * df;
        }
        /////////////////////////////////////////////////////////////
      }
      else { // PML only
        PLOOP_OVER_VOL_OWNED0(gv, c, i) {
          DEF_k;
          DEF_ku;
          realnum df;
          fu[i] += (df = betadt * g[i] * siginv[k]);
          f[i] += siginvu[ku] * df;
        }
      }
    }
    else {          // PML in f, no fu
      if (cndinv) { // conductivity + PML
        PLOOP_OVER_VOL_OWNED0(gv, c, i) {
          DEF_k;
          realnum dfcnd = betadt * g[i] * cndinv[i];
          fcnd[i] += dfcnd;
          f[i] += dfcnd * siginv[k];
        }
      }
      else { // PML only
        PLOOP_OVER_VOL_OWNED0(gv, c, i) {
          DEF_k;
          f[i] += betadt * g[i] * siginv[k];
        }
      }
    }
  }
  else {                         // no PML in f update
    if (dsigu != NO_DIRECTION) { // fu, no PML in f
      KSTRIDE_DEF(dsigu, ku, gv.little_owned_corner0(c));
      if (cndinv) { // conductivity, no PML
        PLOOP_OVER_VOL_OWNED0(gv, c, i) {
          DEF_ku;
          realnum df;
          fu[i] += (df = betadt * g[i] * cndinv[i]);
          f[i] += siginvu[ku] * df;
        }
      }
      else { // no conductivity or PML
        PLOOP_OVER_VOL_OWNED0(gv, c, i) {
          DEF_ku;
          realnum df;
          fu[i] += (df = betadt * g[i]);
          f[i] += siginvu[ku] * df;
        }
      }
    }
    else {          // no PML, no fu
      if (cndinv) { // conductivity, no PML
        PLOOP_OVER_VOL_OWNED0(gv, c, i) { f[i] += betadt * g[i] * cndinv[i]; }
      }
      else { // no conductivity or PML
        PLOOP_OVER_VOL_OWNED0(gv, c, i) { f[i] += betadt * g[i]; }
      }
    }
  }
}

/* Given Dsqr = |D|^2 and Di = component of D, compute the factor f so
   that Ei = chi1inv * f * Di.   In principle, this would involve solving
   a cubic equation, but instead we use a Pade approximant that is
   accurate to several orders.  This is inaccurate if the nonlinear
   index change is large, of course, but in that case the chi2/chi3
   power-series expansion isn't accurate anyway, so the cubic isn't
   physical there either. */
inline realnum calc_nonlinear_u(const realnum Dsqr, const realnum Di, const realnum chi1inv,
                                const realnum chi2, const realnum chi3) {
  realnum c2 = Di * chi2 * (chi1inv * chi1inv);
  realnum c3 = Dsqr * chi3 * (chi1inv * chi1inv * chi1inv);
  return (1 + c2 + 2 * c3) / (1 + 2 * c2 + 3 * c3);
}

/* Update E from D using epsilon and PML, *or* update H from B using
   mu and PML.

   To be generic, here we set f = u * g, where u may
   be a tensor, and we also have a nonlinear susceptibility chi.
   Here, g = (g,g1,g2) where g1 and g2 are the off-diagonal
   components, if any (g2 may be NULL).

   In PML (dsigw != NO_DIR), we have an additional auxiliary field fw,
   which is updated by the equations:
          fw = u * g
          df/dt = kappaw dfw/dt - sigmaw * fw
   That is, fw is updated like the non-PML f, and f is updated from
   fw by a little ODE.  Here, sigw[k] = sigmaw[k]*dt/2, kappaw[k] = kapw[k]

*/

void step_update_EDHB(RPR f, component fc, const grid_volume &gv, const RPR g, const RPR g1,
                      const RPR g2, const RPR u, const RPR u1, const RPR u2, ptrdiff_t s,
                      ptrdiff_t s1, ptrdiff_t s2, const RPR chi2, const RPR chi3, RPR fw,
                      direction dsigw, const RPR sigw, const RPR kapw) {
  if (!f) return;

  if ((!g1 && g2) || (g1 && g2 && !u1 && u2)) { /* swap g1 and g2 */
    SWAP(const RPR, g1, g2);
    SWAP(const RPR, u1, u2);
    SWAP(ptrdiff_t, s1, s2);
  }

// stable averaging of offdiagonal components
#define OFFDIAG(u, g, sx)                                                                          \
  (0.25 * ((g[i] + g[i - sx]) * u[i] + (g[i + s] + g[(i + s) - sx]) * u[i + s]))

  /* As with step_curl, these loops are all essentially copies
     of the "MOST GENERAL CASE" loop with various terms thrown out. */

  if (dsigw != NO_DIRECTION) { //////// PML case (with fw) /////////////
    KSTRIDE_DEF(dsigw, kw, gv.little_owned_corner0(fc));
    if (u1 && u2) { // 3x3 off-diagonal u
      if (chi3) {
        //////////////////// MOST GENERAL CASE //////////////////////
        PLOOP_OVER_VOL_OWNED(gv, fc, i) {
          realnum g1s = g1[i] + g1[i + s] + g1[i - s1] + g1[i + (s - s1)];
          realnum g2s = g2[i] + g2[i + s] + g2[i - s2] + g2[i + (s - s2)];
          realnum gs = g[i];
          realnum us = u[i];
          DEF_kw;
          realnum fwprev = fw[i], kapwkw = kapw[kw], sigwkw = sigw[kw];
          fw[i] = (gs * us + OFFDIAG(u1, g1, s1) + OFFDIAG(u2, g2, s2)) *
                  calc_nonlinear_u(gs * gs + 0.0625 * (g1s * g1s + g2s * g2s), gs, us, chi2[i],
                                   chi3[i]);
          f[i] += (kapwkw + sigwkw) * fw[i] - (kapwkw - sigwkw) * fwprev;
        }
        /////////////////////////////////////////////////////////////
      }
      else {
        PLOOP_OVER_VOL_OWNED(gv, fc, i) {
          realnum gs = g[i];
          realnum us = u[i];
          DEF_kw;
          realnum fwprev = fw[i], kapwkw = kapw[kw], sigwkw = sigw[kw];
          fw[i] = (gs * us + OFFDIAG(u1, g1, s1) + OFFDIAG(u2, g2, s2));
          f[i] += (kapwkw + sigwkw) * fw[i] - (kapwkw - sigwkw) * fwprev;
        }
      }
    }
    else if (u1) { // 2x2 off-diagonal u
      if (chi3) {
        PLOOP_OVER_VOL_OWNED(gv, fc, i) {
          realnum g1s = g1[i] + g1[i + s] + g1[i - s1] + g1[i + (s - s1)];
          realnum gs = g[i];
          realnum us = u[i];
          DEF_kw;
          realnum fwprev = fw[i], kapwkw = kapw[kw], sigwkw = sigw[kw];
          fw[i] = (gs * us + OFFDIAG(u1, g1, s1)) *
                  calc_nonlinear_u(gs * gs + 0.0625 * (g1s * g1s), gs, us, chi2[i], chi3[i]);
          f[i] += (kapwkw + sigwkw) * fw[i] - (kapwkw - sigwkw) * fwprev;
        }
      }
      else {
        PLOOP_OVER_VOL_OWNED(gv, fc, i) {
          realnum gs = g[i];
          realnum us = u[i];
          DEF_kw;
          realnum fwprev = fw[i], kapwkw = kapw[kw], sigwkw = sigw[kw];
          fw[i] = (gs * us + OFFDIAG(u1, g1, s1));
          f[i] += (kapwkw + sigwkw) * fw[i] - (kapwkw - sigwkw) * fwprev;
        }
      }
    }
    else if (u2) { // 2x2 off-diagonal u
      abort("bug - didn't swap off-diagonal terms!?");
    }
    else { // diagonal u
      if (chi3) {
        if (g1 && g2) {
          PLOOP_OVER_VOL_OWNED(gv, fc, i) {
            realnum g1s = g1[i] + g1[i + s] + g1[i - s1] + g1[i + (s - s1)];
            realnum g2s = g2[i] + g2[i + s] + g2[i - s2] + g2[i + (s - s2)];
            realnum gs = g[i];
            realnum us = u[i];
            DEF_kw;
            realnum fwprev = fw[i], kapwkw = kapw[kw], sigwkw = sigw[kw];
            fw[i] = (gs * us) * calc_nonlinear_u(gs * gs + 0.0625 * (g1s * g1s + g2s * g2s), gs, us,
                                                 chi2[i], chi3[i]);
            f[i] += (kapwkw + sigwkw) * fw[i] - (kapwkw - sigwkw) * fwprev;
          }
        }
        else if (g1) {
          PLOOP_OVER_VOL_OWNED(gv, fc, i) {
            realnum g1s = g1[i] + g1[i + s] + g1[i - s1] + g1[i + (s - s1)];
            realnum gs = g[i];
            realnum us = u[i];
            DEF_kw;
            realnum fwprev = fw[i], kapwkw = kapw[kw], sigwkw = sigw[kw];
            fw[i] = (gs * us) *
                    calc_nonlinear_u(gs * gs + 0.0625 * (g1s * g1s), gs, us, chi2[i], chi3[i]);
            f[i] += (kapwkw + sigwkw) * fw[i] - (kapwkw - sigwkw) * fwprev;
          }
        }
        else if (g2) {
          abort("bug - didn't swap off-diagonal terms!?");
        }
        else {
          PLOOP_OVER_VOL_OWNED(gv, fc, i) {
            realnum gs = g[i];
            realnum us = u[i];
            DEF_kw;
            realnum fwprev = fw[i], kapwkw = kapw[kw], sigwkw = sigw[kw];
            fw[i] = (gs * us) * calc_nonlinear_u(gs * gs, gs, us, chi2[i], chi3[i]);
            f[i] += (kapwkw + sigwkw) * fw[i] - (kapwkw - sigwkw) * fwprev;
          }
        }
      }
      else if (u) {
        PLOOP_OVER_VOL_OWNED(gv, fc, i) {
          realnum gs = g[i];
          realnum us = u[i];
          DEF_kw;
          realnum fwprev = fw[i], kapwkw = kapw[kw], sigwkw = sigw[kw];
          fw[i] = (gs * us);
          f[i] += (kapwkw + sigwkw) * fw[i] - (kapwkw - sigwkw) * fwprev;
        }
      }
      else {
        PLOOP_OVER_VOL_OWNED(gv, fc, i) {
          DEF_kw;
          realnum fwprev = fw[i], kapwkw = kapw[kw], sigwkw = sigw[kw];
          fw[i] = g[i];
          f[i] += (kapwkw + sigwkw) * fw[i] - (kapwkw - sigwkw) * fwprev;
        }
      }
    }
  }
  else {            /////////////// no PML (no fw) ///////////////////
    if (u1 && u2) { // 3x3 off-diagonal u
      if (chi3) {
        PLOOP_OVER_VOL_OWNED(gv, fc, i) {
          realnum g1s = g1[i] + g1[i + s] + g1[i - s1] + g1[i + (s - s1)];
          realnum g2s = g2[i] + g2[i + s] + g2[i - s2] + g2[i + (s - s2)];
          realnum gs = g[i];
          realnum us = u[i];
          f[i] = (gs * us + OFFDIAG(u1, g1, s1) + OFFDIAG(u2, g2, s2)) *
                 calc_nonlinear_u(gs * gs + 0.0625 * (g1s * g1s + g2s * g2s), gs, us, chi2[i],
                                  chi3[i]);
        }
      }
      else {
        PLOOP_OVER_VOL_OWNED(gv, fc, i) {
          realnum gs = g[i];
          realnum us = u[i];
          f[i] = (gs * us + OFFDIAG(u1, g1, s1) + OFFDIAG(u2, g2, s2));
        }
      }
    }
    else if (u1) { // 2x2 off-diagonal u
      if (chi3) {
        PLOOP_OVER_VOL_OWNED(gv, fc, i) {
          realnum g1s = g1[i] + g1[i + s] + g1[i - s1] + g1[i + (s - s1)];
          realnum gs = g[i];
          realnum us = u[i];
          f[i] = (gs * us + OFFDIAG(u1, g1, s1)) *
                 calc_nonlinear_u(gs * gs + 0.0625 * (g1s * g1s), gs, us, chi2[i], chi3[i]);
        }
      }
      else {
        PLOOP_OVER_VOL_OWNED(gv, fc, i) {
          realnum gs = g[i];
          realnum us = u[i];
          f[i] = (gs * us + OFFDIAG(u1, g1, s1));
        }
      }
    }
    else if (u2) { // 2x2 off-diagonal u
      abort("bug - didn't swap off-diagonal terms!?");
    }
    else { // diagonal u
      if (chi3) {
        if (g1 && g2) {
          PLOOP_OVER_VOL_OWNED(gv, fc, i) {
            realnum g1s = g1[i] + g1[i + s] + g1[i - s1] + g1[i + (s - s1)];
            realnum g2s = g2[i] + g2[i + s] + g2[i - s2] + g2[i + (s - s2)];
            realnum gs = g[i];
            realnum us = u[i];
            f[i] = (gs * us) * calc_nonlinear_u(gs * gs + 0.0625 * (g1s * g1s + g2s * g2s), gs, us,
                                                chi2[i], chi3[i]);
          }
        }
        else if (g1) {
          PLOOP_OVER_VOL_OWNED(gv, fc, i) {
            realnum g1s = g1[i] + g1[i + s] + g1[i - s1] + g1[i + (s - s1)];
            realnum gs = g[i];
            realnum us = u[i];
            f[i] = (gs * us) *
                   calc_nonlinear_u(gs * gs + 0.0625 * (g1s * g1s), gs, us, chi2[i], chi3[i]);
          }
        }
        else if (g2) {
          abort("bug - didn't swap off-diagonal terms!?");
        }
        else {
          PLOOP_OVER_VOL_OWNED(gv, fc, i) {
            realnum gs = g[i];
            realnum us = u[i];
            f[i] = (gs * us) * calc_nonlinear_u(gs * gs, gs, us, chi2[i], chi3[i]);
          }
        }
      }
      else if (u) {
        PLOOP_OVER_VOL_OWNED(gv, fc, i) {
          realnum gs = g[i];
          realnum us = u[i];
          f[i] = (gs * us);
        }
      }
      else
        PLOOP_OVER_VOL_OWNED(gv, fc, i) { f[i] = g[i]; }
    }
  }
}

} // namespace meep
