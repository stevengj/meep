#include <math.h>

#include "meep_internals.hpp"

#ifdef _OPENMP
#include "omp.h"
#endif

/* This file contains routines to compute the "average" or "effective"
   dielectric constant for a pixel, using an anisotropic averaging
   procedure described in an upcoming paper (similar to the one in
   MPB). */

using namespace std;

namespace meep {

////////////////////////////////////////////////////////////////////////////

#include "sphere-quad.h"

static vec sphere_pt(const vec &cent, double R, int n, double &weight) {
  switch (cent.dim) {
    case D1: {
      weight = sphere_quad[0][n][3];
      vec pt(sphere_quad[0][n][2]);
      return cent + pt * R;
    }
    case D2: {
      weight = sphere_quad[1][n][3];
      vec pt(sphere_quad[1][n][0], sphere_quad[1][n][1]);
      return cent + pt * R;
    }
    case D3: {
      weight = sphere_quad[2][n][3];
      vec pt(sphere_quad[2][n][0], sphere_quad[2][n][1], sphere_quad[2][n][2]);
      return cent + pt * R;
    }
    case Dcyl: {
      weight = sphere_quad[1][n][3];
      return cent + veccyl(sphere_quad[1][n][0], sphere_quad[1][n][1]) * R;
    }
    default: abort("unknown dimensions in sphere_pt\n");
  }
}

////////////////////////////////////////////////////////////////////////////

vec material_function::normal_vector(field_type ft, const volume &v) {
  vec gradient(zero_vec(v.dim));
  vec p(v.center());
  double R = v.diameter();
  int num_dirs = number_of_directions(v.dim);
  int min_iters = 1 << num_dirs;
  double chi1p1_prev = 0;
  bool break_early = true;
  for (int i = 0; i < num_sphere_quad[num_dirs - 1]; ++i) {
    double weight;
    vec pt = sphere_pt(p, R, i, weight);
    double chi1p1_val = chi1p1(ft, pt);

    if (i > 0 && i < min_iters) {
      if (chi1p1_val != chi1p1_prev) { break_early = false; }
      if (i == min_iters - 1 && break_early) {
        // Don't average regions where epsilon is uniform
        return zero_vec(v.dim);
      }
    }
    chi1p1_prev = chi1p1_val;
    gradient += (pt - p) * (weight * chi1p1_val);
  }
  return gradient;
}

/* default: simple numerical integration of surfaces/cubes, relative
   tolerance 'tol'.   This is superseded by the routines in the libctl
   interface, which either use a semi-analytical average or can
   use a proper adaptive cubature. */
void material_function::eff_chi1inv_row(component c, double chi1inv_row[3], const volume &v,
                                        double tol, int maxeval) {
  field_type ft = type(c);
  if (!maxeval) {
  trivial:
    chi1inv_row[0] = chi1inv_row[1] = chi1inv_row[2] = 0.0;
    chi1inv_row[component_direction(c) % 3] = 1 / chi1p1(ft, v.center());
    return;
  }

  vec gradient(normal_vector(ft, v));
  if (abs(gradient) < 1e-8) goto trivial;

  double meps = 1, minveps = 1;
  vec d = v.get_max_corner() - v.get_min_corner();
  int ms = 10;
  double old_meps = 0, old_minveps = 0;
  int iter = 0;
  switch (v.dim) {
    case D3:
      while ((fabs(meps - old_meps) > tol * fabs(old_meps)) &&
             (fabs(minveps - old_minveps) > tol * fabs(old_minveps))) {
        old_meps = meps;
        old_minveps = minveps;
        meps = minveps = 0;
        for (int k = 0; k < ms; k++)
          for (int j = 0; j < ms; j++)
            for (int i = 0; i < ms; i++) {
              double ep = chi1p1(ft, v.get_min_corner() +
                                         vec(i * d.x() / ms, j * d.y() / ms, k * d.z() / ms));
              if (ep < 0) goto trivial;
              meps += ep;
              minveps += 1 / ep;
            }
        meps /= ms * ms * ms;
        minveps /= ms * ms * ms;
        ms *= 2;
        if (maxeval && (iter += ms * ms * ms) >= maxeval) goto done;
      }
      break;
    case D2:
      while ((fabs(meps - old_meps) > tol * old_meps) &&
             (fabs(minveps - old_minveps) > tol * old_minveps)) {
        old_meps = meps;
        old_minveps = minveps;
        meps = minveps = 0;
        for (int j = 0; j < ms; j++)
          for (int i = 0; i < ms; i++) {
            double ep = chi1p1(ft, v.get_min_corner() + vec(i * d.x() / ms, j * d.y() / ms));
            if (ep < 0) goto trivial;
            meps += ep;
            minveps += 1 / ep;
          }
        meps /= ms * ms;
        minveps /= ms * ms;
        ms *= 2;
        if (maxeval && (iter += ms * ms) >= maxeval) goto done;
      }
      break;
    case Dcyl:
      while ((fabs(meps - old_meps) > tol * old_meps) &&
             (fabs(minveps - old_minveps) > tol * old_minveps)) {
        old_meps = meps;
        old_minveps = minveps;
        meps = minveps = 0;
        double sumvol = 0;
        for (int j = 0; j < ms; j++)
          for (int i = 0; i < ms; i++) {
            double r = v.get_min_corner().r() + i * d.r() / ms;
            double ep = chi1p1(ft, v.get_min_corner() + veccyl(i * d.r() / ms, j * d.z() / ms));
            if (ep < 0) goto trivial;
            sumvol += r;
            meps += ep * r;
            minveps += r / ep;
          }
        meps /= sumvol;
        minveps /= sumvol;
        ms *= 2;
        if (maxeval && (iter += ms * ms) >= maxeval) goto done;
      }
      break;
    case D1:
      while ((fabs(meps - old_meps) > tol * old_meps) &&
             (fabs(minveps - old_minveps) > tol * old_minveps)) {
        old_meps = meps;
        old_minveps = minveps;
        meps = minveps = 0;
        for (int i = 0; i < ms; i++) {
          double ep = chi1p1(ft, v.get_min_corner() + vec(i * d.z() / ms));
          if (ep < 0) {
            meps = chi1p1(ft, v.center());
            minveps = 1 / meps;
            goto done;
          }
          meps += ep;
          minveps += 1 / ep;
        }
        meps /= ms;
        minveps /= ms;
        ms *= 2;
        if (maxeval && (iter += ms * ms) >= maxeval) goto done;
      }
      break;
  }

done : {
  double n[3] = {0, 0, 0};
  double nabsinv = 1.0 / abs(gradient);
  LOOP_OVER_DIRECTIONS(gradient.dim, k) { n[k % 3] = gradient.in_direction(k) * nabsinv; }

  /* get rownum'th row of effective tensor
     P * minveps + (I-P) * 1/meps = P * (minveps-1/meps) + I * 1/meps
     where I is the identity and P is the projection matrix
     P_{ij} = n[i] * n[j]. */
  int rownum = component_direction(c) % 3;
  for (int i = 0; i < 3; ++i)
    chi1inv_row[i] = n[rownum] * n[i] * (minveps - 1 / meps);
  chi1inv_row[rownum] += 1 / meps;
}
}

void structure_chunk::set_chi1inv(component c, material_function &medium,
                                  bool use_anisotropic_averaging, double tol, int maxeval) {
  if (!is_mine() || !gv.has_field(c)) return;
  field_type ft = type(c);
  if (ft != E_stuff && ft != H_stuff) abort("only E or H can have chi");
  medium.set_volume(gv.pad().surroundings());

  if (!use_anisotropic_averaging) maxeval = 0;

  const double smoothing_diameter = 1.0; // FIXME: make user-changable?

  // may take a long time in 3d, so prepare to print status messages
  size_t npixels = 0, ipixel = 0;
  size_t loop_npixels = 0;
  LOOP_OVER_VOL(gv, c, i) {
    loop_npixels = loop_n1 * loop_n2 * loop_n3;
    goto breakout; // hack to use loop-size computation from LOOP_OVER_VOL
  }
breakout:
  npixels += loop_npixels;
  double last_output_time = wall_time();

  FOR_FT_COMPONENTS(ft, c2) if (gv.has_field(c2)) {
    direction d = component_direction(c2);
    if (!chi1inv[c][d]) chi1inv[c][d] = new realnum[gv.ntot()];
    if (!chi1inv[c][d]) abort("Memory allocation error.\n");
  }
  direction dc = component_direction(c);
  direction d0 = X, d1 = Y, d2 = Z;
  if (gv.dim == Dcyl) {
    d0 = R;
    d1 = P;
  }
  int idiag = component_index(c);
  bool trivial[3] = {true, true, true};
  double trivial_val[3] = {0, 0, 0};
  trivial_val[idiag] = 1.0;
  ivec shift1(unit_ivec(gv.dim, component_direction(c)) * (ft == E_stuff ? 1 : -1));
  PLOOP_OVER_IVECS_C(gv, (gv).little_corner() + (gv).iyee_shift(c),                                   \
                  (gv).big_corner() + (gv).iyee_shift(c), i, "omp parallel for collapse(3), firstprivate(last_output_time)") {
    double chi1invrow[3], chi1invrow_offdiag[3];
    IVEC_LOOP_ILOC(gv, here);
    medium.eff_chi1inv_row(c, chi1invrow, gv.dV(here, smoothing_diameter), tol, maxeval);
    medium.eff_chi1inv_row(c, chi1invrow_offdiag, gv.dV(here - shift1, smoothing_diameter), tol,
                           maxeval);
    if (chi1inv[c][d0]) {
      chi1inv[c][d0][i] = (d0 == dc) ? chi1invrow[0] : chi1invrow_offdiag[0];
      trivial[0] = trivial[0] && (chi1inv[c][d0][i] == trivial_val[0]);
    }
    if (chi1inv[c][d1]) {
      chi1inv[c][d1][i] = (d1 == dc) ? chi1invrow[1] : chi1invrow_offdiag[1];
      trivial[1] = trivial[1] && (chi1inv[c][d1][i] == trivial_val[1]);
    }
    if (chi1inv[c][d2]) {
      chi1inv[c][d2][i] = (d2 == dc) ? chi1invrow[2] : chi1invrow_offdiag[2];
      trivial[2] = trivial[2] && (chi1inv[c][d2][i] == trivial_val[2]);
    }

    if (verbosity > 0 && (ipixel + 1) % 1000 == 0 &&
        wall_time() > last_output_time + MEEP_MIN_OUTPUT_TIME && (omp_get_thread_num() == 0)) {
      master_printf("%s is %g%% done, %g s remaining\n",
                    use_anisotropic_averaging ? "subpixel-averaging" : "grid initialization",
                    ipixel * 100.0 / npixels,
                    (npixels - ipixel) * (wall_time() - last_output_time) / ipixel);
      last_output_time = wall_time();
    }
    #pragma omp atomic
    ++ipixel;
  }
  direction ds[3];
  ds[0] = d0;
  ds[1] = d1;
  ds[2] = d2;
  for (int i = 0; i < 3; ++i) {
    trivial_chi1inv[c][ds[i]] = trivial[i];
    if (i != idiag && trivial[i]) { // deallocate trivial offdiag
      delete[] chi1inv[c][ds[i]];
      chi1inv[c][ds[i]] = 0;
    }
  }
  // only deallocate trivial diag if entire tensor is trivial
  if (trivial[0] && trivial[1] && trivial[2]) {
    delete[] chi1inv[c][dc];
    chi1inv[c][dc] = 0;
  }
  medium.unset_volume();
}

void structure_chunk::add_susceptibility(material_function &sigma, field_type ft,
                                         const susceptibility &sus) {
  if (ft != E_stuff && ft != H_stuff) abort("susceptibilities must be for E or H fields");

  sigma.set_volume(gv.pad().surroundings());

  susceptibility *newsus = sus.clone();
  newsus->next = NULL;
  newsus->ntot = gv.ntot();
  // get rid of previously allocated sigma, normally not the case here:
  FOR_COMPONENTS(c) FOR_DIRECTIONS(d) if (newsus->sigma[c][d]) {
    delete[] newsus->sigma[c][d];
    newsus->sigma[c][d] = NULL;
    newsus->trivial_sigma[c][d] = true;
  }

  // if we own this chunk, set up the sigma array(s):
  if (is_mine()) FOR_FT_COMPONENTS(ft, c) if (gv.has_field(c)) {
      FOR_FT_COMPONENTS(ft, c2) if (gv.has_field(c2)) {
        direction d = component_direction(c2);
        if (!newsus->sigma[c][d]) newsus->sigma[c][d] = new realnum[gv.ntot()];
        if (!newsus->sigma[c][d]) abort("Memory allocation error.\n");
      }
      bool trivial[3] = {true, true, true};
      direction dc = component_direction(c);
      direction d0 = X, d1 = Y, d2 = Z;
      if (gv.dim == Dcyl) {
        d0 = R;
        d1 = P;
      }
      int idiag = component_index(c);
      realnum *s0 = newsus->sigma[c][d0];
      realnum *s1 = newsus->sigma[c][d1];
      realnum *s2 = newsus->sigma[c][d2];
      vec shift1(gv[unit_ivec(gv.dim, component_direction(c)) * (ft == E_stuff ? 1 : -1)]);
      LOOP_OVER_VOL(gv, c, i) {
        double sigrow[3], sigrow_offdiag[3];
        IVEC_LOOP_LOC(gv, here);
        sigma.sigma_row(c, sigrow, here);
        sigma.sigma_row(c, sigrow_offdiag, here - shift1);
        sigrow[(idiag + 1) % 3] = sigrow_offdiag[(idiag + 1) % 3];
        sigrow[(idiag + 2) % 3] = sigrow_offdiag[(idiag + 2) % 3];
        if (s0 && (s0[i] = sigrow[0]) != 0.) trivial[0] = false;
        if (s1 && (s1[i] = sigrow[1]) != 0.) trivial[1] = false;
        if (s2 && (s2[i] = sigrow[2]) != 0.) trivial[2] = false;
      }

      direction ds[3];
      ds[0] = d0;
      ds[1] = d1;
      ds[2] = d2;
      for (int i = 0; i < 3; ++i) {
        newsus->trivial_sigma[c][ds[i]] = trivial[i];
        if (i != idiag && trivial[i]) { // deallocate trivial offdiag
          delete[] newsus->sigma[c][ds[i]];
          newsus->sigma[c][ds[i]] = 0;
        }
      }
      // only deallocate trivial diag if entire tensor is trivial
      if (trivial[0] && trivial[1] && trivial[2]) {
        delete[] newsus->sigma[c][dc];
        newsus->sigma[c][dc] = 0;
      }
    }

  // finally, add to the beginning of the chiP list:
  newsus->next = chiP[ft];
  chiP[ft] = newsus;

  sigma.unset_volume();
}

} // namespace meep
