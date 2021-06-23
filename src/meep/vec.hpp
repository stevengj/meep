// -*- C++ -*-
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

#ifndef MEEP_VEC_H
#define MEEP_VEC_H

#include <complex>
#include <vector>
#include <stddef.h>

namespace meep {

const int NUM_FIELD_COMPONENTS = 20;
const int NUM_FIELD_TYPES = 8;

enum component {
  Ex = 0,
  Ey,
  Er,
  Ep,
  Ez,
  Hx,
  Hy,
  Hr,
  Hp,
  Hz,
  Dx,
  Dy,
  Dr,
  Dp,
  Dz,
  Bx,
  By,
  Br,
  Bp,
  Bz,
  Dielectric,
  Permeability,
  NO_COMPONENT
};
#define Centered Dielectric // better name for centered "dielectric" grid
enum derived_component {
  Sx = 100,
  Sy,
  Sr,
  Sp,
  Sz,
  EnergyDensity,
  D_EnergyDensity,
  H_EnergyDensity
};
enum ndim { D1 = 0, D2, D3, Dcyl };
enum field_type {
  E_stuff = 0,
  H_stuff = 1,
  D_stuff = 2,
  B_stuff = 3,
  PE_stuff = 4,
  PH_stuff = 5,
  WE_stuff = 6,
  WH_stuff = 7
};
enum boundary_side { High = 0, Low };
enum direction { X = 0, Y, Z, R, P, NO_DIRECTION };
struct signed_direction {
  explicit signed_direction(direction dd = X, bool f = false, std::complex<double> ph = 1.0) {
    d = dd;
    flipped = f;
    phase = ph;
  };
  signed_direction operator*(std::complex<double> ph);
  bool operator==(const signed_direction &sd) const {
    return (d == sd.d && flipped == sd.flipped && phase == sd.phase);
  }
  bool operator!=(const signed_direction &sd) const { return !(*this == sd); }
  direction d;
  bool flipped;
  std::complex<double> phase;
};

inline int number_of_directions(ndim dim) { return (int)(dim + 1 - 2 * (dim == Dcyl)); }

inline direction start_at_direction(ndim dim) {
  return (direction)(((dim == D1) || (dim == Dcyl)) ? 2 : 0);
}

inline direction stop_at_direction(ndim dim) { return (direction)(dim + 1 + 2 * (dim == D1)); }

component first_field_component(field_type ft);

#define FOR_FIELD_TYPES(ft)                                                                        \
  for (meep::field_type ft = meep::E_stuff; ft <= meep::WH_stuff; ft = (meep::field_type)(ft + 1))
#define FOR_ELECTRIC_COMPONENTS(c)                                                                 \
  for (meep::component c = meep::Ex; c < meep::Hx; c = (meep::component)(c + 1))
#define FOR_MAGNETIC_COMPONENTS(c)                                                                 \
  for (meep::component c = meep::Hz; c > meep::Ez; c = (meep::component)(c - 1))
#define FOR_B_COMPONENTS(c)                                                                        \
  for (meep::component c = meep::Bz; c > meep::Dz; c = (meep::component)(c - 1))
#define FOR_H_AND_B(h, b)                                                                          \
  for (meep::component h = meep::Hx, b = meep::Bx; h <= meep::Hz;                                  \
       h = (meep::component)(h + 1), b = (meep::component)(b + 1))
#define FOR_D_COMPONENTS(c)                                                                        \
  for (meep::component c = meep::Dz; c > meep::Hz; c = (meep::component)(c - 1))
#define FOR_E_AND_D(e, d)                                                                          \
  for (meep::component e = meep::Ex, d = meep::Dx; e <= meep::Ez;                                  \
       e = (meep::component)(e + 1), d = (component)(d + 1))
#define FOR_E_AND_H(c)                                                                             \
  for (meep::component c = meep::Ex; c < meep::Dx; c = (meep::component)(c + 1))
#define FOR_D_AND_B(c)                                                                             \
  for (meep::component c = meep::Dx; c < meep::Dielectric; c = (meep::component)(c + 1))
#define FOR_FT_COMPONENTS(ft, c)                                                                   \
  for (meep::component c = meep::first_field_component(ft),                                        \
                       loop_cstop = meep::component(meep::first_field_component(ft) + 5);          \
       c < loop_cstop; c = meep::component(c + 1))
#define FOR_COMPONENTS(c)                                                                          \
  for (meep::component c = meep::Ex, loop_stop_co = meep::Ey; c != loop_stop_co;                   \
       c = (meep::component)((c + 1) % meep::NUM_FIELD_COMPONENTS), loop_stop_co = meep::Ex)
#define FOR_DIRECTIONS(d)                                                                          \
  for (meep::direction d = meep::X, loop_stop_di = meep::Y; d != loop_stop_di;                     \
       d = (meep::direction)((d + 1) % 5), loop_stop_di = meep::X)
#define FOR_SIDES(s)                                                                               \
  for (meep::boundary_side s = meep::High, loop_stop_bi = meep::Low; s != loop_stop_bi;            \
       s = (meep::boundary_side)((s + 1) % 2), loop_stop_bi = meep::High)

// only loop over directions where we have coordinates
#define LOOP_OVER_DIRECTIONS(dim, d)                                                               \
  for (meep::direction d = meep::start_at_direction(dim),                                          \
                       loop_stop_directi = meep::stop_at_direction(dim);                           \
       d < loop_stop_directi; d = (meep::direction)(d + 1))

// loop over all directions in which we might have fields
#define LOOP_OVER_FIELD_DIRECTIONS(dim, d)                                                         \
  for (meep::direction d = dim == meep::Dcyl ? meep::Z : meep::X;                                  \
       d < (dim == meep::Dcyl ? meep::NO_DIRECTION : meep::R); d = meep::direction(d + 1))


#define LOOP_OVER_IVECS(gv, is, ie, idx)                                                           \
  for (ptrdiff_t loop_is1 = (is).yucky_val(0), loop_is2 = (is).yucky_val(1),                       \
                 loop_is3 = (is).yucky_val(2), loop_n1 = ((ie).yucky_val(0) - loop_is1) / 2 + 1,   \
                 loop_n2 = ((ie).yucky_val(1) - loop_is2) / 2 + 1,                                 \
                 loop_n3 = ((ie).yucky_val(2) - loop_is3) / 2 + 1,                                 \
                 loop_d1 = (gv).yucky_direction(0), loop_d2 = (gv).yucky_direction(1),             \
                 loop_d3 = (gv).yucky_direction(2),                                                \
                 loop_s1 = (gv).stride((meep::direction)loop_d1),                                  \
                 loop_s2 = (gv).stride((meep::direction)loop_d2),                                  \
                 loop_s3 = (gv).stride((meep::direction)loop_d3),                                  \
                 idx0 = (is - (gv).little_corner()).yucky_val(0) / 2 * loop_s1 +                   \
                        (is - (gv).little_corner()).yucky_val(1) / 2 * loop_s2 +                   \
                        (is - (gv).little_corner()).yucky_val(2) / 2 * loop_s3,                    \
                 loop_i1 = 0;                                                                      \
       loop_i1 < loop_n1; loop_i1++)                                                               \
    for (int loop_i2 = 0; loop_i2 < loop_n2; loop_i2++)                                            \
      for (ptrdiff_t idx = idx0 + loop_i1 * loop_s1 + loop_i2 * loop_s2, loop_i3 = 0;              \
           loop_i3 < loop_n3; loop_i3++, idx += loop_s3)

#define LOOP_OVER_VOL(gv, c, idx)                                                                  \
  LOOP_OVER_IVECS(gv, (gv).little_corner() + (gv).iyee_shift(c),                                   \
                  (gv).big_corner() + (gv).iyee_shift(c), idx)

#define LOOP_OVER_VOL_OWNED(gv, c, idx)                                                            \
  LOOP_OVER_IVECS(gv, (gv).little_owned_corner(c), (gv).big_corner(), idx)

#define LOOP_OVER_VOL_OWNED0(gv, c, idx)                                                           \
  LOOP_OVER_IVECS(gv, (gv).little_owned_corner0(c), (gv).big_corner(), idx)

#define LOOP_OVER_VOL_NOTOWNED(gv, c, idx)                                                         \
  for (ivec loop_notowned_is((gv).dim, 0), loop_notowned_ie((gv).dim, 0);                          \
       loop_notowned_is == zero_ivec((gv).dim);)                                                   \
    for (int loop_ibound = 0;                                                                      \
         (gv).get_boundary_icorners(c, loop_ibound, &loop_notowned_is, &loop_notowned_ie);         \
         loop_ibound++)                                                                            \
  LOOP_OVER_IVECS(gv, loop_notowned_is, loop_notowned_ie, idx)

// The following work identically to the LOOP_* macros above,
// but are optimized for shared memory parallelism. This
// is really only implemented in step_generic and code
// generated from step_generic

/* for the parallel implementation, we
introduce 2 dummy loops, one at the begininnging
and one at the end, in order to "trick" openMP
to allow us to define our localy variables
without having to change any other code
in the main codebase. We can proceed to do
a collapse over all three main loops. */

#define CHUNK_OPENMP _Pragma("omp parallel for")

// the most generic use case where the user
// can specify a custom clause
#define PLOOP_OVER_IVECS_C(gv, is, ie, idx, clause)			                                                     \
_Pragma("unroll(1)") \
for(ptrdiff_t loop_is1 = (is).yucky_val(0), loop_is2 = (is).yucky_val(1),                          \
                 loop_is3 = (is).yucky_val(2), loop_n1 = ((ie).yucky_val(0) - loop_is1) / 2 + 1,   \
                 loop_n2 = ((ie).yucky_val(1) - loop_is2) / 2 + 1,                                 \
                 loop_n3 = ((ie).yucky_val(2) - loop_is3) / 2 + 1,                                 \
                 loop_d1 = (gv).yucky_direction(0), loop_d2 = (gv).yucky_direction(1),             \
                 loop_d3 = (gv).yucky_direction(2),                                                \
                 loop_s1 = (gv).stride((meep::direction)loop_d1),                                  \
                 loop_s2 = (gv).stride((meep::direction)loop_d2),                                  \
                 loop_s3 = (gv).stride((meep::direction)loop_d3),                                  \
                 idx0 = (is - (gv).little_corner()).yucky_val(0) / 2 * loop_s1 +                   \
                        (is - (gv).little_corner()).yucky_val(1) / 2 * loop_s2 +                   \
                        (is - (gv).little_corner()).yucky_val(2) / 2 * loop_s3,                    \
                  dummy_first=0;dummy_first<1;dummy_first++)                                       \
_Pragma(clause)     				                                                     \
  for (ptrdiff_t loop_i1 = 0; loop_i1 < loop_n1; loop_i1++)                                        \
    for (ptrdiff_t loop_i2 = 0; loop_i2 < loop_n2; loop_i2++)                                      \
      for (ptrdiff_t loop_i3 = 0; loop_i3 < loop_n3; loop_i3++)                                    \
      _Pragma("unroll(1)") \
        for (ptrdiff_t idx = idx0 + loop_i1*loop_s1 + loop_i2*loop_s2 +                            \
           loop_i3*loop_s3, dummy_last=0;dummy_last<1;dummy_last++)

// For the main timestepping events, we know
// we want to do a simple collapse
#define PLOOP_OVER_IVECS(gv, is, ie, idx)                                                          \
  /*master_printf("Entered ploop\n");*/ \
  PLOOP_OVER_IVECS_C(gv, is, ie, idx, "omp parallel for collapse(3)")

#define PLOOP_OVER_VOL(gv, c, idx)                                                                  \
  PLOOP_OVER_IVECS(gv, (gv).little_corner() + (gv).iyee_shift(c),                                   \
                  (gv).big_corner() + (gv).iyee_shift(c), idx)

#define PLOOP_OVER_VOL_OWNED(gv, c, idx)                                                            \
  PLOOP_OVER_IVECS(gv, (gv).little_owned_corner(c), (gv).big_corner(), idx)

#define PLOOP_OVER_VOL_OWNED0(gv, c, idx)                                                           \
  PLOOP_OVER_IVECS(gv, (gv).little_owned_corner0(c), (gv).big_corner(), idx)

#define PLOOP_OVER_VOL_NOTOWNED(gv, c, idx)                                                         \
  for (ivec loop_notowned_is((gv).dim, 0), loop_notowned_ie((gv).dim, 0);                          \
       loop_notowned_is == zero_ivec((gv).dim);)                                                   \
    for (int loop_ibound = 0;                                                                      \
         (gv).get_boundary_icorners(c, loop_ibound, &loop_notowned_is, &loop_notowned_ie);         \
         loop_ibound++)                                                                            \
  PLOOP_OVER_IVECS(gv, loop_notowned_is, loop_notowned_ie, idx)

#define LOOPS_ARE_STRIDE1(gv) ((gv).stride((gv).yucky_direction(2)) == 1)

// The following work identically to the LOOP_* macros above,
// but assume that the inner loop is stride-1: LOOPS_ARE_STRIDE1(gv) *must*
// be true.  These are useful in allowing gcc to auto-vectorize the inner
// loop, since gcc's vectorizer requires the array stride to be known at
// compile time.  Note that stride-1 loops are the most common case in Meep.
// Note that we also specify _Pragma("ivdep"), which is a hint to
// compilers like icc (and hopefully gcc at some point) that the loop
// iterations don't have data dependencies.  This means that you
// should only use these macros where that is true!  (Basically,
// all of this is here to support performance hacks of step_generic.)

#if !defined(__INTEL_COMPILER) && !defined(__clang__) && !defined(_OPENMP) && (defined(__GNUC__) || defined(__GNUG__))
#define IVDEP _Pragma("GCC ivdep")
#elif defined(_OPENMP)
#define IVDEP _Pragma("omp simd")
#elif defined(__INTEL_COMPILER)
#define IVDEP _Pragma("ivdep")
#else
#define IVDEP
#endif

// loop over indices idx from is to ie (inclusive) in gv
#define S1LOOP_OVER_IVECS(gv, is, ie, idx)                                                         \
  for (ptrdiff_t loop_is1 = (is).yucky_val(0), loop_is2 = (is).yucky_val(1),                       \
                 loop_is3 = (is).yucky_val(2), loop_n1 = ((ie).yucky_val(0) - loop_is1) / 2 + 1,   \
                 loop_n2 = ((ie).yucky_val(1) - loop_is2) / 2 + 1,                                 \
                 loop_n3 = ((ie).yucky_val(2) - loop_is3) / 2 + 1,                                 \
                 loop_d1 = (gv).yucky_direction(0), loop_d2 = (gv).yucky_direction(1),             \
                 loop_s1 = (gv).stride((meep::direction)loop_d1),                                  \
                 loop_s2 = (gv).stride((meep::direction)loop_d2), loop_s3 = 1,                     \
                 idx0 = (is - (gv).little_corner()).yucky_val(0) / 2 * loop_s1 +                   \
                        (is - (gv).little_corner()).yucky_val(1) / 2 * loop_s2 +                   \
                        (is - (gv).little_corner()).yucky_val(2) / 2 * loop_s3,                    \
                 loop_i1 = 0;                                                                      \
       loop_i1 < loop_n1; loop_i1++)                                                               \
    for (int loop_i2 = 0; loop_i2 < loop_n2; loop_i2++)                                            \
      IVDEP                                                                                        \
  for (ptrdiff_t idx = idx0 + loop_i1 * loop_s1 + loop_i2 * loop_s2, loop_i3 = 0;                  \
       loop_i3 < loop_n3; loop_i3++, idx++)

#define S1LOOP_OVER_VOL(gv, c, idx)                                                                \
  S1LOOP_OVER_IVECS(gv, (gv).little_corner() + (gv).iyee_shift(c),                                 \
                    (gv).big_corner() + (gv).iyee_shift(c), idx)

#define S1LOOP_OVER_VOL_OWNED(gv, c, idx)                                                          \
  S1LOOP_OVER_IVECS(gv, (gv).little_owned_corner(c), (gv).big_corner(), idx)

#define S1LOOP_OVER_VOL_OWNED0(gv, c, idx)                                                         \
  S1LOOP_OVER_IVECS(gv, (gv).little_owned_corner0(c), (gv).big_corner(), idx)

#define S1LOOP_OVER_VOL_NOTOWNED(gv, c, idx)                                                       \
  for (ivec loop_notowned_is((gv).dim, 0), loop_notowned_ie((gv).dim, 0);                          \
       loop_notowned_is == meep::zero_ivec((gv).dim);)                                             \
    for (int loop_ibound = 0;                                                                      \
         (gv).get_boundary_icorners(c, loop_ibound, &loop_notowned_is, &loop_notowned_ie);         \
         loop_ibound++)                                                                            \
  S1LOOP_OVER_IVECS(gv, loop_notowned_is, loop_notowned_ie, idx)

/* The following is the stride optimized
version of the parallel loops from above.
We can use simd vectorization in addition
to the usual par for optimization */
// loop over indices idx from is to ie (inclusive) in gv
#define PS1LOOP_OVER_IVECS(gv, is, ie, idx)                                                        \
_Pragma("unroll(1)") \
for(ptrdiff_t loop_is1 = (is).yucky_val(0), loop_is2 = (is).yucky_val(1),                          \
                 loop_is3 = (is).yucky_val(2), loop_n1 = ((ie).yucky_val(0) - loop_is1) / 2 + 1,   \
                 loop_n2 = ((ie).yucky_val(1) - loop_is2) / 2 + 1,                                 \
                 loop_n3 = ((ie).yucky_val(2) - loop_is3) / 2 + 1,                                 \
                 loop_d1 = (gv).yucky_direction(0), loop_d2 = (gv).yucky_direction(1),             \
                 loop_s1 = (gv).stride((meep::direction)loop_d1),                                  \
                 loop_s2 = (gv).stride((meep::direction)loop_d2), loop_s3 = 1,                     \
                 idx0 = (is - (gv).little_corner()).yucky_val(0) / 2 * loop_s1 +                   \
                        (is - (gv).little_corner()).yucky_val(1) / 2 * loop_s2 +                   \
                        (is - (gv).little_corner()).yucky_val(2) / 2 * loop_s3,                    \
                  dummy_first=0;dummy_first<1;dummy_first++)                                       \
_Pragma("omp parallel for collapse(2)")				                                                     \
  for (ptrdiff_t loop_i1 = 0; loop_i1 < loop_n1; loop_i1++)                                        \
    for (ptrdiff_t loop_i2 = 0; loop_i2 < loop_n2; loop_i2++)                                      \
      _Pragma("omp simd") \
      for (ptrdiff_t loop_i3 = 0; loop_i3 < loop_n3; loop_i3++)                                    \
        _Pragma("unroll(1)") \
        for (ptrdiff_t idx = idx0 + loop_i1 * loop_s1 + loop_i2 * loop_s2 + loop_i3, dummy_last=0;dummy_last<1;dummy_last++)

#define PS1LOOP_OVER_VOL(gv, c, idx)                                                                \
  PS1LOOP_OVER_IVECS(gv, (gv).little_corner() + (gv).iyee_shift(c),                                 \
                    (gv).big_corner() + (gv).iyee_shift(c), idx)

#define PS1LOOP_OVER_VOL_OWNED(gv, c, idx)                                                          \
  PS1LOOP_OVER_IVECS(gv, (gv).little_owned_corner(c), (gv).big_corner(), idx)

#define PS1LOOP_OVER_VOL_OWNED0(gv, c, idx)                                                         \
  PS1LOOP_OVER_IVECS(gv, (gv).little_owned_corner0(c), (gv).big_corner(), idx)

#define PS1LOOP_OVER_VOL_NOTOWNED(gv, c, idx)                                                      \
  for (ivec loop_notowned_is((gv).dim, 0), loop_notowned_ie((gv).dim, 0);                          \
       loop_notowned_is == meep::zero_ivec((gv).dim);)                                             \
    for (int loop_ibound = 0;                                                                      \
         (gv).get_boundary_icorners(c, loop_ibound, &loop_notowned_is, &loop_notowned_ie);         \
         loop_ibound++)                                                                            \
  PS1LOOP_OVER_IVECS(gv, loop_notowned_is, loop_notowned_ie, idx)

#define IVEC_LOOP_AT_BOUNDARY                                                                      \
  ((loop_s1 != 0 && (loop_i1 == 0 || loop_i1 == loop_n1 - 1)) ||                                   \
   (loop_s2 != 0 && (loop_i2 == 0 || loop_i2 == loop_n2 - 1)) ||                                   \
   (loop_s3 != 0 && (loop_i3 == 0 || loop_i3 == loop_n3 - 1)))

#define IVEC_LOOP_ILOC(gv, iloc)                                                                   \
  ivec iloc((gv).dim);                                                                             \
  iloc.set_direction(direction(loop_d1), loop_is1 + 2 * loop_i1);                                  \
  iloc.set_direction(direction(loop_d2), loop_is2 + 2 * loop_i2);                                  \
  iloc.set_direction(direction(loop_d3), loop_is3 + 2 * loop_i3)

#define IVEC_LOOP_LOC(gv, loc)                                                                     \
  vec loc((gv).dim);                                                                               \
  loc.set_direction(direction(loop_d1), (0.5 * loop_is1 + loop_i1) * (gv).inva);                   \
  loc.set_direction(direction(loop_d2), (0.5 * loop_is2 + loop_i2) * (gv).inva);                   \
  loc.set_direction(direction(loop_d3), (0.5 * loop_is3 + loop_i3) * (gv).inva)

// integration weight for using LOOP_OVER_IVECS with field::integrate
#define IVEC_LOOP_WEIGHT1x(s0, s1, e0, e1, i, n, dir)                                              \
  ((i > 1 && i < n - 2)                                                                            \
       ? 1.0                                                                                       \
       : (i == 0 ? (s0).in_direction(meep::direction(dir))                                         \
                 : (i == 1 ? (s1).in_direction(meep::direction(dir))                               \
                           : i == n - 1                                                            \
                                 ? (e0).in_direction(meep::direction(dir))                         \
                                 : (i == n - 2 ? (e1).in_direction(meep::direction(dir)) : 1.0))))
#define IVEC_LOOP_WEIGHT1(s0, s1, e0, e1, k)                                                       \
  IVEC_LOOP_WEIGHT1x(s0, s1, e0, e1, loop_i##k, loop_n##k, loop_d##k)
#define IVEC_LOOP_WEIGHT(s0, s1, e0, e1, dV)                                                       \
  (IVEC_LOOP_WEIGHT1(s0, s1, e0, e1, 3) *                                                          \
   (IVEC_LOOP_WEIGHT1(s0, s1, e0, e1, 2) * ((dV)*IVEC_LOOP_WEIGHT1(s0, s1, e0, e1, 1))))

inline signed_direction flip(signed_direction d) {
  signed_direction d2 = d;
  d2.flipped = !d.flipped;
  return d2;
}

inline bool has_direction(ndim dim, direction d) {
  LOOP_OVER_DIRECTIONS(dim, dd) if (dd == d) return true;
  return false;
}

inline bool has_field_direction(ndim dim, direction d) {
  LOOP_OVER_FIELD_DIRECTIONS(dim, dd) if (dd == d) return true;
  return false;
}

// true if d is polar while dim is cartesian, or vice versa
inline bool coordinate_mismatch(ndim dim, direction d) {
  return (d != NO_DIRECTION && ((dim >= D1 && dim <= D3 && d != X && d != Y && d != Z) ||
                                (dim == Dcyl && d != R && d != P && d != Z)));
}

bool is_tm(component c);

#ifdef SWIG
extern void abort(const char *, ...); // mympi.cpp
#else
[[noreturn]] extern void abort(const char *, ...); // mympi.cpp
#endif

inline bool is_electric(component c) { return c < Hx; }
inline bool is_magnetic(component c) { return c >= Hx && c < Dx; }
inline bool is_D(component c) { return c >= Dx && c < Bx; }
inline bool is_B(component c) { return c >= Bx && c < Dielectric; }
inline bool is_derived(int c) { return c >= Sx; }
inline bool is_poynting(derived_component c) { return c < EnergyDensity; }
inline bool is_energydensity(derived_component c) { return c >= EnergyDensity; }
inline field_type type(component c) {
  if (is_electric(c))
    return E_stuff;
  else if (is_magnetic(c))
    return H_stuff;
  else if (is_D(c))
    return D_stuff;
  else if (is_B(c))
    return B_stuff;
  abort("Invalid field in type.\n");
  return E_stuff; // This is never reached.
}
const char *component_name(component c);
const char *component_name(derived_component c);
const char *component_name(int c);
const char *direction_name(direction);
const char *dimension_name(ndim);

inline int component_index(component c) {
  switch (c) {
    case Ex:
    case Hx:
    case Dx:
    case Bx: return 0;
    case Ey:
    case Hy:
    case Dy:
    case By: return 1;
    case Ez:
    case Hz:
    case Dz:
    case Bz: return 2;
    case Er:
    case Hr:
    case Dr:
    case Br: return 0;
    case Ep:
    case Hp:
    case Dp:
    case Bp: return 1;
    case Dielectric:
    case Permeability:
    case NO_COMPONENT: return -1;
  }
  return -2; // This code is never reached...
}

direction component_direction(int c);
int direction_component(int c, direction d);
inline direction component_direction(component c) {
  switch (c) {
    case Ex:
    case Hx:
    case Dx:
    case Bx: return X;
    case Ey:
    case Hy:
    case Dy:
    case By: return Y;
    case Ez:
    case Hz:
    case Dz:
    case Bz: return Z;
    case Er:
    case Hr:
    case Dr:
    case Br: return R;
    case Ep:
    case Hp:
    case Dp:
    case Bp: return P;
    case Dielectric:
    case Permeability:
    case NO_COMPONENT: return NO_DIRECTION;
  }
  return X; // This code is never reached...
}
inline direction component_direction(derived_component c) {
  switch (c) {
    case Sx: return X;
    case Sy: return Y;
    case Sz: return Z;
    case Sr: return R;
    case Sp: return P;
    case EnergyDensity:
    case D_EnergyDensity:
    case H_EnergyDensity: return NO_DIRECTION;
  }
  return X; // This code is never reached...
}
inline direction component_direction(int c) {
  if (is_derived(c))
    return component_direction(derived_component(c));
  else
    return component_direction(component(c));
}
inline component direction_component(component c, direction d) {
  component start_point;
  if (is_electric(c))
    start_point = Ex;
  else if (is_magnetic(c))
    start_point = Hx;
  else if (is_D(c))
    start_point = Dx;
  else if (is_B(c))
    start_point = Bx;
  else if (d == NO_DIRECTION && component_direction(c) == d)
    return c;
  else
    abort("unknown field component %d", c);
  switch (d) {
    case X: return start_point;
    case Y: return (component)(start_point + 1);
    case Z: return (component)(start_point + 4);
    case R: return (component)(start_point + 2);
    case P: return (component)(start_point + 3);
    case NO_DIRECTION: abort("vector %d component in NO_DIRECTION", c);
  }
  return Ex; // This is never reached.
}
inline derived_component direction_component(derived_component c, direction d) {
  derived_component start_point;
  if (is_poynting(c))
    start_point = Sx;
  else if (is_energydensity(c) && d == NO_DIRECTION)
    return c;
  else
    abort("unknown field component %d", c);
  switch (d) {
    case X: return start_point;
    case Y: return (derived_component)(start_point + 1);
    case Z: return (derived_component)(start_point + 4);
    case R: return (derived_component)(start_point + 2);
    case P: return (derived_component)(start_point + 3);
    case NO_DIRECTION: abort("vector %d derived_component in NO_DIRECTION", c);
  }
  return Sx; // This is never reached.
}
inline int direction_component(int c, direction d) {
  if (is_derived(c))
    return int(direction_component(derived_component(c), d));
  else
    return int(direction_component(component(c), d));
}

inline component field_type_component(field_type ft, component c) {
  return direction_component(first_field_component(ft), component_direction(c));
}

inline bool coordinate_mismatch(ndim dim, component c) {
  return coordinate_mismatch(dim, component_direction(c));
}
inline bool coordinate_mismatch(ndim dim, derived_component c) {
  return coordinate_mismatch(dim, component_direction(c));
}

// cyclically shift a direction d or a component c by shift
// assumes: shift >= -99, {d, component_direction(c)} != NO_DIRECTION,
//          and has_direction(dim, {d, component_direction(c)})
inline direction cycle_direction(ndim dim, direction d, int shift) {
  int start = dim == Dcyl ? 2 : 0;
  return direction((d - start + shift + 99) % 3 + start);
}
inline component cycle_component(ndim dim, component c, int shift) {
  return direction_component(c, cycle_direction(dim, component_direction(c), shift));
}

class vec;
vec veccyl(double rr, double zz);
vec zero_vec(ndim);

class vec {
public:
  vec() { init_t(); };
  vec(ndim di) {
    init_t();
    dim = di;
  };
  vec(ndim di, double val) {
    dim = di;
    t[0] = t[1] = t[2] = t[3] = t[4] = val;
  };
  vec(double zz) {
    init_t();
    dim = D1;
    t[Z] = zz;
  };
  vec(double xx, double yy) {
    init_t();
    dim = D2;
    t[X] = xx;
    t[Y] = yy;
  };
  vec(double xx, double yy, double zz) {
    init_t();
    dim = D3;
    t[X] = xx;
    t[Y] = yy;
    t[Z] = zz;
  };
  friend vec veccyl(double rr, double zz);
  ~vec(){};

  vec operator+(const vec &a) const {
    vec result = a;
    LOOP_OVER_DIRECTIONS(dim, d) result.t[d] += t[d];
    return result;
  };

  vec operator+=(const vec &a) {
    LOOP_OVER_DIRECTIONS(dim, d) t[d] += a.t[d];
    return *this;
  };

  vec operator-(const vec &a) const {
    vec result = a;
    LOOP_OVER_DIRECTIONS(dim, d) result.t[d] = t[d] - result.t[d];
    return result;
  };

  vec operator-(void) const {
    vec result(dim);
    LOOP_OVER_DIRECTIONS(dim, d) result.t[d] = -t[d];
    return result;
  };

  vec operator-=(const vec &a) {
    LOOP_OVER_DIRECTIONS(dim, d) t[d] -= a.t[d];
    return *this;
  };

  bool operator!=(const vec &a) const {
    LOOP_OVER_DIRECTIONS(dim, d) if (t[d] != a.t[d]) return true;
    return false;
  };

  bool operator==(const vec &a) const {
    LOOP_OVER_DIRECTIONS(dim, d) if (t[d] != a.t[d]) return false;
    return true;
  };

  vec round_float(void) const {
    vec result = *this;
    LOOP_OVER_DIRECTIONS(dim, d) result.t[d] = float(result.t[d]);
    return result;
  }

  vec operator*(double s) const {
    vec result = *this;
    LOOP_OVER_DIRECTIONS(dim, d) result.t[d] *= s;
    return result;
  };

  vec operator/(double s) const {
    vec result = *this;
    LOOP_OVER_DIRECTIONS(dim, d) result.t[d] *= (1.0 / s);
    return result;
  };

  // I use & as a dot product.
  double operator&(const vec &a) const {
    double result = 0.0;
    LOOP_OVER_DIRECTIONS(dim, d) result += t[d] * a.t[d];
    return result;
  };
  ndim dim;

  double r() const { return t[R]; };
  double x() const { return t[X]; };
  double y() const { return t[Y]; };
  double z() const { return t[Z]; };
  double in_direction(direction d) const { return t[d]; };
  void set_direction(direction d, double val) { t[d] = val; };

  // pretty-print to a user-supplied buffer (if provided) or to a static internal buffer (in which
  // case not thread-safe)
  const char *str(char *buffer = 0, size_t buflen = 0);

  double project_to_boundary(direction, double boundary_loc);
  friend vec zero_vec(ndim);
  friend vec one_vec(ndim);

private:
  double t[5];
  void init_t() {
    for (int i = 0; i < 5; ++i) {
      t[i] = 0;
    }
  }
};

inline double abs(const vec &pt) { return sqrt(pt & pt); }

inline vec zero_vec(ndim di) {
  vec pt(di);
  LOOP_OVER_DIRECTIONS(di, d) pt.set_direction(d, 0.0);
  return pt;
}

inline vec one_vec(ndim di) {
  vec pt(di);
  LOOP_OVER_DIRECTIONS(di, d) pt.set_direction(d, 1.0);
  return pt;
}

inline vec unit_vec(ndim di, direction d) {
  vec pt(zero_vec(di));
  pt.set_direction(d, 1.0);
  return pt;
}

inline vec clean_vec(const vec &pt, double val_unused = 0.0) {
  vec ptc(pt.dim, val_unused);
  LOOP_OVER_DIRECTIONS(pt.dim, d) ptc.set_direction(d, pt.in_direction(d));
  return ptc;
}

inline vec veccyl(double rr, double zz) {
  vec pt(Dcyl);
  pt.t[R] = rr;
  pt.t[Z] = zz;
  return pt;
}

class ivec;
ivec iveccyl(int xx, int yy);
ivec zero_ivec(ndim);
ivec one_ivec(ndim);

class ivec {
public:
  ivec() {
    init_t();
    dim = D2;
  };
  ivec(ndim di) {
    init_t();
    dim = di;
  };
  ivec(ndim di, int val) {
    dim = di;
    t[0] = t[1] = t[2] = t[3] = t[4] = val;
  };
  ivec(int zz) {
    init_t();
    dim = D1;
    t[Z] = zz;
  };
  ivec(int xx, int yy) {
    init_t();
    dim = D2;
    t[X] = xx;
    t[Y] = yy;
  };
  ivec(int xx, int yy, int zz) {
    init_t();
    dim = D3;
    t[X] = xx;
    t[Y] = yy;
    t[Z] = zz;
  };
  friend ivec iveccyl(int xx, int yy);
  ~ivec(){};

  // Only an idiot (or a macro) would use a yucky function.  Don't be an
  // idiot.
  int yucky_val(int) const;

  ivec operator+(const ivec &a) const {
    ivec result = a;
    LOOP_OVER_DIRECTIONS(dim, d) result.t[d] += t[d];
    return result;
  };

  ivec operator+=(const ivec &a) {
    LOOP_OVER_DIRECTIONS(dim, d) t[d] += a.t[d];
    return *this;
  };

  ivec operator-(const ivec &a) const {
    ivec result = a;
    LOOP_OVER_DIRECTIONS(dim, d) result.t[d] = t[d] - result.t[d];
    return result;
  };

  ivec operator-(void) const {
    ivec result(dim);
    LOOP_OVER_DIRECTIONS(dim, d) result.t[d] = -t[d];
    return result;
  };

  ivec operator-=(const ivec &a) {
    LOOP_OVER_DIRECTIONS(dim, d) t[d] -= a.t[d];
    return *this;
  };

  bool operator!=(const ivec &a) const {
    LOOP_OVER_DIRECTIONS(dim, d) if (t[d] != a.t[d]) return true;
    return false;
  };

  bool operator==(const ivec &a) const {
    LOOP_OVER_DIRECTIONS(dim, d) if (t[d] != a.t[d]) return false;
    return true;
  };

  bool operator<=(const ivec &a) const {
    LOOP_OVER_DIRECTIONS(dim, d) if (t[d] > a.t[d]) return false;
    return true;
  };

  bool operator>=(const ivec &a) const {
    LOOP_OVER_DIRECTIONS(dim, d) if (t[d] < a.t[d]) return false;
    return true;
  };

  bool operator<(const ivec &a) const {
    LOOP_OVER_DIRECTIONS(dim, d) if (t[d] >= a.t[d]) return false;
    return true;
  };

  bool operator>(const ivec &a) const {
    LOOP_OVER_DIRECTIONS(dim, d) if (t[d] <= a.t[d]) return false;
    return true;
  };

  ivec operator*(int s) const {
    ivec result = *this;
    LOOP_OVER_DIRECTIONS(dim, d) result.t[d] *= s;
    return result;
  };

  vec operator*(double s) const {
    vec result(dim);
    LOOP_OVER_DIRECTIONS(dim, d) result.set_direction(d, t[d] * s);
    return result;
  };
  ndim dim;

  int r() const { return t[R]; };
  int x() const { return t[X]; };
  int y() const { return t[Y]; };
  int z() const { return t[Z]; };
  int in_direction(direction d) const { return t[d]; };
  void set_direction(direction d, int val) { t[d] = val; };

  // pretty-print to a user-supplied buffer (if provided) or to a static internal buffer (in which
  // case not thread-safe)
  const char *str(char *buffer = 0, size_t buflen = 0);

  ivec round_up_to_even(void) const {
    ivec result(dim);
    LOOP_OVER_DIRECTIONS(dim, d)
    result.t[d] = t[d] + (t[d] >= 0 ? t[d] : -t[d]) % 2;
    return result;
  }

  friend ivec zero_ivec(ndim);
  friend ivec one_ivec(ndim);

private:
  int t[5];
  void init_t() {
    for (int i = 0; i < 5; ++i) {
      t[i] = 0;
    }
  }
};

inline ivec zero_ivec(ndim di) {
  ivec pt;
  pt.dim = di;
  LOOP_OVER_DIRECTIONS(di, d) pt.set_direction(d, 0);
  return pt;
}

inline ivec one_ivec(ndim di) {
  ivec pt;
  pt.dim = di;
  LOOP_OVER_DIRECTIONS(di, d) pt.set_direction(d, 1);
  return pt;
}

inline ivec unit_ivec(ndim di, direction d) {
  ivec pt(zero_ivec(di));
  pt.set_direction(d, 1);
  return pt;
}

inline ivec iveccyl(int rr, int zz) {
  ivec pt(Dcyl);
  pt.t[R] = rr;
  pt.t[Z] = zz;
  return pt;
}

vec max(const vec &vec1, const vec &vec2);
vec min(const vec &vec1, const vec &vec2);
ivec max(const ivec &ivec1, const ivec &ivec2);
ivec min(const ivec &ivec1, const ivec &ivec2);
ivec max_to_all(const ivec &); // in mympi.cpp

class volume {
public:
  ndim dim;
  volume(ndim di) {
    dim = di;
    min_corner.dim = di;
    max_corner.dim = di;
  };
  volume(const vec &vec1, const vec &vec2);
  volume(const vec &pt);
  void set_direction_min(direction d, double val) { min_corner.set_direction(d, val); };
  void set_direction_max(direction d, double val) { max_corner.set_direction(d, val); };
  double in_direction_min(direction d) const { return min_corner.in_direction(d); };
  double in_direction_max(direction d) const { return max_corner.in_direction(d); };
  double in_direction(direction d) const { return in_direction_max(d) - in_direction_min(d); }
  double computational_volume() const;
  double integral_volume() const;
  double full_volume() const;
  vec center() const { return (min_corner + max_corner) * 0.5; }
  double diameter() const;
  bool contains(const vec &h) const;
  bool contains(const volume &a) const;
  volume intersect_with(const volume &a) const;
  volume operator&(const volume &a) const { return intersect_with(a); };
  volume operator|(const volume &a) const {
    return volume(min(min_corner, a.min_corner), max(max_corner, a.max_corner));
  };
  volume operator+(const vec &a) const { return volume(min_corner + a, max_corner + a); }
  volume operator+=(const vec &a) {
    min_corner += a;
    max_corner += a;
    return *this;
  }
  volume operator-(const vec &a) const { return volume(min_corner - a, max_corner - a); }
  volume operator-=(const vec &a) {
    min_corner -= a;
    max_corner -= a;
    return *this;
  }
  bool operator==(const volume &a) const {
    return (min_corner == a.min_corner && max_corner == a.max_corner);
  }
  bool operator!=(const volume &a) const { return !(*this == a); };
  volume round_float(void) const {
    return volume(min_corner.round_float(), max_corner.round_float());
  }
  bool intersects(const volume &a) const;
  bool operator&&(const volume &a) const { return intersects(a); };
  vec get_min_corner() const { return min_corner; };
  vec get_max_corner() const { return max_corner; };
  direction normal_direction() const;

  const char *str(char *buffer = 0, size_t buflen = 0);

private:
  vec min_corner, max_corner;
};

class grid_volume;
grid_volume volcyl(double rsize, double zsize, double a);
grid_volume volone(double zsize, double a);
grid_volume vol1d(double zsize, double a);
grid_volume voltwo(double xsize, double ysize, double a);
grid_volume vol2d(double xsize, double ysize, double a);
grid_volume vol3d(double xsize, double ysize, double zsize, double a);

class grid_volume {
public:
  grid_volume(){};

  grid_volume subvolume(ivec is, ivec ie);
  void init_subvolume(ivec is, ivec ie);

  ndim dim;
  double a, inva /* = 1/a */;

  void print() const;
  ptrdiff_t stride(direction d) const { return the_stride[d]; };
  int num_direction(direction d) const { return num[((int)d) % 3]; };
  // Only an idiot (or a macro) would use a yucky function.  Don't be an
  // idiot.
  int yucky_num(int) const;
  direction yucky_direction(int) const;
  void set_num_direction(direction d, int value);
  int nr() const { return num_direction(R); }
  int nx() const { return num_direction(X); }
  int ny() const { return num_direction(Y); }
  int nz() const { return num_direction(Z); }

  bool has_field(component c) const {
    if (dim == D1) return c == Ex || c == Hy || c == Dx || c == By;
    return (dim == Dcyl) ? component_direction(c) > Y : component_direction(c) < R;
  }
  int has_boundary(boundary_side, direction) const;

  vec dr() const;
  vec dx() const;
  vec dy() const;
  vec dz() const;

  size_t ntot() const { return the_ntot; }
  size_t nowned_min() const {
    size_t n = 1;
    LOOP_OVER_DIRECTIONS(dim, d) n *= (size_t)(num_direction(d));
    return n;
  }
  size_t nowned(component c) const;
  vec operator[](const ivec &p) const { return p * (0.5 * inva); };
  ptrdiff_t index(component, const ivec &) const;
  ivec round_vec(const vec &) const;
  void interpolate(component, const vec &, ptrdiff_t indices[8], double weights[8]) const;
  void interpolate(component, const vec &, ivec locs[8], double weights[8]) const;

  volume dV(component c, ptrdiff_t index) const;
  volume dV(const ivec &, double diameter = 1.0) const;
  bool intersect_with(const grid_volume &vol_in, grid_volume *intersection = NULL,
                      grid_volume *others = NULL, int *num_others = NULL) const;
  double rmin() const;
  double rmax() const;
  double xmin() const;
  double xmax() const;
  double ymin() const;
  double ymax() const;
  double zmin() const;
  double zmax() const;
  vec center() const;
  ivec icenter() const;
  vec loc(component, ptrdiff_t index) const;
  vec loc_at_resolution(ptrdiff_t index, double res) const;
  size_t ntot_at_resolution(double res) const;
  ivec iloc(component, ptrdiff_t index) const;
  size_t surface_area() const;

  ptrdiff_t yee_index(component c) const {
    ptrdiff_t idx = 0;
    LOOP_OVER_DIRECTIONS(dim, d)
    idx += (1 - iyee_shift(c).in_direction(d)) * stride(d);
    return idx;
  }
  vec yee_shift(component) const;
  component eps_component() const;
  void yee2cent_offsets(component c, ptrdiff_t &offset1, ptrdiff_t &offset2) const;
  void cent2yee_offsets(component c, ptrdiff_t &offset1, ptrdiff_t &offset2) const;

  double boundary_location(boundary_side, direction) const;
  ivec big_corner() const;
  ivec little_corner() const { return io; };
  vec corner(boundary_side b) const;

  bool contains(const vec &) const;
  bool contains(const ivec &) const;

  /* differs from little_owned_corner in that it doesn't count
     "ownership" of the r=0 origin for Dcyl, which is updated separately */
  ivec little_owned_corner0(component c) const {
    return ivec(little_corner() + one_ivec(dim) * 2 - iyee_shift(c));
  }

  ivec little_owned_corner(component c) const;
  bool owns(const ivec &) const;
  volume surroundings() const;
  volume interior() const;

  bool get_boundary_icorners(component c, int ib, ivec *cs, ivec *ce) const;

  friend grid_volume volcyl(double rsize, double zsize, double a);
  friend grid_volume volone(double zsize, double a);
  friend grid_volume vol1d(double zsize, double a);
  friend grid_volume voltwo(double xsize, double ysize, double a);
  friend grid_volume vol2d(double xsize, double ysize, double a);
  friend grid_volume vol3d(double xsize, double ysize, double zsize, double a);

  grid_volume split_at_fraction(bool side_high, int split_pt, int split_dir) const;
  double get_cost() const;
  grid_volume halve(direction d) const;
  void pad_self(direction d);
  grid_volume pad(direction d) const;
  grid_volume pad() const {
    grid_volume gv(*this);
    LOOP_OVER_DIRECTIONS(dim, d)
    gv.pad_self(d);
    return gv;
  }
  ivec iyee_shift(component c) const {
    ivec out = zero_ivec(dim);
    LOOP_OVER_DIRECTIONS(dim, d)
    if (c == Dielectric || c == Permeability ||
        ((is_electric(c) || is_D(c)) && d == component_direction(c)) ||
        ((is_magnetic(c) || is_B(c)) && d != component_direction(c)))
      out.set_direction(d, 1);
    return out;
  }

  vec get_origin() const { return origin; }
  void set_origin(const vec &o);
  void set_origin(const ivec &o);
  void shift_origin(const vec &s) { set_origin(origin + s); }
  void shift_origin(const ivec &s) { set_origin(io + s); }
  void shift_origin(direction d, int s) { shift_origin(unit_ivec(dim, d) * s); }
  void set_origin(direction d, int o);
  void center_origin(void) { shift_origin(-icenter()); }
  double origin_in_direction(direction d) const { return origin.in_direction(d); }
  int iorigin_in_direction(direction d) const { return io.in_direction(d); }
  double origin_r() const { return origin.r(); }
  double origin_x() const { return origin.x(); }
  double origin_y() const { return origin.y(); }
  double origin_z() const { return origin.z(); }

  const char *str(char *buffer = 0, size_t buflen = 0);

  std::complex<double> get_split_costs(direction d, int split_point,
                                       bool frag_cost) const;
  void find_best_split(int desired_chunks, bool frag_cost,
                       int &best_split_point, direction &best_split_direction,
                       double &left_effort_fraction) const;

private:
  grid_volume(ndim d, double ta, int na, int nb, int nc);
  ivec io;    // integer origin ... always change via set_origin etc.!
  vec origin; // cache of operator[](io), for performance
  void update_ntot();
  void set_strides();
  void num_changed() {
    update_ntot();
    set_strides();
  }
  int num[3];
  ptrdiff_t the_stride[5];
  size_t the_ntot;
};

class volume_list;

class symmetry;
symmetry identity();
symmetry rotate4(direction, const grid_volume &);
symmetry rotate2(direction, const grid_volume &);
symmetry mirror(direction, const grid_volume &);
symmetry r_to_minus_r_symmetry(double m);

class symmetry {
public:
  symmetry();
  symmetry(const symmetry &);
  ~symmetry();
  friend symmetry identity();
  friend symmetry rotate4(direction, const grid_volume &);
  friend symmetry rotate2(direction, const grid_volume &);
  friend symmetry mirror(direction, const grid_volume &);

  signed_direction transform(direction d, int n) const;
  ivec transform(const ivec &, int n) const;
  vec transform(const vec &, int n) const;
  ivec transform_unshifted(const ivec &, int n) const;
  volume transform(const volume &, int n) const;
  component transform(component, int n) const;
  std::complex<double> phase_shift(component, int n) const;
  derived_component transform(derived_component, int n) const;
  std::complex<double> phase_shift(derived_component, int n) const;
  int transform(int, int n) const;
  std::complex<double> phase_shift(int, int n) const;
  int multiplicity() const;
  int multiplicity(ivec &) const;
  bool is_primitive(const ivec &) const;

  volume_list *reduce(const volume_list *gl) const;

  symmetry operator+(const symmetry &) const;
  symmetry operator*(std::complex<double>)const;
  symmetry operator-(const symmetry &b) const { return *this + b * (-1.0); }
  symmetry operator-(void) const { return *this * (-1.0); }
  void operator=(const symmetry &);
  bool operator==(const symmetry &) const;
  bool operator!=(const symmetry &S) const { return !(*this == S); };

private:
  signed_direction S[5];
  std::complex<double> ph;
  vec symmetry_point;
  ivec i_symmetry_point;
  int g; // g is the multiplicity of the symmetry.
  symmetry *next;
  friend symmetry r_to_minus_r_symmetry(double m);
};

class volume_list {
public:
  volume_list(const volume &v, int c, std::complex<double> weight = 1.0, volume_list *next = 0)
      : v(v), c(c), weight(weight), next(next) {}
  ~volume_list() { delete next; }
  volume_list(const volume_list *vl) : v(vl->v), c(vl->c), weight(vl->weight), next(0) {
    volume_list *p = vl->next, *q = this;
    while (p) {
      q->next = new volume_list(*p);
      q = q->next;
      p = p->next;
    }
  }

  volume v;
  int c; // component or derived component associated with v (e.g. for flux)
  std::complex<double> weight;
  volume_list *next;
};

} /* namespace meep */

#endif /* MEEP_VEC_H */
