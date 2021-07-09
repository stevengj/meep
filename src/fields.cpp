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

#include <algorithm>
#include <utility>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex>

#include "meep.hpp"
#include "meep_internals.hpp"

using namespace std;

namespace meep {

fields::fields(structure *s, double m, double beta, bool zero_fields_near_cylorigin)
    : S(s->S), gv(s->gv), user_volume(s->user_volume), v(s->v), m(m), beta(beta) {
  shared_chunks = s->shared_chunks;
  components_allocated = false;
  synchronized_magnetic_fields = 0;
  outdir = new char[strlen(s->outdir) + 1];
  strcpy(outdir, s->outdir);
  if (gv.dim == Dcyl) S = S + r_to_minus_r_symmetry(m);
  phasein_time = 0;
  for (int d = 0; d < 5; d++) {
    k[d] = 0.0;
    eikna[d] = 1.0;
  }
  is_real = 0;
  a = gv.a;
  dt = s->dt;
  t = 0;
  sources = NULL;
  fluxes = NULL;
  // Time stuff:
  reset_timers();
  last_step_output_wall_time = -1;

  num_chunks = s->num_chunks;
  typedef fields_chunk *fields_chunk_ptr;
  chunks = new fields_chunk_ptr[num_chunks];
  for (int i = 0; i < num_chunks; i++)
    chunks[i] = new fields_chunk(s->chunks[i], outdir, m, beta, zero_fields_near_cylorigin, i);
  FOR_FIELD_TYPES(ft) {
    for (int ip = 0; ip < 3; ip++) {
      comm_sizes[ft][ip] = new size_t[num_chunks * num_chunks];
      for (int i = 0; i < num_chunks * num_chunks; i++)
        comm_sizes[ft][ip][i] = 0;
    }
    typedef realnum *realnum_ptr;
    comm_blocks[ft] = new realnum_ptr[num_chunks * num_chunks];
    for (int i = 0; i < num_chunks * num_chunks; i++)
      comm_blocks[ft][i] = 0;
  }
  for (int b = 0; b < 2; b++)
    FOR_DIRECTIONS(d) {
      if (gv.has_boundary((boundary_side)b, d))
        boundaries[b][d] = Metallic;
      else
        boundaries[b][d] = None;
    }
  chunk_connections_valid = false;
  changed_materials = true;

  // unit directions are periodic by default:
  FOR_DIRECTIONS(d) {
    if (gv.has_boundary(High, d) && gv.has_boundary(Low, d) && d != R &&
        s->user_volume.num_direction(d) == 1)
      use_bloch(d, 0.0);
  }
}

fields::fields(const fields &thef)
    : S(thef.S), gv(thef.gv), user_volume(thef.user_volume), v(thef.v) {
  shared_chunks = thef.shared_chunks;
  components_allocated = thef.components_allocated;
  synchronized_magnetic_fields = thef.synchronized_magnetic_fields;
  outdir = new char[strlen(thef.outdir) + 1];
  strcpy(outdir, thef.outdir);
  m = thef.m;
  beta = thef.beta;
  phasein_time = thef.phasein_time;
  for (int d = 0; d < 5; d++) {
    k[d] = thef.k[d];
    eikna[d] = thef.eikna[d];
  }
  is_real = thef.is_real;
  a = thef.a;
  dt = thef.dt;
  t = thef.t;
  sources = NULL;
  fluxes = NULL;
  // Time stuff:
  reset_timers();
  last_step_output_wall_time = -1;

  num_chunks = thef.num_chunks;
  typedef fields_chunk *fields_chunk_ptr;
  chunks = new fields_chunk_ptr[num_chunks];
  for (int i = 0; i < num_chunks; i++)
    chunks[i] = new fields_chunk(*thef.chunks[i], i);
  FOR_FIELD_TYPES(ft) {
    for (int ip = 0; ip < 3; ip++) {
      comm_sizes[ft][ip] = new size_t[num_chunks * num_chunks];
      for (int i = 0; i < num_chunks * num_chunks; i++)
        comm_sizes[ft][ip][i] = 0;
    }
    typedef realnum *realnum_ptr;
    comm_blocks[ft] = new realnum_ptr[num_chunks * num_chunks];
    for (int i = 0; i < num_chunks * num_chunks; i++)
      comm_blocks[ft][i] = 0;
  }
  for (int b = 0; b < 2; b++)
    FOR_DIRECTIONS(d) { boundaries[b][d] = thef.boundaries[b][d]; }
  chunk_connections_valid = false;
  changed_materials = true;
}

fields::~fields() {
  for (int i = 0; i < num_chunks; i++)
    delete chunks[i];
  delete[] chunks;
  FOR_FIELD_TYPES(ft) {
    for (int i = 0; i < num_chunks * num_chunks; i++)
      delete[] comm_blocks[ft][i];
    delete[] comm_blocks[ft];
    for (int ip = 0; ip < 3; ip++)
      delete[] comm_sizes[ft][ip];
  }
  delete sources;
  delete fluxes;
  delete[] outdir;
}

void fields::use_real_fields() {
  LOOP_OVER_DIRECTIONS(gv.dim, d) {
    if (boundaries[High][d] == Periodic && k[d] != 0.0)
      meep::abort("Can't use real fields with bloch boundary conditions!\n");
  }
  is_real = 1;
  for (int i = 0; i < num_chunks; i++)
    chunks[i]->use_real_fields();

  // don't need to call sync_chunk_connections() since use_real_fields()
  // should always be called on every process
  chunk_connections_valid = false;
}

bool fields::have_component(component c) {
  for (int i = 0; i < num_chunks; i++)
    if (chunks[i]->f[c][0]) return true;
  return false;
}

fields_chunk::~fields_chunk() {
  is_real = 0; // So that we can make sure to delete everything...
  // for mu=1 non-PML regions, H==B to save space/time - don't delete twice!
  DOCMP2 FOR_H_AND_B(hc, bc) {
    if (f[hc][cmp] == f[bc][cmp]) f[bc][cmp] = NULL;
  }
  DOCMP2 FOR_COMPONENTS(c) {
    delete[] f[c][cmp];
    delete[] f_u[c][cmp];
    delete[] f_w[c][cmp];
    delete[] f_cond[c][cmp];
    delete[] f_minus_p[c][cmp];
    delete[] f_w_prev[c][cmp];
    delete[] f_backup[c][cmp];
    delete[] f_u_backup[c][cmp];
    delete[] f_w_backup[c][cmp];
    delete[] f_cond_backup[c][cmp];
  }
  delete[] f_rderiv_int;
  FOR_FIELD_TYPES(ft) {
    for (int ip = 0; ip < 3; ip++)
      for (int io = 0; io < 2; io++)
        delete[] connections[ft][ip][io];
  }
  FOR_FIELD_TYPES(ft) { delete[] connection_phases[ft]; }
  while (dft_chunks) {
    dft_chunk *nxt = dft_chunks->next_in_chunk;
    delete dft_chunks;
    dft_chunks = nxt;
  }
  FOR_FIELD_TYPES(ft) {
    delete[] zeroes[ft];
  }
  FOR_FIELD_TYPES(ft) {
    for (polarization_state *cur = pol[ft]; cur;) {
      polarization_state *p = cur;
      cur = cur->next;
      p->s->delete_internal_data(p->data);
      delete p;
    }
  }
  if (s->refcount-- <= 1) delete s;                  // delete if not shared
  if (new_s && new_s->refcount-- <= 1) delete new_s; // delete if not shared
}

static void split_into_tiles(grid_volume gvol, std::vector<grid_volume> *result) {
  const size_t base_nowned_min = 20000;
  if (gvol.nowned_min() < base_nowned_min) {
    result->push_back(gvol);
    return;
  }

  int best_split_point;
  direction best_split_direction;
  double left_effort_fraction;
  gvol.find_best_split(2, false /* fragment_cost */, best_split_point, best_split_direction,
                       left_effort_fraction);
  grid_volume left_gvol = gvol.split_at_fraction(false, best_split_point, best_split_direction);
  split_into_tiles(left_gvol, result);
  grid_volume right_gvol = gvol.split_at_fraction(true, best_split_point, best_split_direction);
  split_into_tiles(right_gvol, result);
  return;
}

// First check that the tile volumes gvs do not intersect and that they add
// up to the chunk's total grid_volume gv
static void check_tiles(grid_volume gv, const std::vector<grid_volume> &gvs) {
  grid_volume vol_intersection;
  for (size_t i = 0; i < gvs.size(); i++)
    for (size_t j = i + 1; j < gvs.size(); j++)
      if (gvs[i].intersect_with(gvs[j], &vol_intersection))
        meep::abort("gvs[%zu] intersects with gvs[%zu]\n", i, j);
  size_t sum = 0;
  for (const auto& sub_gv : gvs) {
    size_t grid_points = 1;
    LOOP_OVER_DIRECTIONS(sub_gv.dim, d) { grid_points *= sub_gv.num_direction(d); }
    sum += grid_points;
  }
  size_t v_grid_points = 1;
  LOOP_OVER_DIRECTIONS(gv.dim, d) { v_grid_points *= gv.num_direction(d); }
  if (sum != v_grid_points) meep::abort("v_grid_points = %zu, sum(chunks) = %zu\n", v_grid_points, sum);
}

fields_chunk::fields_chunk(structure_chunk *the_s, const char *od, double m, double beta,
                           bool zero_fields_near_cylorigin, int chunkidx)
    : gv(the_s->gv), v(the_s->v), m(m), zero_fields_near_cylorigin(zero_fields_near_cylorigin),
      beta(beta) {
  s = the_s;
  chunk_idx = chunkidx;
  s->refcount++;
  outdir = od;
  new_s = NULL;
  is_real = 0;
  a = s->a;
  Courant = s->Courant;
  dt = s->dt;
  dft_chunks = NULL;
  split_into_tiles(gv, &gvs);
  check_tiles(gv, gvs);
  FOR_FIELD_TYPES(ft) {
    polarization_state *cur = NULL;
    pol[ft] = NULL;
    for (susceptibility *chiP = the_s->chiP[ft]; chiP; chiP = chiP->next) {
      polarization_state *p = new polarization_state;
      // P and data lazily allocated in update_pols
      p->data = NULL;
      p->s = chiP;
      p->next = NULL;
      if (cur) {
        cur->next = p;
        cur = p;
      }
      else {
        pol[ft] = cur = p;
      }
    }
  }
  doing_solve_cw = false;
  solve_cw_omega = 0.0;
  FOR_COMPONENTS(c) DOCMP2 {
    f[c][cmp] = NULL;
    f_u[c][cmp] = NULL;
    f_w[c][cmp] = NULL;
    f_cond[c][cmp] = NULL;
    f_minus_p[c][cmp] = NULL;
    f_w_prev[c][cmp] = NULL;
    f_backup[c][cmp] = NULL;
    f_u_backup[c][cmp] = NULL;
    f_w_backup[c][cmp] = NULL;
    f_cond_backup[c][cmp] = NULL;
  }
  f_rderiv_int = NULL;
  FOR_FIELD_TYPES(ft) {
    for (int ip = 0; ip < 3; ip++)
      num_connections[ft][ip][Incoming] = num_connections[ft][ip][Outgoing] = 0;
    connection_phases[ft] = 0;
    for (int ip = 0; ip < 3; ip++)
      for (int io = 0; io < 2; io++)
        connections[ft][ip][io] = NULL;
    zeroes[ft] = NULL;
    num_zeroes[ft] = 0;
  }
  figure_out_step_plan();
}

fields_chunk::fields_chunk(const fields_chunk &thef, int chunkidx) : gv(thef.gv), v(thef.v) {
  chunk_idx = chunkidx;
  s = thef.s;
  s->refcount++;
  outdir = thef.outdir;
  m = thef.m;
  zero_fields_near_cylorigin = thef.zero_fields_near_cylorigin;
  beta = thef.beta;
  new_s = thef.new_s;
  new_s->refcount++;
  is_real = thef.is_real;
  a = thef.a;
  Courant = thef.Courant;
  dt = thef.dt;
  dft_chunks = NULL;
  gvs = thef.gvs;
  FOR_FIELD_TYPES(ft) {
    polarization_state *cur = NULL;
    for (polarization_state *ocur = thef.pol[ft]; ocur; ocur = ocur->next) {
      polarization_state *p = new polarization_state;
      p->data = NULL;
      p->s = ocur->s;
      p->next = NULL;
      pol[ft] = NULL;
      if (ocur->data) p->data = p->s->copy_internal_data(p->data);
      if (cur) {
        cur->next = p;
        cur = p;
      }
      else {
        pol[ft] = cur = p;
      }
    }
  }
  doing_solve_cw = thef.doing_solve_cw;
  solve_cw_omega = thef.solve_cw_omega;
  FOR_COMPONENTS(c) DOCMP2 {
    f[c][cmp] = NULL;
    f_u[c][cmp] = NULL;
    f_w[c][cmp] = NULL;
    f_cond[c][cmp] = NULL;
    f_backup[c][cmp] = NULL;
    f_u_backup[c][cmp] = NULL;
    f_w_backup[c][cmp] = NULL;
    f_cond_backup[c][cmp] = NULL;
  }
  FOR_COMPONENTS(c) DOCMP {
    if (!is_magnetic(c) && thef.f[c][cmp]) {
      f[c][cmp] = new realnum[gv.ntot()];
      memcpy(f[c][cmp], thef.f[c][cmp], sizeof(realnum) * gv.ntot());
    }
    if (thef.f_u[c][cmp]) {
      f_u[c][cmp] = new realnum[gv.ntot()];
      memcpy(f_u[c][cmp], thef.f_u[c][cmp], sizeof(realnum) * gv.ntot());
    }
    if (thef.f_w[c][cmp]) {
      f_w[c][cmp] = new realnum[gv.ntot()];
      memcpy(f_w[c][cmp], thef.f_w[c][cmp], sizeof(realnum) * gv.ntot());
    }
    if (thef.f_cond[c][cmp]) {
      f_cond[c][cmp] = new realnum[gv.ntot()];
      memcpy(f_cond[c][cmp], thef.f_cond[c][cmp], sizeof(realnum) * gv.ntot());
    }
  }
  FOR_MAGNETIC_COMPONENTS(c) DOCMP {
    if (thef.f[c][cmp] == thef.f[c - Hx + Bx][cmp])
      f[c][cmp] = f[c - Hx + Bx][cmp];
    else if (thef.f[c][cmp]) {
      f[c][cmp] = new realnum[gv.ntot()];
      memcpy(f[c][cmp], thef.f[c][cmp], sizeof(realnum) * gv.ntot());
    }
  }
  FOR_FIELD_TYPES(ft) {
    for (int ip = 0; ip < 3; ip++)
      num_connections[ft][ip][Incoming] = num_connections[ft][ip][Outgoing] = 0;
    connection_phases[ft] = 0;
    for (int ip = 0; ip < 3; ip++)
      for (int io = 0; io < 2; io++)
        connections[ft][ip][io] = NULL;
    zeroes[ft] = NULL;
    num_zeroes[ft] = 0;
  }
  FOR_COMPONENTS(c) DOCMP2 {
    if (thef.f_minus_p[c][cmp]) {
      f_minus_p[c][cmp] = new realnum[gv.ntot()];
      memcpy(f_minus_p[c][cmp], thef.f_minus_p[c][cmp], sizeof(realnum) * gv.ntot());
    }
    if (thef.f_w_prev[c][cmp]) {
      f_w_prev[c][cmp] = new realnum[gv.ntot()];
      memcpy(f_w_prev[c][cmp], thef.f_w_prev[c][cmp], sizeof(realnum) * gv.ntot());
    }
  }
  f_rderiv_int = NULL;
  figure_out_step_plan();
}

static inline bool cross_negative(direction a, direction b) {
  if (a >= R) a = direction(a - 3);
  if (b >= R) b = direction(b - 3);
  return ((3 + b - a) % 3) == 2;
}

static inline direction cross(direction a, direction b) {
  if (a == b) meep::abort("bug - cross expects different directions");
  bool dcyl = a >= R || b >= R;
  if (a >= R) a = direction(a - 3);
  if (b >= R) b = direction(b - 3);
  direction c = direction((3 + 2 * a - b) % 3);
  if (dcyl && c < Z) return direction(c + 3);
  return c;
}

/* Call this whenever we modify the structure_chunk (fields_chunk::s) to
   implement copy-on-write semantics.  See also structure::changing_chunks. */
void fields_chunk::changing_structure() {
  if (s->refcount > 1) { // this chunk is shared, so make a copy
    s->refcount--;
    s = new structure_chunk(s);
  }
}

void fields::figure_out_step_plan() {
  for (int i = 0; i < num_chunks; ++i)
    if (chunks[i]->is_mine()) chunks[i]->figure_out_step_plan();
}

void fields_chunk::figure_out_step_plan() {
  FOR_COMPONENTS(cc) { have_minus_deriv[cc] = have_plus_deriv[cc] = false; }
  FOR_COMPONENTS(c1) {
    if (f[c1][0]) {
      const direction dc1 = component_direction(c1);
      // Figure out which field components contribute.
      FOR_COMPONENTS(c2)
      if ((is_electric(c1) && is_magnetic(c2)) || (is_D(c1) && is_magnetic(c2)) ||
          (is_magnetic(c1) && is_electric(c2)) || (is_B(c1) && is_electric(c2))) {
        const direction dc2 = component_direction(c2);
        if (dc1 != dc2 && gv.has_field(c2) && gv.has_field(c1) &&
            (has_direction(gv.dim, cross(dc1, dc2)) ||
             (gv.dim == Dcyl && has_field_direction(gv.dim, cross(dc1, dc2))))) {
          direction d_deriv = cross(dc1, dc2);
          if (cross_negative(dc2, dc1)) {
            minus_component[c1] = c2;
            have_minus_deriv[c1] = true;
            minus_deriv_direction[c1] = d_deriv;
          }
          else {
            plus_component[c1] = c2;
            have_plus_deriv[c1] = true;
            plus_deriv_direction[c1] = d_deriv;
          }
        }
      }
    }
  }
}

bool is_tm(component c) {
  switch (c) {
    case Hx:
    case Hy:
    case Bx:
    case By:
    case Ez:
    case Dz: return true;
    default: return false;
  }
  return false;
}

static bool is_like(ndim d, component c1, component c2) {
  if (d != D2) return true;
  return !(is_tm(c1) ^ is_tm(c2));
}

// this function should ordinarily not be called directly;
// instead it should be called via require_component,
// since only require_component knows what other field components
// need to be allocated in addition to c
bool fields_chunk::alloc_f(component c) {
  bool changed = false;
  if (is_mine()) DOCMP {
      if (!f[c][cmp]) {
        changed = true;
        if (is_magnetic(c)) {
          /* initially, we just set H == B ... later on, we lazily allocate
             H fields if needed (if mu != 1 or in PML) in update_eh */
          component bc = direction_component(Bx, component_direction(c));
          if (!f[bc][cmp]) {
            f[bc][cmp] = new realnum[gv.ntot()];
            for (size_t i = 0; i < gv.ntot(); i++)
              f[bc][cmp][i] = 0.0;
          }
          f[c][cmp] = f[bc][cmp];
        }
        else {
          f[c][cmp] = new realnum[gv.ntot()];
          for (size_t i = 0; i < gv.ntot(); i++)
            f[c][cmp][i] = 0.0;
        }
      }
    }
  return changed;
}

// allocate fields for components required by any source on any process
// ... this is needed after calling the low-level fields::add_srcdata
void fields::require_source_components() {
  int needed[NUM_FIELD_COMPONENTS];
  memset(needed, 0, sizeof(needed));
  for (int i = 0; i < num_chunks; i++) {
    FOR_FIELD_TYPES(ft) {
      for (const auto& src : chunks[i]->get_sources(ft)) {
        needed[src.c] = 1;
      }
    }
  }
  int allneeded[NUM_FIELD_COMPONENTS];
  am_now_working_on(MpiAllTime);
  or_to_all(needed, allneeded, NUM_FIELD_COMPONENTS);
  finished_working();

  bool aniso2d = is_aniso2d();
  for (int c = 0; c < NUM_FIELD_COMPONENTS; ++c)
    if (allneeded[c])
      _require_component(component(c), aniso2d);
  sync_chunk_connections();
}

// check if we are in 2d but anisotropy couples xy with z
bool fields::is_aniso2d() {
  bool aniso2d = false;
  if (gv.dim == D2) {
    int i;
    for (i = 0; i < num_chunks; ++i)
      if (chunks[i]->s->has_chi(Ex, Z) || chunks[i]->s->has_chi(Ey, Z) ||
          chunks[i]->s->has_chi(Ez, X) || chunks[i]->s->has_chi(Ez, Y) ||
          chunks[i]->s->has_chi(Hx, Z) || chunks[i]->s->has_chi(Hy, Z) ||
          chunks[i]->s->has_chi(Hz, X) || chunks[i]->s->has_chi(Hz, Y))
        break;
    am_now_working_on(MpiAllTime);
    aniso2d = or_to_all(i < num_chunks);
    finished_working();
  }
  else if (beta != 0)
    meep::abort("Nonzero beta unsupported in dimensions other than 2.");
  if (aniso2d && beta != 0 && is_real)
    meep::abort("Nonzero beta need complex fields when mu/epsilon couple TE and TM");
  return aniso2d || (beta != 0); // beta couples TE/TM
}

void fields::_require_component(component c, bool aniso2d) {
  if (!gv.has_field(c))
    meep::abort("cannot require a %s component in a %s grid", component_name(c), dimension_name(gv.dim));

  components_allocated = true;

  // allocate fields if they haven't been allocated yet for this component
  int need_to_reconnect = 0;
  FOR_COMPONENTS(c_alloc) {
    if (gv.has_field(c_alloc) && (is_like(gv.dim, c, c_alloc) || aniso2d))
      for (int i = 0; i < num_chunks; ++i)
        if (chunks[i]->alloc_f(c_alloc)) need_to_reconnect++;
  }

  if (need_to_reconnect) {
    figure_out_step_plan();
    // we will eventually call sync_chunk_connections(), in either require_component(c)
    // or require_components(), to synchronize this across processes:
    chunk_connections_valid = false;
  }
}

void fields_chunk::add_source(field_type ft, src_vol &&src) {
  auto it = std::find_if(sources[ft].begin(), sources[ft].end(), [&src](const src_vol &other) {
    return src_vol::combinable(src, other);
  });

  if (it != sources[ft].end()) {
    it->add_amplitudes_from(src);
    return;
  }

  sources[ft].push_back(std::move(src));
}

void fields_chunk::remove_sources() {
  FOR_FIELD_TYPES(ft) {
    sources[ft].clear();
  }
}

void fields::remove_sources() {
  delete sources;
  sources = NULL;
  for (int i = 0; i < num_chunks; i++)
    chunks[i]->remove_sources();
}

void fields_chunk::remove_susceptibilities(bool shared_chunks) {
  FOR_FIELD_TYPES(ft) {
    for (polarization_state *cur = pol[ft]; cur;) {
      polarization_state *p = cur;
      cur = cur->next;
      p->s->delete_internal_data(p->data);
      delete p;
    }
    pol[ft] = NULL;
  }

  if (!shared_chunks) { changing_structure(); }
  s->remove_susceptibilities();
}

void fields::remove_susceptibilities() {
  changed_materials = true;
  for (int i = 0; i < num_chunks; i++)
    chunks[i]->remove_susceptibilities(shared_chunks);
}

void fields::remove_fluxes() {
  delete fluxes;
  fluxes = NULL;
}

void fields_chunk::zero_fields() {
  FOR_COMPONENTS(c) DOCMP {
#define ZERO(array)                                                                                \
  if (array) memset(array, 0, sizeof(realnum) * gv.ntot())
    ZERO(f[c][cmp]);
    ZERO(f_u[c][cmp]);
    ZERO(f_w[c][cmp]);
    ZERO(f_cond[c][cmp]);
    ZERO(f_backup[c][cmp]);
    ZERO(f_u_backup[c][cmp]);
    ZERO(f_w_backup[c][cmp]);
    ZERO(f_cond_backup[c][cmp]);
#undef ZERO
  }
  if (is_mine()) FOR_FIELD_TYPES(ft) {
      for (polarization_state *p = pol[ft]; p; p = p->next) {
        if (p->data) p->s->init_internal_data(f, dt, gv, p->data);
      }
    }
}

void fields::zero_fields() {
  for (int i = 0; i < num_chunks; i++)
    chunks[i]->zero_fields();
}

void fields::reset() {
  remove_sources();
  remove_fluxes();
  zero_fields();
  t = 0;
}

void fields_chunk::use_real_fields() {
  is_real = 1;
  // for mu=1 non-PML regions, H==B to save space/time - don't delete twice!
  FOR_H_AND_B(hc, bc) {
    if (f[hc][1] == f[bc][1]) f[bc][1] = NULL;
  }
  FOR_COMPONENTS(c) if (f[c][1]) {
    delete[] f[c][1];
    f[c][1] = 0;
  }
  if (is_mine()) FOR_FIELD_TYPES(ft) {
      for (polarization_state *p = pol[ft]; p; p = p->next) {
        if (p->data) { // TODO: print an error message in this case?
          p->s->delete_internal_data(p->data);
          p->data = p->s->new_internal_data(f, gv);
          p->s->init_internal_data(f, dt, gv, p->data);
        }
      }
    }
}

int fields::phase_in_material(const structure *snew, double time) {
  if (snew->num_chunks != num_chunks)
    meep::abort("Can only phase in similar sets of chunks: %d vs %d\n", snew->num_chunks, num_chunks);
  for (int i = 0; i < num_chunks; i++)
    if (chunks[i]->is_mine()) chunks[i]->phase_in_material(snew->chunks[i]);
  phasein_time = (int)(time / dt);
  changed_materials = true;
  // FIXME: how to handle changes in susceptibilities?
  return phasein_time;
}

void fields_chunk::phase_in_material(structure_chunk *snew) {
  new_s = snew;
  new_s->refcount++;
}

int fields::is_phasing() { return phasein_time > 0; }

bool fields::equal_layout(const fields &f) const {
  if (a != f.a || num_chunks != f.num_chunks || v != f.v || S != f.S) return false;
  for (int d = 0; d < 5; d++)
    if (k[d] != f.k[d]) return false;
  for (int i = 0; i < num_chunks; ++i)
    if (chunks[i]->a != f.chunks[i]->a || chunks[i]->v != f.chunks[i]->v) return false;
  return true;
}

// total computational grid_volume, including regions redundant by symmetry
volume fields::total_volume(void) const {
  volume gv0 = gv.interior();
  volume v = gv0;
  for (int n = 1; n < S.multiplicity(); ++n)
    v = v | S.transform(gv0, n);
  if (v.dim == Dcyl && v.in_direction_min(R) < 0) v.set_direction_min(R, 0);
  return v;
}

/* One-pixel periodic dimensions are used almost exclusively to
   emulate lower-dimensional computations, so if the user passes an
   empty size in that direction, they probably really intended to
   specify that whole dimension.  This function detects that case. */
bool fields::nosize_direction(direction d) const {
  return (gv.has_boundary(Low, d) && gv.has_boundary(High, d) && boundaries[Low][d] == Periodic &&
          boundaries[High][d] == Periodic && gv.num_direction(d) == 1);
}

void fields::set_solve_cw_omega(complex<double> omega) {
  for (int i = 0; i < num_chunks; ++i)
    chunks[i]->set_solve_cw_omega(omega);
}

void fields::unset_solve_cw_omega() {
  for (int i = 0; i < num_chunks; ++i)
    chunks[i]->unset_solve_cw_omega();
}

/* implement mirror boundary conditions for i outside 0..n-1: */
int mirrorindex(int i, int n) { return i >= n ? 2 * n - 1 - i : (i < 0 ? -1 - i : i); }

/* map the cell coordinates into the range [0,1].
   anything outside [0,1] is *mirror* reflected into [0,1] */
void map_coordinates(double rx, double ry, double rz,
                     int nx, int ny, int nz,
                     int &x1, int &y1, int &z1,
                     int &x2, int &y2, int &z2,
                     double &dx, double &dy, double &dz,
                     bool do_fabs) {

  /* mirror boundary conditions for r just beyond the boundary */
  rx = rx < 0.0 ? -rx : (rx > 1.0 ? 1.0 - rx : rx);
  ry = ry < 0.0 ? -ry : (ry > 1.0 ? 1.0 - ry : ry);
  rz = rz < 0.0 ? -rz : (rz > 1.0 ? 1.0 - rz : rz);

  /* get the point corresponding to r in the epsilon array grid: */
  x1 = mirrorindex(int(rx * nx), nx);
  y1 = mirrorindex(int(ry * ny), ny);
  z1 = mirrorindex(int(rz * nz), nz);

  /* get the difference between (x,y,z) and the actual point */
  dx = rx * nx - x1 - 0.5;
  dy = ry * ny - y1 - 0.5;
  dz = rz * nz - z1 - 0.5;

  /* get the other closest point in the grid, with mirror boundaries: */
  x2 = mirrorindex(dx >= 0.0 ? x1 + 1 : x1 - 1, nx);
  y2 = mirrorindex(dy >= 0.0 ? y1 + 1 : y1 - 1, ny);
  z2 = mirrorindex(dz >= 0.0 ? z1 + 1 : z1 - 1, nz);

  /* take abs(d{xyz}) to get weights for {xyz} and {xyz}2: */
  if (do_fabs) {
    dx = fabs(dx);
    dy = fabs(dy);
    dz = fabs(dz);
  }

}

/* linearly interpolate a given point in a 3d grid of data. */
double linear_interpolate(double rx, double ry, double rz, double *data,
                          int nx, int ny, int nz, int stride) {

  int x1, y1, z1, x2, y2, z2;
  double dx, dy, dz;

  map_coordinates(rx, ry, rz, nx, ny, nz,
                  x1, y1, z1, x2, y2, z2,
                  dx, dy, dz);

  /* define a macro to give us data(x,y,z) on the grid,
     in row-major order (the order used by HDF5): */
#define D(x, y, z) (data[(((x)*ny + (y)) * nz + (z)) * stride])

  return (((D(x1, y1, z1) * (1.0 - dx) + D(x2, y1, z1) * dx) * (1.0 - dy) +
           (D(x1, y2, z1) * (1.0 - dx) + D(x2, y2, z1) * dx) * dy) *
              (1.0 - dz) +
          ((D(x1, y1, z2) * (1.0 - dx) + D(x2, y1, z2) * dx) * (1.0 - dy) +
           (D(x1, y2, z2) * (1.0 - dx) + D(x2, y2, z2) * dx) * dy) *
              dz);

#undef D
}

} // namespace meep
