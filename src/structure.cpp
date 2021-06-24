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
#include <memory>

#include "meep.hpp"
#include "meep_internals.hpp"
#include "meepgeom.hpp"

using namespace std;

namespace meep {

structure::structure()
    : Courant(0.5), v(D1) // Aaack, this is very hokey.
{
  num_chunks = 0;
  num_effort_volumes = 0;
  effort_volumes = NULL;
  effort = NULL;
  outdir = ".";
  S = identity();
  a = 1;
  dt = Courant / a;
  shared_chunks = false;
}

typedef structure_chunk *structure_chunk_ptr;

structure::structure(const grid_volume &thegv, material_function &eps, const boundary_region &br,
                     const symmetry &s, int num, double Courant, bool use_anisotropic_averaging,
                     double tol, int maxeval, const binary_partition *bp)
    : Courant(Courant), v(D1) // Aaack, this is very hokey.
{
  outdir = ".";
  shared_chunks = false;
  if (!br.check_ok(thegv)) abort("invalid boundary absorbers for this grid_volume");
  double tstart = wall_time();
  choose_chunkdivision(thegv, num, br, s, bp);
  if (verbosity > 0) master_printf("time for choose_chunkdivision = %g s\n", wall_time() - tstart);
  set_materials(eps, use_anisotropic_averaging, tol, maxeval);
}

structure::structure(const grid_volume &thegv, double eps(const vec &), const boundary_region &br,
                     const symmetry &s, int num, double Courant, bool use_anisotropic_averaging,
                     double tol, int maxeval, const binary_partition *bp)
    : Courant(Courant), v(D1) // Aaack, this is very hokey.
{
  outdir = ".";
  shared_chunks = false;
  if (!br.check_ok(thegv)) abort("invalid boundary absorbers for this grid_volume");
  double tstart = wall_time();
  choose_chunkdivision(thegv, num, br, s, bp);
  if (verbosity > 0) master_printf("time for choose_chunkdivision = %g s\n", wall_time() - tstart);
  if (eps) {
    simple_material_function epsilon(eps);
    set_materials(epsilon, use_anisotropic_averaging, tol, maxeval);
  }
}

static binary_partition *split_by_cost(int n, grid_volume gvol, bool fragment_cost) {
  if (n == 1) {
    binary_partition *bp_leaf = new binary_partition(-1);
    return bp_leaf;
  }
  else {
    int best_split_point;
    direction best_split_direction;
    double best_split_position;
    double left_effort_fraction;
    gvol.find_best_split(n, fragment_cost, best_split_point, best_split_direction, left_effort_fraction);
    best_split_position = gvol.surroundings().get_min_corner().in_direction(best_split_direction) +
      (gvol.surroundings().in_direction(best_split_direction) * best_split_point) /
      gvol.num_direction(best_split_direction);
    binary_partition *bp_split = new binary_partition(best_split_direction, best_split_position);
    grid_volume left_gvol = gvol.split_at_fraction(false, best_split_point, best_split_direction);
    const int num_left = (size_t)(left_effort_fraction * n + 0.5);
    bp_split->left = split_by_cost(num_left, left_gvol, fragment_cost);
    grid_volume right_gvol = gvol.split_at_fraction(true, best_split_point, best_split_direction);
    bp_split->right = split_by_cost(n - num_left, right_gvol, fragment_cost);
    return bp_split;
  }
}

void structure::choose_chunkdivision(const grid_volume &thegv, int desired_num_chunks,
                                     const boundary_region &br, const symmetry &s,
                                     const binary_partition *bp) {

  if (thegv.dim == Dcyl && thegv.get_origin().r() < 0) abort("r < 0 origins are not supported");

  user_volume = thegv;
  gv = thegv;
  v = gv.surroundings();
  S = s;
  a = gv.a;
  dt = Courant / a;

  std::unique_ptr<binary_partition> my_bp;
  if (!bp) my_bp.reset(meep::choose_chunkdivision(gv, v, desired_num_chunks, s));

  // create the chunks:
  std::vector<grid_volume> chunk_volumes;
  std::vector<int> ids;
  split_by_binarytree(gv, chunk_volumes, ids, (!bp) ? my_bp.get() : bp);

  // initialize effort volumes
  num_effort_volumes = 1;
  effort_volumes = new grid_volume[num_effort_volumes];
  effort_volumes[0] = gv;
  effort = new double[num_effort_volumes];
  effort[0] = 1.0;

  // Next, add effort volumes for PML boundary regions:
  br.apply(this);

  // Break off PML regions into their own chunks
  num_chunks = 0;
  chunks = new structure_chunk_ptr[chunk_volumes.size() * num_effort_volumes];
  for (size_t i = 0, stop = chunk_volumes.size(); i < stop; ++i) {
    const int proc = (!bp) ? i * count_processors() / chunk_volumes.size() : ids[i] % count_processors();
    for (int j = 0; j < num_effort_volumes; ++j) {
      grid_volume vc;
      if (chunk_volumes[i].intersect_with(effort_volumes[j], &vc)) {
        chunks[num_chunks] = new structure_chunk(vc, v, Courant, proc);
        br.apply(this, chunks[num_chunks++]);
      }
    }
  }

  check_chunks();
  if (meep_geom::fragment_stats::resolution != 0) {
    // Save cost of each chunk's grid_volume
    for (int i = 0; i < num_chunks; ++i) {
      chunks[i]->cost = chunks[i]->gv.get_cost();
    }
  }

}

binary_partition *choose_chunkdivision(grid_volume &gv, volume &v, int desired_num_chunks,
                                       const symmetry &S) {

  if (desired_num_chunks == 0) desired_num_chunks = count_processors();
  if (gv.dim == Dcyl && gv.get_origin().r() < 0) abort("r < 0 origins are not supported");

  // First, reduce overall grid_volume gv by symmetries:
  if (S.multiplicity() > 1) {
    bool break_this[3];
    for (int dd = 0; dd < 3; dd++) {
      const direction d = (direction)dd;
      break_this[dd] = false;
      for (int n = 0; n < S.multiplicity(); n++)
        if (has_direction(gv.dim, d) &&
            (S.transform(d, n).d != d || S.transform(d, n).flipped)) {
          if (gv.num_direction(d) & 1 && !break_this[d] && verbosity > 0)
            master_printf("Padding %s to even number of grid points.\n", direction_name(d));
          break_this[dd] = true;
        }
    }
    int break_mult = 1;
    for (int d = 0; d < 3; d++) {
      if (break_mult == S.multiplicity()) break_this[d] = false;
      if (break_this[d]) {
        break_mult *= 2;
        if (verbosity > 0)
          master_printf("Halving computational cell along direction %s\n",
                        direction_name(direction(d)));
        gv = gv.halve((direction)d);
      }
    }
    // Before padding, find the corresponding geometric grid_volume.
    v = gv.surroundings();
    // Pad the little cell in any direction that we've shrunk:
    for (int d = 0; d < 3; d++)
      if (break_this[d]) gv = gv.pad((direction)d);
  }

  if (meep_geom::fragment_stats::resolution == 0 ||
      meep_geom::fragment_stats::split_chunks_evenly) {
    if (verbosity > 0 && desired_num_chunks > 1)
      master_printf("Splitting into %d chunks by voxels\n", desired_num_chunks);
    return split_by_cost(desired_num_chunks, gv, false);
  }
  else {
    if (verbosity > 0 && desired_num_chunks > 1)
      master_printf("Splitting into %d chunks by cost\n", desired_num_chunks);
    return split_by_cost(desired_num_chunks, gv, true);
  }

}

double structure::estimated_cost(int process) {
  double proc_cost = 0;
  for (int i = 0; i < num_chunks; i++) {
    if (chunks[i]->n_proc() == process) { proc_cost += chunks[i]->cost; }
  }
  return proc_cost;
}

void boundary_region::apply(structure *s) const {
  if (has_direction(s->gv.dim, d) && s->user_volume.has_boundary(side, d) &&
      s->user_volume.num_direction(d) > 1) {
    switch (kind) {
      case NOTHING_SPECIAL: break;
      case PML: s->use_pml(d, side, thickness); break;
      default: abort("unknown boundary region kind");
    }
  }
  if (next) next->apply(s);
}

void boundary_region::apply(const structure *s, structure_chunk *sc) const {
  if (has_direction(s->gv.dim, d) && s->user_volume.has_boundary(side, d) &&
      s->user_volume.num_direction(d) > 1) {
    switch (kind) {
      case NOTHING_SPECIAL: break;
      case PML:
        sc->use_pml(d, thickness, s->user_volume.boundary_location(side, d), Rasymptotic,
                    mean_stretch, pml_profile, pml_profile_data, pml_profile_integral,
                    pml_profile_integral_u);
        break;
      default: abort("unknown boundary region kind");
    }
  }
  if (next) next->apply(s, sc);
}

bool boundary_region::check_ok(const grid_volume &gv) const {
  double thick[5][2];
  FOR_DIRECTIONS(d) FOR_SIDES(s) { thick[d][s] = 0; }
  for (const boundary_region *r = this; r; r = r->next) {
    if (r->kind != NOTHING_SPECIAL && gv.num_direction(r->d) > 1 && has_direction(gv.dim, r->d) &&
        gv.has_boundary(r->side, r->d)) {
      if (r->thickness < 0 || thick[r->d][r->side] > 0) return false;
      thick[r->d][r->side] = r->thickness;
    }
  }
  LOOP_OVER_DIRECTIONS(gv.dim, d) {
    if (thick[d][High] + thick[d][Low] > gv.interior().in_direction(d)) return false;
  }
  return true;
}

double pml_quadratic_profile(double u, void *d) {
  (void)d;
  return u * u;
}

boundary_region pml(double thickness, direction d, boundary_side side, double Rasymptotic,
                    double mean_stretch) {
  return boundary_region(boundary_region::PML, thickness, Rasymptotic, mean_stretch,
                         pml_quadratic_profile, NULL, 1. / 3., 1. / 4., d, side, NULL);
}
boundary_region pml(double thickness, direction d, double Rasymptotic, double mean_stretch) {
  return (pml(thickness, d, Low, Rasymptotic, mean_stretch) +
          pml(thickness, d, High, Rasymptotic, mean_stretch));
}
boundary_region pml(double thickness, double Rasymptotic, double mean_stretch) {
  boundary_region r;
  for (int id = 0; id < 5; ++id)
    r = r + pml(thickness, (direction)id, Rasymptotic, mean_stretch);
  return r;
}

// First check that the chunk volumes do not intersect and that they add
// up to the total grid_volume
void structure::check_chunks() {
  grid_volume vol_intersection;
  for (int i = 0; i < num_chunks; i++)
    for (int j = i + 1; j < num_chunks; j++)
      if (chunks[i]->gv.intersect_with(chunks[j]->gv, &vol_intersection))
        abort("chunks[%d] intersects with chunks[%d]\n", i, j);
  size_t sum = 0;
  for (int i = 0; i < num_chunks; i++) {
    size_t grid_points = 1;
    LOOP_OVER_DIRECTIONS(chunks[i]->gv.dim, d) { grid_points *= chunks[i]->gv.num_direction(d); }
    sum += grid_points;
  }
  size_t v_grid_points = 1;
  LOOP_OVER_DIRECTIONS(gv.dim, d) { v_grid_points *= gv.num_direction(d); }
  if (sum != v_grid_points) abort("v_grid_points = %zd, sum(chunks) = %zd\n", v_grid_points, sum);
}

void structure::add_to_effort_volumes(const grid_volume &new_effort_volume, double extra_effort) {
  grid_volume *temp_volumes =
      new grid_volume[(2 * number_of_directions(gv.dim) + 1) * num_effort_volumes];
  double *temp_effort = new double[(2 * number_of_directions(gv.dim) + 1) * num_effort_volumes];
  // Intersect previous mat_volumes with this new_effort_volume
  int counter = 0;
  for (int j = 0; j < num_effort_volumes; j++) {
    grid_volume intersection, others[6];
    int num_others;
    if (effort_volumes[j].intersect_with(new_effort_volume, &intersection, others, &num_others)) {
      if (num_others > 1) {
        printf("effort_volumes[%d]  ", j);
        effort_volumes[j].print();
        printf("new_effort_volume  ");
        new_effort_volume.print();
        // NOTE: this may not be a bug if this function is used for
        // something other than PML.
        abort("Did not expect num_others > 1 in add_to_effort_volumes\n");
      }
      temp_effort[counter] = extra_effort + effort[j];
      temp_volumes[counter] = intersection;
      counter++;
      for (int k = 0; k < num_others; k++) {
        temp_effort[counter] = effort[j];
        temp_volumes[counter] = others[k];
        counter++;
      }
    }
    else {
      temp_effort[counter] = effort[j];
      temp_volumes[counter] = effort_volumes[j];
      counter++;
    }
  }

  delete[] effort_volumes;
  delete[] effort;
  effort_volumes = temp_volumes;
  effort = temp_effort;
  num_effort_volumes = counter;
}

structure::structure(const structure *s) : v(s->v) {
  shared_chunks = false;
  num_chunks = s->num_chunks;
  outdir = s->outdir;
  gv = s->gv;
  S = s->S;
  user_volume = s->user_volume;
  chunks = new structure_chunk_ptr[num_chunks];
  for (int i = 0; i < num_chunks; i++)
    chunks[i] = new structure_chunk(s->chunks[i]);
  num_effort_volumes = s->num_effort_volumes;
  effort_volumes = new grid_volume[num_effort_volumes];
  effort = new double[num_effort_volumes];
  for (int i = 0; i < num_effort_volumes; i++) {
    effort_volumes[i] = s->effort_volumes[i];
    effort[i] = s->effort[i];
  }
  a = s->a;
  Courant = s->Courant;
  dt = s->dt;
}

structure::structure(const structure &s) : v(s.v) {
  shared_chunks = false;
  num_chunks = s.num_chunks;
  outdir = s.outdir;
  gv = s.gv;
  S = s.S;
  user_volume = s.user_volume;
  chunks = new structure_chunk_ptr[num_chunks];
  for (int i = 0; i < num_chunks; i++) {
    chunks[i] = new structure_chunk(s.chunks[i]);
  }
  num_effort_volumes = s.num_effort_volumes;
  effort_volumes = new grid_volume[num_effort_volumes];
  effort = new double[num_effort_volumes];
  for (int i = 0; i < num_effort_volumes; i++) {
    effort_volumes[i] = s.effort_volumes[i];
    effort[i] = s.effort[i];
  }
  a = s.a;
  Courant = s.Courant;
  dt = s.dt;
}

structure::~structure() {
  for (int i = 0; i < num_chunks; i++) {
    if (chunks[i]->refcount-- <= 1) delete chunks[i];
    chunks[i] = NULL; // Just to be sure...
  }
  delete[] chunks;
  delete[] effort_volumes;
  delete[] effort;
}

/* To save memory, the structure chunks are shared with the
   fields_chunk objects instead of making a copy.  However, to
   preserve the illusion that the structure and fields are
   independent objects, we implement copy-on-write semantics. */
void structure::changing_chunks() { // call this whenever chunks are modified
  if (shared_chunks) return;        // shared view of chunks with fields, no COW
  for (int i = 0; i < num_chunks; i++)
    if (chunks[i]->refcount > 1) { // this chunk is shared, so make a copy
      chunks[i]->refcount--;
      chunks[i] = new structure_chunk(chunks[i]);
    }
}

void structure::set_materials(material_function &mat, bool use_anisotropic_averaging, double tol,
                              int maxeval) {
  set_epsilon(mat, use_anisotropic_averaging, tol, maxeval);
  if (mat.has_mu()) set_mu(mat, use_anisotropic_averaging, tol, maxeval);
  FOR_D_AND_B(c) {
    if (mat.has_conductivity(c)) set_conductivity(c, mat);
  }
  FOR_E_AND_H(c) {
    if (mat.has_chi3(c)) set_chi3(c, mat);
  }
  FOR_E_AND_H(c) {
    if (mat.has_chi2(c)) set_chi2(c, mat);
  }
}

void structure::set_chi1inv(component c, material_function &eps, bool use_anisotropic_averaging,
                            double tol, int maxeval) {
  changing_chunks();
  for (int i = 0; i < num_chunks; i++)
    if (chunks[i]->is_mine())
      chunks[i]->set_chi1inv(c, eps, use_anisotropic_averaging, tol, maxeval);
}

void structure::set_epsilon(material_function &eps, bool use_anisotropic_averaging, double tol,
                            int maxeval) {
  double tstart = wall_time();
  FOR_ELECTRIC_COMPONENTS(c) { set_chi1inv(c, eps, use_anisotropic_averaging, tol, maxeval); }
  all_wait(); // sync so that timing results are accurate
  if (verbosity > 0) master_printf("time for set_epsilon = %g s\n", wall_time() - tstart);
}

void structure::set_epsilon(double eps(const vec &), bool use_anisotropic_averaging, double tol,
                            int maxeval) {
  simple_material_function epsilon(eps);
  set_epsilon(epsilon, use_anisotropic_averaging, tol, maxeval);
}

void structure::set_mu(material_function &m, bool use_anisotropic_averaging, double tol,
                       int maxeval) {
  double tstart = wall_time();
  FOR_MAGNETIC_COMPONENTS(c) { set_chi1inv(c, m, use_anisotropic_averaging, tol, maxeval); }
  all_wait(); // sync so that timing results are accurate
  if (verbosity > 0) master_printf("time for set_mu = %g s\n", wall_time() - tstart);
}

void structure::set_mu(double mufunc(const vec &), bool use_anisotropic_averaging, double tol,
                       int maxeval) {
  simple_material_function mu(mufunc);
  set_mu(mu, use_anisotropic_averaging, tol, maxeval);
}

void structure::set_conductivity(component c, material_function &C) {
  if (!gv.has_field(c)) return;
  double tstart = wall_time();
  changing_chunks();
  for (int i = 0; i < num_chunks; i++)
    if (chunks[i]->is_mine()) chunks[i]->set_conductivity(c, C);
  all_wait(); // sync so that timing results are accurate
  if (verbosity > 0) master_printf("time for set_conductivity = %g s\n", wall_time() - tstart);
}

void structure::set_conductivity(component c, double Cfunc(const vec &)) {
  simple_material_function conductivity(Cfunc);
  set_conductivity(c, conductivity);
}

void structure::set_chi3(component c, material_function &eps) {
  if (!gv.has_field(c)) return;
  changing_chunks();
  for (int i = 0; i < num_chunks; i++)
    if (chunks[i]->is_mine()) chunks[i]->set_chi3(c, eps);
}

void structure::set_chi3(material_function &eps) {
  FOR_ELECTRIC_COMPONENTS(c) { set_chi3(c, eps); }
}

void structure::set_chi3(double eps(const vec &)) {
  simple_material_function epsilon(eps);
  set_chi3(epsilon);
}

void structure::set_chi2(component c, material_function &eps) {
  changing_chunks();
  for (int i = 0; i < num_chunks; i++)
    if (chunks[i]->is_mine()) chunks[i]->set_chi2(c, eps);
}

void structure::set_chi2(material_function &eps) {
  FOR_ELECTRIC_COMPONENTS(c) { set_chi2(c, eps); }
}

void structure::set_chi2(double eps(const vec &)) {
  simple_material_function epsilon(eps);
  set_chi2(epsilon);
}

void structure::add_susceptibility(double sigma(const vec &), field_type ft,
                                   const susceptibility &sus) {
  simple_material_function sig(sigma);
  add_susceptibility(sig, ft, sus);
}

void structure::add_susceptibility(material_function &sigma, field_type ft,
                                   const susceptibility &sus) {
  changing_chunks();
  for (int i = 0; i < num_chunks; i++)
    chunks[i]->add_susceptibility(sigma, ft, sus);

  /* Now, synchronize the trivial_sigma array among all
     chunks/processes.  This will result in some "wasted" memory: if a
     particular polarization P is needed on *any* chunk, it will be
     allocated on *every* chunk.  However, this greatly simplifies
     handling of boundary conditions between chunks; see also the
     susceptibility::needs_P function.  (Note that the new
     susceptibility object was added to the beginning of each chunk's
     chiP[ft] list.) */
  int trivial_sigma[NUM_FIELD_COMPONENTS][5];
  FOR_COMPONENTS(c) FOR_DIRECTIONS(d) { trivial_sigma[c][d] = true; }
  for (int i = 0; i < num_chunks; i++) {
    const susceptibility *newsus = chunks[i]->chiP[ft];
    FOR_FT_COMPONENTS(ft, c) FOR_DIRECTIONS(d) {
      trivial_sigma[c][d] = trivial_sigma[c][d] && newsus->trivial_sigma[c][d];
    }
  }
  int trivial_sigma_sync[NUM_FIELD_COMPONENTS][5];
  and_to_all(&trivial_sigma[0][0], &trivial_sigma_sync[0][0], NUM_FIELD_COMPONENTS * 5);
  for (int i = 0; i < num_chunks; i++) {
    susceptibility *newsus = chunks[i]->chiP[ft];
    FOR_FT_COMPONENTS(ft, c) FOR_DIRECTIONS(d) {
      newsus->trivial_sigma[c][d] = trivial_sigma_sync[c][d];
    }
  }
}

void structure::use_pml(direction d, boundary_side b, double dx) {
  if (dx <= 0.0) return;
  grid_volume pml_volume = gv;
  pml_volume.set_num_direction(d, int(dx * user_volume.a + 1 + 0.5)); // FIXME: exact value?
  if (b == High)
    pml_volume.set_origin(d, user_volume.big_corner().in_direction(d) -
                                 pml_volume.num_direction(d) * 2);
  const int v_to_user_shift =
      (user_volume.little_corner().in_direction(d) - gv.little_corner().in_direction(d)) / 2;
  if (b == Low && v_to_user_shift != 0)
    pml_volume.set_num_direction(d, pml_volume.num_direction(d) + v_to_user_shift);
  add_to_effort_volumes(pml_volume, 0.60); // FIXME: manual value for pml effort
}

bool structure::has_chi(component c, direction d) const {
  int i;
  for (i = 0; i < num_chunks && !chunks[i]->has_chi(c, d); i++)
    ;
  return or_to_all(i < num_chunks);
}

bool structure_chunk::has_chi(component c, direction d) const {
  return has_chisigma(c, d) || has_chi1inv(c, d);
}

bool structure_chunk::has_chisigma(component c, direction d) const {
  if (is_mine()) {
    for (susceptibility *sus = chiP[type(c)]; sus; sus = sus->next)
      if (sus->sigma[c][d] && !sus->trivial_sigma[c][d]) return true;
  }
  return false;
}

bool structure_chunk::has_chi1inv(component c, direction d) const {
  return is_mine() && chi1inv[c][d] && !trivial_chi1inv[c][d];
}

void structure::mix_with(const structure *oth, double f) {
  if (num_chunks != oth->num_chunks)
    abort("You can't phase materials with different chunk topologies...\n");
  changing_chunks();
  for (int i = 0; i < num_chunks; i++)
    if (chunks[i]->is_mine()) chunks[i]->mix_with(oth->chunks[i], f);
}

structure_chunk::~structure_chunk() {
  FOR_COMPONENTS(c) {
    FOR_DIRECTIONS(d) {
      delete[] chi1inv[c][d];
      delete[] conductivity[c][d];
      delete[] condinv[c][d];
    }
    delete[] chi2[c];
    delete[] chi3[c];
  }
  FOR_DIRECTIONS(d) {
    delete[] sig[d];
    delete[] kap[d];
    delete[] siginv[d];
  }
  FOR_FIELD_TYPES(ft) { delete chiP[ft]; }
}

void structure_chunk::mix_with(const structure_chunk *n, double f) {
  FOR_COMPONENTS(c) FOR_DIRECTIONS(d) {
    if (!chi1inv[c][d] && n->chi1inv[c][d]) {
      chi1inv[c][d] = new realnum[gv.ntot()];
      trivial_chi1inv[c][d] = n->trivial_chi1inv[c][d];
      if (component_direction(c) == d) // diagonal components = 1 by default
        PS1LOOP_OVER_VOL(gv, c, i)
          chi1inv[c][d][i] = 1.0;
      else
        PS1LOOP_OVER_VOL(gv, c, i)
          chi1inv[c][d][i] = 0.0;
    }
    if (!conductivity[c][d] && n->conductivity[c][d]) {
      conductivity[c][d] = new realnum[gv.ntot()];
      PS1LOOP_OVER_VOL(gv, c, i)
        conductivity[c][d][i] = 0.0;
    }
    if (chi1inv[c][d]) {
      trivial_chi1inv[c][d] = trivial_chi1inv[c][d] && n->trivial_chi1inv[c][d];
      if (n->chi1inv[c][d])
        PS1LOOP_OVER_VOL(gv, c, i)
          chi1inv[c][d][i] += f * (n->chi1inv[c][d][i] - chi1inv[c][d][i]);
      else {
        double nval = component_direction(c) == d ? 1.0 : 0.0; // default
        PS1LOOP_OVER_VOL(gv, c, i)
          chi1inv[c][d][i] += f * (nval - chi1inv[c][d][i]);
      }
    }
    if (conductivity[c][d]) {
      if (n->conductivity[c][d])
        PS1LOOP_OVER_VOL(gv, c, i)
          conductivity[c][d][i] += f * (n->conductivity[c][d][i] - conductivity[c][d][i]);
      else
        PS1LOOP_OVER_VOL(gv, c, i)
          conductivity[c][d][i] += f * (0.0 - conductivity[c][d][i]);
    }
    condinv_stale = true;
  }
  // Mix in the susceptibility....FIXME.
}

static inline double pml_x(int i, double dx, double bloc, double a) {
  double here = i * 0.5 / a;
  return (0.5 / a * ((int)(dx * (2 * a) + 0.5) - (int)(fabs(bloc - here) * (2 * a) + 0.5)));
}

void structure_chunk::use_pml(direction d, double dx, double bloc, double Rasymptotic,
                              double mean_stretch, pml_profile_func pml_profile,
                              void *pml_profile_data, double pml_profile_integral,
                              double pml_profile_integral_u) {
  if (dx <= 0.0) return;
  const double prefac = (-log(Rasymptotic)) / (4 * dx * pml_profile_integral);
  /* The sigma term scales as 1/dx, since Rasymptotic is fixed.  To
     give the same thickness scaling of the transition reflections,
     the kappa (stretch) term must be *smoother* by one derivative
     than the sigma term.  [See Oskooi et al, Opt. Express 16,
     p. 11376 (2008)].  We accomplish this by making the kappa term
     scale as pml_profile(x/dx) * (x/dx).   (The pml_profile_integral_u
     parameter is the integral of this function.) */
  const double kappa_prefac = (mean_stretch - 1) / pml_profile_integral_u;

  // Don't bother with PML if we don't even overlap with the PML region
  // ...note that we should calculate overlap in exactly the same
  // way that "x > 0" is computed below.
  bool found_pml = false;
  for (int i = gv.little_corner().in_direction(d); i <= gv.big_corner().in_direction(d) + 1; ++i)
    if (pml_x(i, dx, bloc, a) > 0) {
      found_pml = true;
      break;
    }
  if (!found_pml) return;
  if (is_mine()) {
    if (sig[d]) {
      delete[] sig[d];
      delete[] kap[d];
      delete[] siginv[d];
      sig[d] = kap[d] = NULL;
      siginv[d] = NULL;
    }
    LOOP_OVER_FIELD_DIRECTIONS(gv.dim, dd) {
      if (!sig[dd]) {
        int spml = (dd == d) ? (2 * gv.num_direction(d) + 2) : 1;
        sigsize[dd] = spml;
        sig[dd] = new realnum[spml];
        kap[dd] = new realnum[spml];
        siginv[dd] = new realnum[spml];
        for (int i = 0; i < spml; ++i) {
          sig[dd][i] = 0.0;
          kap[dd][i] = 1.0;
          siginv[dd][i] = 1.0;
        }
      }
    }

    for (int i = gv.little_corner().in_direction(d); i <= gv.big_corner().in_direction(d) + 1;
         ++i) {
      int idx = i - gv.little_corner().in_direction(d);
      double x = pml_x(i, dx, bloc, a);
      if (x > 0) {
        double s = pml_profile(x / dx, pml_profile_data);
        sig[d][idx] = 0.5 * dt * prefac * s;
        kap[d][idx] = 1 + kappa_prefac * s * (x / dx);
        siginv[d][idx] = 1 / (kap[d][idx] + sig[d][idx]);
      }
    }
  }
  condinv_stale = true;
}

void structure_chunk::update_condinv() {
  if (!condinv_stale || !is_mine()) return;
  FOR_COMPONENTS(c) {
    direction d = component_direction(c);
    if (conductivity[c][d]) {
      if (!condinv[c][d]) condinv[c][d] = new realnum[gv.ntot()];
      PLOOP_OVER_VOL(gv, c, i) { condinv[c][d][i] = 1 / (1 + conductivity[c][d][i] * dt * 0.5); }
    }
    else if (condinv[c][d]) { // condinv not needed
      delete[] condinv[c][d];
      condinv[c][d] = NULL;
    }
  }
  condinv_stale = false;
}

structure_chunk::structure_chunk(const structure_chunk *o) : v(o->v) {
  refcount = 1;

  FOR_FIELD_TYPES(ft) {
    {
      susceptibility *cur = NULL;
      chiP[ft] = NULL;
      for (const susceptibility *ocur = o->chiP[ft]; ocur; ocur = ocur->next) {
        if (cur) {
          cur->next = ocur->clone();
          cur = cur->next;
        }
        else {
          chiP[ft] = cur = ocur->clone();
        }
        cur->next = NULL;
      }
    }
  }
  a = o->a;
  Courant = o->Courant;
  dt = o->dt;
  gv = o->gv;
  the_proc = o->the_proc;
  the_is_mine = my_rank() == n_proc();
  cost = o->cost;
  FOR_COMPONENTS(c) {
    if (is_mine() && o->chi3[c]) {
      chi3[c] = new realnum[gv.ntot()];
      if (chi3[c] == NULL) abort("Out of memory!\n");
      PS1LOOP_OVER_VOL(gv, c, i)
        chi3[c][i] = o->chi3[c][i];
    }
    else {
      chi3[c] = NULL;
    }
    if (is_mine() && o->chi2[c]) {
      chi2[c] = new realnum[gv.ntot()];
      if (chi2[c] == NULL) abort("Out of memory!\n");
      PS1LOOP_OVER_VOL(gv, c, i)
        chi2[c][i] = o->chi2[c][i];
    }
    else {
      chi2[c] = NULL;
    }
  }
  FOR_COMPONENTS(c) FOR_DIRECTIONS(d) { trivial_chi1inv[c][d] = true; }
  FOR_COMPONENTS(c) FOR_DIRECTIONS(d) {
    if (is_mine()) {
      trivial_chi1inv[c][d] = o->trivial_chi1inv[c][d];
      if (o->chi1inv[c][d]) {
        chi1inv[c][d] = new realnum[gv.ntot()];
        memcpy(chi1inv[c][d], o->chi1inv[c][d], gv.ntot() * sizeof(realnum));
      }
      else
        chi1inv[c][d] = NULL;
      if (o->conductivity[c][d]) {
        conductivity[c][d] = new realnum[gv.ntot()];
        memcpy(conductivity[c][d], o->conductivity[c][d], gv.ntot() * sizeof(realnum));
        condinv[c][d] = new realnum[gv.ntot()];
        memcpy(condinv[c][d], o->condinv[c][d], gv.ntot() * sizeof(realnum));
      }
      else
        conductivity[c][d] = condinv[c][d] = NULL;
    }
  }
  condinv_stale = o->condinv_stale;
  // Allocate the PML conductivity arrays:
  for (int d = 0; d < 6; ++d) {
    sig[d] = NULL;
    kap[d] = NULL;
    siginv[d] = NULL;
    sigsize[d] = 0;
  }
  for (int i = 0; i < 5; ++i)
    sigsize[i] = 0;
  // Copy over the PML conductivity arrays:
  if (is_mine()) FOR_DIRECTIONS(d) {
      if (o->sig[d]) {
        sig[d] = new realnum[2 * gv.num_direction(d) + 1];
        kap[d] = new realnum[2 * gv.num_direction(d) + 1];
        siginv[d] = new realnum[2 * gv.num_direction(d) + 1];
        sigsize[d] = o->sigsize[d];
        for (int i = 0; i < 2 * gv.num_direction(d) + 1; i++) {
          sig[d][i] = o->sig[d][i];
          kap[d][i] = o->kap[d][i];
          siginv[d][i] = o->siginv[d][i];
        }
      }
    }
}

void structure_chunk::set_chi3(component c, material_function &epsilon) {
  if (!is_mine() || !gv.has_field(c)) return;
  if (!is_electric(c) && !is_magnetic(c)) abort("only E or H can have chi3");

  epsilon.set_volume(gv.pad().surroundings());

  if (!chi1inv[c][component_direction(c)]) { // require chi1 if we have chi3
    chi1inv[c][component_direction(c)] = new realnum[gv.ntot()];
    PS1LOOP_OVER_VOL(gv, c, i)
      chi1inv[c][component_direction(c)][i] = 1.0;
  }

  if (!chi3[c]) chi3[c] = new realnum[gv.ntot()];
  bool trivial = true;
  PLOOP_OVER_VOL(gv, c, i) {
    IVEC_LOOP_LOC(gv, here);
    chi3[c][i] = epsilon.chi3(c, here);
    trivial = trivial && (chi3[c][i] == 0.0);
  }

  /* currently, our update_e_from_d routine requires that
     chi2 be present if chi3 is, and vice versa */
  if (!chi2[c]) {
    if (!trivial) {
      chi2[c] = new realnum[gv.ntot()];
      memset(chi2[c], 0, gv.ntot() * sizeof(realnum)); // chi2 = 0
    }
    else { // no chi3, and chi2 is trivial (== 0), so delete
      delete[] chi3[c];
      chi3[c] = NULL;
    }
  }

  epsilon.unset_volume();
}

void structure_chunk::set_chi2(component c, material_function &epsilon) {
  if (!is_mine() || !gv.has_field(c)) return;
  if (!is_electric(c) && !is_magnetic(c)) abort("only E or H can have chi2");

  epsilon.set_volume(gv.pad().surroundings());

  if (!chi1inv[c][component_direction(c)]) { // require chi1 if we have chi2
    chi1inv[c][component_direction(c)] = new realnum[gv.ntot()];
    PS1LOOP_OVER_VOL(gv, c, i)
      chi1inv[c][component_direction(c)][i] = 1.0;
  }

  if (!chi2[c]) chi2[c] = new realnum[gv.ntot()];
  bool trivial = true;
  LOOP_OVER_VOL(gv, c, i) {
    IVEC_LOOP_LOC(gv, here);
    chi2[c][i] = epsilon.chi2(c, here);
    trivial = trivial && (chi2[c][i] == 0.0);
  }

  /* currently, our update_e_from_d routine requires that
     chi3 be present if chi2 is, and vice versa */
  if (!chi3[c]) {
    if (!trivial) {
      chi3[c] = new realnum[gv.ntot()];
      memset(chi3[c], 0, gv.ntot() * sizeof(realnum)); // chi3 = 0
    }
    else { // no chi2, and chi3 is trivial (== 0), so delete
      delete[] chi2[c];
      chi2[c] = NULL;
    }
  }

  epsilon.unset_volume();
}

void structure_chunk::set_conductivity(component c, material_function &C) {
  if (!is_mine() || !gv.has_field(c)) return;

  C.set_volume(gv.pad().surroundings());

  if (!is_electric(c) && !is_magnetic(c) && !is_D(c) && !is_B(c))
    abort("invalid component for conductivity");

  direction c_d = component_direction(c);
  component c_C = is_electric(c) ? direction_component(Dx, c_d)
                                 : (is_magnetic(c) ? direction_component(Bx, c_d) : c);
  realnum *multby = is_electric(c) || is_magnetic(c) ? chi1inv[c][c_d] : 0;
  if (!conductivity[c_C][c_d]) conductivity[c_C][c_d] = new realnum[gv.ntot()];
  if (!conductivity[c_C][c_d]) abort("Memory allocation error.\n");
  bool trivial = true;
  realnum *cnd = conductivity[c_C][c_d];
  if (multby) {
    LOOP_OVER_VOL(gv, c_C, i) {
      IVEC_LOOP_LOC(gv, here);
      cnd[i] = C.conductivity(c, here) * multby[i];
      trivial = trivial && (cnd[i] == 0.0);
    }
  }
  else {
    LOOP_OVER_VOL(gv, c_C, i) {
      IVEC_LOOP_LOC(gv, here);
      cnd[i] = C.conductivity(c, here);
      trivial = trivial && (cnd[i] == 0.0);
    }
  }
  if (trivial) { // skip conductivity computations if conductivity == 0
    delete[] conductivity[c_C][c_d];
    conductivity[c_C][c_d] = NULL;
  }
  condinv_stale = true;

  C.unset_volume();
}

structure_chunk::structure_chunk(const grid_volume &thegv, const volume &vol_limit, double Courant,
                                 int pr)
    : Courant(Courant), v(thegv.surroundings() & vol_limit), cost(0.0) {
  refcount = 1;
  pml_fmin = 0.2;
  FOR_FIELD_TYPES(ft) { chiP[ft] = NULL; }
  gv = thegv;
  a = thegv.a;
  dt = Courant / a;
  the_proc = pr;
  the_is_mine = n_proc() == my_rank();

  // initialize materials arrays to NULL
  FOR_COMPONENTS(c) { chi3[c] = NULL; }
  FOR_COMPONENTS(c) { chi2[c] = NULL; }
  FOR_COMPONENTS(c) FOR_DIRECTIONS(d) {
    trivial_chi1inv[c][d] = true;
    chi1inv[c][d] = NULL;
    conductivity[c][d] = NULL;
    condinv[c][d] = NULL;
  }
  condinv_stale = false;
  for (int d = 0; d < 6; ++d) {
    sig[d] = NULL;
    kap[d] = NULL;
    siginv[d] = NULL;
    sigsize[d] = 0;
  }
}

double structure::max_eps() const {
  double themax = 0.0;
  for (int i = 0; i < num_chunks; i++)
    if (chunks[i]->is_mine()) themax = max(themax, chunks[i]->max_eps());
  return max_to_all(themax);
}

double fields::max_eps() const {
  double themax = 0.0;
  for (int i = 0; i < num_chunks; i++)
    if (chunks[i]->is_mine()) themax = max(themax, chunks[i]->s->max_eps());
  return max_to_all(themax);
}

double structure_chunk::max_eps() const {
  double themax = 0.0;
  FOR_COMPONENTS(c) {
    direction d = component_direction(c);
    if (chi1inv[c][d])
      for (size_t i = 0; i < gv.ntot(); i++)
        themax = max(themax, 1 / chi1inv[c][d][i]);
  }
  return themax;
}

bool structure::equal_layout(const structure &s) const {
  if (a != s.a || num_chunks != s.num_chunks || v != s.v || S != s.S) return false;
  for (int i = 0; i < num_chunks; ++i)
    if (chunks[i]->a != s.chunks[i]->a || chunks[i]->v != s.chunks[i]->v) return false;
  return true;
}

void structure_chunk::remove_susceptibilities() {
  FOR_FIELD_TYPES(ft) {
    delete chiP[ft];
    chiP[ft] = NULL;
  }
}

void structure::remove_susceptibilities() {
  changing_chunks();
  for (int i = 0; i < num_chunks; i++)
    chunks[i]->remove_susceptibilities();
}

// for debugging, display the chunk layout
void structure::print_layout(void) const {
  direction d0 = gv.yucky_direction(0);
  direction d1 = gv.yucky_direction(1);
  direction d2 = gv.yucky_direction(2);
  for (int i = 0; i < num_chunks; ++i) {
    master_printf("chunk[%d] on process %d, resolution %g (%s,%s,%s):"
                  " (%d,%d,%d) - (%d,%d,%d)\n",
                  i, chunks[i]->n_proc(), chunks[i]->a, direction_name(d0), direction_name(d1),
                  direction_name(d2), chunks[i]->gv.little_corner().yucky_val(0),
                  chunks[i]->gv.little_corner().yucky_val(1),
                  chunks[i]->gv.little_corner().yucky_val(2),
                  chunks[i]->gv.big_corner().yucky_val(0), chunks[i]->gv.big_corner().yucky_val(1),
                  chunks[i]->gv.big_corner().yucky_val(2));
  }
}

std::vector<grid_volume> structure::get_chunk_volumes() const {
  std::vector<grid_volume> result;

  for (int i = 0; i < num_chunks; ++i) {
    result.push_back(chunks[i]->gv);
  }
  return result;
}

std::vector<int> structure::get_chunk_owners() const {
  std::vector<int> result;

  for (int i = 0; i < num_chunks; ++i) {
    result.push_back(chunks[i]->n_proc());
  }
  return result;
}
} // namespace meep
