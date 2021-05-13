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

#include <stdlib.h>
#include <complex>

#include "meep.hpp"
#include "meep_internals.hpp"

#define UNUSED(x) (void)x // silence compiler warnings

using namespace std;

namespace meep {

void fields::set_boundary(boundary_side b, direction d, boundary_condition cond) {
  if (boundaries[b][d] != cond) {
    boundaries[b][d] = cond;
    chunk_connections_valid = false;
  }
}

void fields::use_bloch(direction d, complex<double> kk) {
  k[d] = kk;
  for (int b = 0; b < 2; b++)
    set_boundary(boundary_side(b), d, Periodic);
  if (real(kk) * gv.num_direction(d) == 0.5 * a) // check b.z. edge exactly
    eikna[d] = -exp(-imag(kk) * ((2 * pi / a) * gv.num_direction(d)));
  else {
    const complex<double> I = complex<double>(0.0, 1.0);
    eikna[d] = exp(I * kk * ((2 * pi / a) * gv.num_direction(d)));
  }
  coskna[d] = real(eikna[d]);
  sinkna[d] = imag(eikna[d]);
  if (is_real && kk != 0.0) // FIXME: allow real phases (c.f. CONNECT_PHASE)
    abort("Can't use real fields with bloch boundary conditions!\n");
  chunk_connections_valid = false; // FIXME: we don't always need to invalidate
}

void fields::use_bloch(const vec &k) {
  // Note that I allow a 1D k input when in cylindrical, since in that case
  // it is unambiguous.
  if (k.dim != gv.dim && !(k.dim == D1 && gv.dim == Dcyl))
    abort("Aaaack, k has wrong dimensions!\n");
  LOOP_OVER_DIRECTIONS(gv.dim, d) {
    if (gv.has_boundary(Low, d) && d != R) use_bloch(d, k.in_direction(d));
  }
}

ivec fields::ilattice_vector(direction d) const {
  switch (user_volume.dim) {
    case D1: return ivec(2 * user_volume.nz());
    case Dcyl: return iveccyl(0, 2 * user_volume.nz()); // Only Z direction here
    case D2:
      switch (d) {
        case X: return ivec(user_volume.nx() * 2, 0);
        case Y: return ivec(0, user_volume.ny() * 2);
        case Z:
        case R:
        case P:
        case NO_DIRECTION: break;
      }
    case D3:
      switch (d) {
        case X: return ivec(user_volume.nx() * 2, 0, 0);
        case Y: return ivec(0, user_volume.ny() * 2, 0);
        case Z: return ivec(0, 0, user_volume.nz() * 2);
        case R:
        case P:
        case NO_DIRECTION: break;
      }
  }
  abort("Aaack in ilattice_vector.\n");
  return ivec(0);
}

vec fields::lattice_vector(direction d) const { return gv[ilattice_vector(d)]; }

void fields::disconnect_chunks() {
  chunk_connections_valid = false;
  for (int i = 0; i < num_chunks; i++) {
    DOCMP {
      FOR_FIELD_TYPES(f) {
        for (int ip = 0; ip < 3; ++ip)
          for (int io = 0; io < 2; io++) {
            delete[] chunks[i]->connections[f][ip][io];
            chunks[i]->connections[f][ip][io] = NULL;
          }
      }
    }
    FOR_FIELD_TYPES(f) {
      delete[] chunks[i]->connection_phases[f];
      chunks[i]->connection_phases[f] = NULL;
      for (int ip = 0; ip < 3; ++ip)
        for (int io = 0; io < 2; io++)
          chunks[i]->num_connections[f][ip][io] = 0;
    }
  }
  FOR_FIELD_TYPES(ft) {
    for (int i = 0; i < num_chunks * num_chunks; i++) {
      delete[] comm_blocks[ft][i];
      comm_blocks[ft][i] = 0;
      for (int ip = 0; ip < 3; ++ip)
        comm_sizes[ft][ip][i] = 0;
    }
  }
}

void fields::connect_chunks() {
  /* make sure all processes agree on chunk_connections_valid to avoid deadlocks */
  am_now_working_on(MpiAllTime);
  chunk_connections_valid = and_to_all(chunk_connections_valid);
  finished_working();

  if (!chunk_connections_valid) {
    am_now_working_on(Connecting);
    disconnect_chunks();
    find_metals();
    connect_the_chunks();
    finished_working();
    chunk_connections_valid = true;
  }
}

inline bool fields::on_metal_boundary(const ivec &here) {
  LOOP_OVER_DIRECTIONS(gv.dim, d) {
    if (user_volume.has_boundary(High, d) &&
        here.in_direction(d) == user_volume.big_corner().in_direction(d)) {
      if (boundaries[High][d] == Metallic) return true;
    }
    if (boundaries[Low][d] == Magnetic &&
        here.in_direction(d) == user_volume.little_corner().in_direction(d) + 1)
      return true;
    if (boundaries[Low][d] == Metallic &&
        here.in_direction(d) == user_volume.little_corner().in_direction(d))
      return true;
  }
  return false;
}

bool fields::locate_point_in_user_volume(ivec *there, complex<double> *phase) const {
  // Check if a translational symmetry is needed to bring the point in...
  if (!user_volume.owns(*there)) {
    LOOP_OVER_DIRECTIONS(gv.dim, d) {
      if (boundaries[High][d] == Periodic &&
          there->in_direction(d) <= user_volume.little_corner().in_direction(d)) {
        while (there->in_direction(d) <= user_volume.little_corner().in_direction(d)) {
          *there += ilattice_vector(d);
          *phase *= conj(eikna[d]);
        }
      }
      else if (boundaries[High][d] == Periodic &&
               there->in_direction(d) - ilattice_vector(d).in_direction(d) >
                   user_volume.little_corner().in_direction(d)) {
        while (there->in_direction(d) - ilattice_vector(d).in_direction(d) >
               user_volume.little_corner().in_direction(d)) {
          *there -= ilattice_vector(d);
          *phase *= eikna[d];
        }
      }
    }
  }
  return user_volume.owns(*there);
}

void fields::locate_volume_source_in_user_volume(const vec p1, const vec p2, vec newp1[8],
                                                 vec newp2[8], complex<double> kphase[8],
                                                 int &ncopies) const {
  // For periodic boundary conditions,
  // this function locates up to 8 translated copies of the initial grid_volume specified by (p1,p2)
  // First bring center of grid_volume inside
  ncopies = 1;
  newp1[0] = p1;
  newp2[0] = p2;
  kphase[0] = 1;
  vec cen = (newp1[0] + newp2[0]) * 0.5;
  LOOP_OVER_DIRECTIONS(gv.dim, d) {
    if (boundaries[High][d] == Periodic) {
      while (cen.in_direction(d) < gv.boundary_location(Low, d)) {
        newp1[0] += lattice_vector(d);
        newp2[0] += lattice_vector(d);
        kphase[0] *= conj(eikna[d]);
        cen = (newp1[0] + newp2[0]) * 0.5;
      }
      while (cen.in_direction(d) > gv.boundary_location(High, d)) {
        newp1[0] -= lattice_vector(d);
        newp2[0] -= lattice_vector(d);
        kphase[0] *= eikna[d];
        cen = (newp1[0] + newp2[0]) * 0.5;
      }
    }
  }

  // if grid_volume extends outside user_volume in any direction, we need to duplicate already
  // existing copies
  LOOP_OVER_DIRECTIONS(gv.dim, d) {
    if (boundaries[High][d] == Periodic) {
      if (newp1[0].in_direction(d) < gv.boundary_location(Low, d) ||
          newp2[0].in_direction(d) < gv.boundary_location(Low, d)) {
        for (int j = 0; j < ncopies; j++) {
          newp1[ncopies + j] = newp1[j] + lattice_vector(d);
          newp2[ncopies + j] = newp2[j] + lattice_vector(d);
          kphase[ncopies + j] = kphase[j] * conj(eikna[d]);
        }
        ncopies *= 2;
      }
      else if (newp1[0].in_direction(d) > gv.boundary_location(High, d) ||
               newp2[0].in_direction(d) > gv.boundary_location(High, d)) {
        for (int j = 0; j < ncopies; j++) {
          newp1[ncopies + j] = newp1[j] - lattice_vector(d);
          newp2[ncopies + j] = newp2[j] - lattice_vector(d);
          kphase[ncopies + j] = kphase[j] * eikna[d];
        }
        ncopies *= 2;
      }
    }
  }
}

bool fields::locate_component_point(component *c, ivec *there, complex<double> *phase) const {
  // returns true if this point and component exist in the user_volume.  If
  // that is the case, on return *c and *there store the component and
  // location of where the point actually is, and *phase determines holds
  // the phase needed to get the true field.  If the point is not located,
  // *c and *there will hold undefined values.

  // Check if nothing tricky is needed...
  *phase = 1.0;
  if (!locate_point_in_user_volume(there, phase)) return false;
  // Check if a rotation or inversion brings the point in...
  if (user_volume.owns(*there))
    for (int sn = 0; sn < S.multiplicity(); sn++) {
      const ivec here = S.transform(*there, sn);
      if (gv.owns(here)) {
        *there = here;
        *phase *= S.phase_shift(*c, sn);
        *c = direction_component(*c, S.transform(component_direction(*c), sn).d);
        return true;
      }
    }
  return false;
}

void fields_chunk::zero_metal(field_type ft) {
  for (size_t i = 0; i < num_zeroes[ft]; i++)
    *(zeroes[ft][i]) = 0.0;
}

template <class T>
void fields::find_metals() {
  for (int i = 0; i < num_chunks; i++)
    if (chunks[i]->is_mine()) {
      const grid_volume vi = chunks[i]->gv;
      FOR_FIELD_TYPES(ft) {
        delete[] chunks[i]->zeroes[ft];
        // First electric components...
        chunks[i]->num_zeroes[ft] = 0;
        DOCMP FOR_COMPONENTS(c) {
          if (type(c) == ft && chunks[i]->f[c][cmp]) LOOP_OVER_VOL_OWNED(vi, c, n) {
              if (IVEC_LOOP_AT_BOUNDARY) { // todo: just loop over boundaries
                IVEC_LOOP_ILOC(vi, here);
                if (on_metal_boundary(here)) chunks[i]->num_zeroes[ft]++;
              }
            }
        }
        typedef T *T_ptr;
        chunks[i]->zeroes[ft] = new T_ptr[chunks[i]->num_zeroes[ft]];
        size_t num = 0;
        DOCMP FOR_COMPONENTS(c) {
          if (type(c) == ft && chunks[i]->f[c][cmp]) LOOP_OVER_VOL_OWNED(vi, c, n) {
              if (IVEC_LOOP_AT_BOUNDARY) { // todo: just loop over boundaries
                IVEC_LOOP_ILOC(vi, here);
                if (on_metal_boundary(here))
                  chunks[i]->zeroes[ft][num++] = chunks[i]->f[c][cmp] + n;
              }
            }
        }
      }
    }
}

bool fields_chunk::needs_W_notowned(component c) {
  for (susceptibility *chiP = s->chiP[type(c)]; chiP; chiP = chiP->next)
    if (chiP->needs_W_notowned(c, f)) return true;
  return false;
}

template <class T>
void fields::connect_the_chunks() {
  size_t *nc[NUM_FIELD_TYPES][3][2];
  FOR_FIELD_TYPES(f) {
    for (int ip = 0; ip < 3; ip++)
      for (int io = 0; io < 2; io++) {
        nc[f][ip][io] = new size_t[num_chunks];
        for (int i = 0; i < num_chunks; i++)
          nc[f][ip][io][i] = 0;
      }
  }

  /* For some of the chunks, H==B, and we definitely don't need to
     send B between two such chunks.   We'll still send B when
     the recipient has H != B, since the recipient needs to get B
     from somewhere (although it could get it locally, in principle).
     When the sender has H != B, we'll skip sending B (we'll only send H)
     since we need to get the correct curl H in the E update.  This is
     a bit subtle since the non-owned B may be different from H even
     on an H==B chunk (true?), but since we don't use the non-owned B
     for anything(?) it shouldn't matter. */
  int *B_redundant = new int[num_chunks * 2 * 5];
  for (int i = 0; i < num_chunks; ++i)
    FOR_H_AND_B(hc, bc) {
      B_redundant[5 * (num_chunks + i) + bc - Bx] = chunks[i]->f[hc][0] == chunks[i]->f[bc][0];
    }
  am_now_working_on(MpiAllTime);
  and_to_all(B_redundant + 5 * num_chunks, B_redundant, 5 * num_chunks);
  finished_working();

  /* Figure out whether we need the notowned W field (== E/H in
     non-PML regions) in update_pols, e.g. if we have an anisotropic
     susceptibility.  In this case, we have an additional
     communication step where we communicate the notowned W.  Then,
     after updating the polarizations, we communicate the notowned E/H
     ... this does the E/H communication twice between non-PML regions
     and hence is somewhat wasteful, but greatly simplifies the case
     of communicating between a PML region (which has a separate W
     array) and a non-PML region (no separate W). */
  bool needs_W_notowned[NUM_FIELD_COMPONENTS];
  FOR_COMPONENTS(c) { needs_W_notowned[c] = false; }
  FOR_E_AND_H(c) {
    for (int i = 0; i < num_chunks; i++)
      needs_W_notowned[c] = needs_W_notowned[c] || chunks[i]->needs_W_notowned(c);
  }
  am_now_working_on(MpiAllTime);
  FOR_E_AND_H(c) { needs_W_notowned[c] = or_to_all(needs_W_notowned[c]); }
  finished_working();

  for (int i = 0; i < num_chunks; i++) {
    // First count the border elements...
    const grid_volume vi = chunks[i]->gv;
    FOR_FIELD_TYPES(ft) {
      for (int ip = 0; ip < 3; ip++)
        for (int j = 0; j < num_chunks; j++)
          comm_sizes[ft][ip][j + i * num_chunks] = 0;
    }
    FOR_COMPONENTS(corig) {
      if (have_component(corig)) LOOP_OVER_VOL_NOTOWNED(vi, corig, n) {
          IVEC_LOOP_ILOC(vi, here);
          component c = corig;
          // We're looking at a border element...
          complex<double> thephase;
          if (locate_component_point(&c, &here, &thephase) && !on_metal_boundary(here))
            for (int j = 0; j < num_chunks; j++) {
              if ((chunks[i]->is_mine() || chunks[j]->is_mine()) && chunks[j]->gv.owns(here) &&
                  !(is_B(corig) && is_B(c) && B_redundant[5 * i + corig - Bx] &&
                    B_redundant[5 * j + c - Bx])) {
                const int pair = j + i * num_chunks;
                const connect_phase ip = thephase == 1.0
                                             ? CONNECT_COPY
                                             : (thephase == -1.0 ? CONNECT_NEGATE : CONNECT_PHASE);
                {
                  field_type f = type(c);
                  const int nn = is_real ? 1 : 2;
                  nc[f][ip][Incoming][i] += nn;
                  nc[f][ip][Outgoing][j] += nn;
                  comm_sizes[f][ip][pair] += nn;
                }
                if (needs_W_notowned[corig]) {
                  field_type f = is_electric(corig) ? WE_stuff : WH_stuff;
                  const int nn = is_real ? 1 : 2;
                  nc[f][ip][Incoming][i] += nn;
                  nc[f][ip][Outgoing][j] += nn;
                  comm_sizes[f][ip][pair] += nn;
                }
                if (is_electric(corig) || is_magnetic(corig)) {
                  field_type f = is_electric(corig) ? PE_stuff : PH_stuff;
                  size_t ni = 0, cni = 0;
                  for (polarization_state *pi = chunks[i]->pol[type(corig)]; pi; pi = pi->next)
                    for (polarization_state *pj = chunks[j]->pol[type(c)]; pj; pj = pj->next)
                      if (*pi->s == *pj->s) {
                        if (pi->data && chunks[i]->is_mine()) {
                          ni += pi->s->num_internal_notowned_needed(corig, pi->data);
                          cni += pi->s->num_cinternal_notowned_needed(corig, pi->data);
                        }
                        else if (pj->data && chunks[j]->is_mine()) {
                          ni += pj->s->num_internal_notowned_needed(c, pj->data);
                          cni += pj->s->num_cinternal_notowned_needed(c, pj->data);
                        }
                      }
                  const size_t nn = (is_real ? 1 : 2) * (cni);
                  nc[f][ip][Incoming][i] += nn;
                  nc[f][ip][Outgoing][j] += nn;
                  comm_sizes[f][ip][pair] += nn;
                  const connect_phase iip = CONNECT_COPY;
                  nc[f][iip][Incoming][i] += ni;
                  nc[f][iip][Outgoing][j] += ni;
                  comm_sizes[f][iip][pair] += ni;
                }
              } // if is_mine and owns...
            }   // loop over j chunks
        }       // LOOP_OVER_VOL_NOTOWNED
    }           // FOR_COMPONENTS

    // Allocating comm blocks as we go...
    FOR_FIELD_TYPES(ft) {
      for (int j = 0; j < num_chunks; j++) {
        delete[] comm_blocks[ft][j + i * num_chunks];
        comm_blocks[ft][j + i * num_chunks] = new T[comm_size_tot(ft, j + i * num_chunks)];
      }
    }
  } // loop over i chunks

  /* Note that the ordering of the connections arrays must be kept
     consistent with the fields::step_boundaries.  In particular, we
     must set up the connections array so that all of the connections
     for process i come before all of the connections for process i'
     for i < i' */

  // wh stores the current indices in the connections array(s)
  size_t *wh[NUM_FIELD_TYPES][3][2];

  /* Now allocate the connection arrays... this is still slightly
     wasteful (by a factor of 2) because we allocate for chunks we
     don't own if we have a connection with them. Removing this waste
     would mean a bunch more is_mine() checks below. */
  FOR_FIELD_TYPES(f) {
    for (int ip = 0; ip < 3; ip++) {
      for (int io = 0; io < 2; io++) {
        for (int i = 0; i < num_chunks; i++)
          chunks[i]->alloc_extra_connections(field_type(f), connect_phase(ip), in_or_out(io),
                                             nc[f][ip][io][i]);
        delete[] nc[f][ip][io];
        wh[f][ip][io] = new size_t[num_chunks];
      }
      for (int i = 0; i < num_chunks; i++)
        wh[f][ip][Outgoing][i] = 0;
    }
  }

  // Next start setting up the connections...

  for (int i = 0; i < num_chunks; i++) {
    const grid_volume vi = chunks[i]->gv;

    // initialize wh[f][ip][Incoming][j] to sum of comm_sizes for jj < j
    FOR_FIELD_TYPES(f) {
      for (int ip = 0; ip < 3; ip++) {
        wh[f][ip][Incoming][0] = 0;
        for (int j = 1; j < num_chunks; ++j)
          wh[f][ip][Incoming][j] =
              wh[f][ip][Incoming][j - 1] + comm_sizes[f][ip][(j - 1) + i * num_chunks];
      }
    }

    FOR_COMPONENTS(corig) {
      if (have_component(corig)) LOOP_OVER_VOL_NOTOWNED(vi, corig, n) {
          IVEC_LOOP_ILOC(vi, here);
          component c = corig;
          // We're looking at a border element...
          complex<double> thephase;
          if (locate_component_point(&c, &here, &thephase) && !on_metal_boundary(here))
            for (int j = 0; j < num_chunks; j++) {
              if ((chunks[i]->is_mine() || chunks[j]->is_mine()) && chunks[j]->gv.owns(here) &&
                  !(is_B(corig) && is_B(c) && B_redundant[5 * i + corig - Bx] &&
                    B_redundant[5 * j + c - Bx])) {
                const connect_phase ip = thephase == 1.0
                                             ? CONNECT_COPY
                                             : (thephase == -1.0 ? CONNECT_NEGATE : CONNECT_PHASE);
                const ptrdiff_t m = chunks[j]->gv.index(c, here);

                {
                  field_type f = type(c);
                  if (ip == CONNECT_PHASE)
                    chunks[i]->connection_phases[f][wh[f][ip][Incoming][j] / 2] = thephase;
                  DOCMP {
                    chunks[i]->connections[f][ip][Incoming][wh[f][ip][Incoming][j]++] =
                        chunks[i]->f[corig][cmp] + n;
                    chunks[j]->connections[f][ip][Outgoing][wh[f][ip][Outgoing][j]++] =
                        chunks[j]->f[c][cmp] + m;
                  }
                }

                if (needs_W_notowned[corig]) {
                  field_type f = is_electric(corig) ? WE_stuff : WH_stuff;
                  if (ip == CONNECT_PHASE)
                    chunks[i]->connection_phases[f][wh[f][ip][Incoming][j] / 2] = thephase;
                  DOCMP {
                    chunks[i]->connections[f][ip][Incoming][wh[f][ip][Incoming][j]++] =
                        (chunks[i]->f_w[corig][cmp] ? chunks[i]->f_w[corig][cmp]
                                                    : chunks[i]->f[corig][cmp]) +
                        n;
                    chunks[j]->connections[f][ip][Outgoing][wh[f][ip][Outgoing][j]++] =
                        (chunks[j]->f_w[c][cmp] ? chunks[j]->f_w[c][cmp] : chunks[j]->f[c][cmp]) +
                        m;
                  }
                }

                if (is_electric(corig) || is_magnetic(corig)) {
                  field_type f = is_electric(corig) ? PE_stuff : PH_stuff;
                  for (polarization_state *pi = chunks[i]->pol[type(corig)]; pi; pi = pi->next)
                    for (polarization_state *pj = chunks[j]->pol[type(c)]; pj; pj = pj->next)
                      if (*pi->s == *pj->s) {
                        polarization_state *po = NULL;
                        if (pi->data && chunks[i]->is_mine())
                          po = pi;
                        else if (pj->data && chunks[j]->is_mine())
                          po = pj;
                        if (po) {
                          const connect_phase iip = CONNECT_COPY;
                          const size_t ni = po->s->num_internal_notowned_needed(corig, po->data);
                          for (size_t k = 0; k < ni; ++k) {
                            chunks[i]->connections[f][iip][Incoming][wh[f][iip][Incoming][j]++] =
                                po->s->internal_notowned_ptr(k, corig, n, pi->data);
                            chunks[j]->connections[f][iip][Outgoing][wh[f][iip][Outgoing][j]++] =
                                po->s->internal_notowned_ptr(k, c, m, pj->data);
                          }
                          const size_t cni = po->s->num_cinternal_notowned_needed(corig, po->data);
                          for (size_t k = 0; k < cni; ++k) {
                            if (ip == CONNECT_PHASE)
                              chunks[i]->connection_phases[f][wh[f][ip][Incoming][j] / 2] =
                                  thephase;
                            DOCMP {
                              chunks[i]->connections[f][ip][Incoming][wh[f][ip][Incoming][j]++] =
                                  po->s->cinternal_notowned_ptr(k, corig, cmp, n, pi->data);
                              chunks[j]->connections[f][ip][Outgoing][wh[f][ip][Outgoing][j]++] =
                                  po->s->cinternal_notowned_ptr(k, c, cmp, m, pj->data);
                            }
                          }
                        }
                      }
                } // is_electric(corig)
              }   // if is_mine and owns...
            }     // loop over j chunks
        }         // LOOP_OVER_VOL_NOTOWNED
    }             // FOR_COMPONENTS
  }               // loop over i chunks
  FOR_FIELD_TYPES(f) {
    for (int ip = 0; ip < 3; ip++)
      for (int io = 0; io < 2; io++)
        delete[] wh[f][ip][io];
  }
  delete[] B_redundant;
}

template <class T>
void fields_chunk::alloc_extra_connections(field_type f, connect_phase ip, in_or_out io,
                                           size_t num) {
  if (num == 0) return; // No need to go to any bother...
  const size_t tot = num_connections[f][ip][io] + num;
  if (io == Incoming && ip == CONNECT_PHASE) {
    delete[] connection_phases[f];
    connection_phases[f] = new complex<T>[tot];
  }
  typedef T *T_ptr;
  T **conn = new T_ptr[tot];
  if (!conn) abort("Out of memory!\n");
  delete[] connections[f][ip][io];
  connections[f][ip][io] = conn;
  num_connections[f][ip][io] = tot;
}

} // namespace meep
