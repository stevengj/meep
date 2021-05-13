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
#include <math.h>
#include <string.h>

#include "meep.hpp"
#include "meep_internals.hpp"
#include "config.h"

using namespace std;

namespace meep {

void fields::update_pols(field_type ft) {
  for (int i = 0; i < num_chunks; i++)
    if (chunks[i]->is_mine())
      if (chunks[i]->update_pols(ft)) chunk_connections_valid = false;
}

template <class T>
bool fields_chunk::update_pols(field_type ft) {
  bool allocated_fields = false;

  T *w[NUM_FIELD_COMPONENTS][2];
  FOR_COMPONENTS(c) DOCMP2 { w[c][cmp] = f_w[c][cmp] ? f_w[c][cmp] : f[c][cmp]; }

  for (polarization_state *p = pol[ft]; p; p = p->next) {

    // Lazily allocate internal polarization data:
    if (!p->data) {
      p->data = p->s->new_internal_data(f, gv);
      if (p->data) {
        p->s->init_internal_data(f, dt, gv, p->data);
        allocated_fields = true;
      }
    }

    // Finally, timestep the polarizations:
    p->s->update_P(w, f_w_prev, dt, gv, p->data);
  }

  return allocated_fields;
}

} // namespace meep
