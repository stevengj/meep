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
#ifndef MATERIAL_DATA_H
#define MATERIAL_DATA_H

/***************************************************************/
/* the void *data field in geometric_object_struct points to   */
/* a material_data structure, defined below.                   */
/***************************************************************/
#include <vector>

#include <meep.hpp>
#include <ctlgeom.h>

#include "meep.hpp"

namespace meep_geom {

/* FIXME: we don't have to emulate the Scheme/libctl code here, which was
  limited to C functionality.  These types, especially the material types,
  should be proper C++ classes */

struct transition {
  int from_level;
  int to_level;
  double transition_rate;
  double frequency;
  vector3 sigma_diag;
  double gamma;
  double pumping_rate;

  bool operator==(const transition &other) const;
  bool operator!=(const transition &other) const;
};

typedef struct susceptibility_struct {
  vector3 sigma_offdiag;
  vector3 sigma_diag;
  vector3 bias;
  double frequency;
  double gamma;
  double alpha;
  double noise_amp;
  bool drude;
  bool saturated_gyrotropy;
  bool is_file;
  std::vector<transition> transitions;
  std::vector<double> initial_populations;
} susceptibility;

using susceptibility_list = std::vector<susceptibility>;

struct medium_struct {
  vector3 epsilon_diag;
  cvector3 epsilon_offdiag;
  vector3 mu_diag;
  cvector3 mu_offdiag;
  susceptibility_list E_susceptibilities;
  susceptibility_list H_susceptibilities;
  vector3 E_chi2_diag;
  vector3 E_chi3_diag;
  vector3 H_chi2_diag;
  vector3 H_chi3_diag;
  vector3 D_conductivity_diag;
  vector3 B_conductivity_diag;

  explicit medium_struct(double epsilon = 1);

  // Aborts Meep if a non-zero imaginary part of an offdiagonal mu or epsilon entry is found.
  void check_offdiag_im_zero_or_abort() const;
};

// prototype for user-defined material function,
// which should fill in medium as appropriate to
// describe the material properties at point x
typedef void (*user_material_func)(vector3 x, void *user_data, medium_struct *medium);

// the various types of materials are as follows:
//  MEDIUM:        material properties independent of position. In
//                 this case the 'medium' field below is
//                 initialized once and doesn't change.
//  MATERIAL_FILE: material properties position-dependent, described
//                 by user-supplied data file. In this case the
//                 'medium' field is filled in appropriately at
//                 each evaluation point by interpolating file data.
//  MATERIAL_USER: material properties position-dependent, described
//                 by user-supplied function. In this case the
//                 'medium' field is filled in appropriately at
//                 each evaluation point by calling the user's
//                 routine.
//  MATERIAL_GRID: material properties position-dependent, described
//                 by user-supplied array of grid points. In this case
//                 the 'medium' field is filled in appropriately at
//                 each evaluation point by interpolating the array.
//  PERFECT_METAL: the 'medium' field is never referenced in this case.
struct material_data {
  enum {
    MEDIUM,
    MATERIAL_FILE, // formerly MATERIAL_TYPE_SELF
    MATERIAL_USER, // formerly MATERIAL_FUNCTION
    MATERIAL_GRID,
    PERFECT_METAL
  } which_subclass;

  // this field is used for all material types except PERFECT_METAL and MATERIAL_GRID
  medium_struct medium;

  // these fields used only if which_subclass==MATERIAL_USER
  user_material_func user_func;
  void *user_data;

  // used for any variable material (USER, FILE, or GRID):
  // indicates whether we should use the fallback subpixel averaging via quadrature
  bool do_averaging;

  // these fields used only if which_subclass==MATERIAL_FILE
  double *epsilon_data;
  size_t epsilon_dims[3];

  // these fields used only if which_subclass==MATERIAL_GRID
  vector3 grid_size;
  double *weights;
  medium_struct medium_1;
  medium_struct medium_2;
  double beta;
  double eta;
  /*
  There are several possible scenarios when material grids overlap -- these
  different scenarios enable different applications.

  For U_MIN: Where multiple grids overlap, only those grids that contribute
  the minimum u contribute, and for other grids the gradient is zero.
  This unfortunately makes the gradient only piecewise continuous.

  For U_PROD: The gradient is multiplied by the product of u's from
  overlapping grids, divided by the u from the current grid.  This
  unfortunately makes the gradient zero when two or more u's are zero,
  stalling convergence, although we try to avoid this by making the
  minimum u = 1e-4 instead of 0.

  For U_MEAN: The gradient is divided by the number of overlapping grids.
  This doesn't have the property that u=0 in one grid makes the total
  u=0, unfortunately, which is desirable if u=0 indicates "drilled holes".

  For U_DEFAULT: Expect the default behavior with libctl objects; that is
  the object on top always wins and everything underneath is ignored.
  Specifically, that means that u = the top material grid value at that point.
  */
  enum { U_MIN = 0, U_PROD = 1, U_MEAN = 2, U_DEFAULT = 3 } material_grid_kinds;

  material_data();

  void copy_from(const material_data& from);
};

typedef material_data *material_type;

struct material_type_list {
  material_type *items;
  int num_items;

  material_type_list();
};

// global variables
extern material_type vacuum;

// exported functions for creating particular material types
material_type make_dielectric(double epsilon);
material_type make_user_material(user_material_func user_func, void *user_data);
material_type make_file_material(char *epsilon_input_file);
material_type make_material_grid(bool do_averaging, double beta, double eta);
void read_epsilon_file(const char *eps_input_file);
void update_weights(material_type matgrid, double *weights);

}; // namespace meep_geom

#endif // #ifndef MATERIAL_DATA_H
