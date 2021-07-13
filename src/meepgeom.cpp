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
%  GNU General Public License for more details.  %
%  You should have received a copy of the GNU General Public License
%  along with this program; if not, write to the Free Software Foundation,
%  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
*/

#include <algorithm>
#include <vector>
#include "meepgeom.hpp"
#include "meep_internals.hpp"

namespace meep_geom {

#define master_printf meep::master_printf

/***************************************************************/
/* global variables for default material                       */
/***************************************************************/
material_data vacuum_material_data;
material_type vacuum = &vacuum_material_data;

static void set_default_material(material_type _default_material) {
  if (default_material != NULL) {
    if (default_material == _default_material) return;
    material_free((material_type)default_material);
    delete (material_type)default_material;
    default_material = NULL;
  }

  if (_default_material != NULL) {
    material_type new_material = new material_data();
    new_material->copy_from(*_default_material);
    default_material = (void *)new_material;
  }
}

bool susceptibility_equal(const susceptibility &s1, const susceptibility &s2) {
  return (vector3_equal(s1.sigma_diag, s2.sigma_diag) &&
          vector3_equal(s1.sigma_offdiag, s2.sigma_offdiag) && vector3_equal(s1.bias, s2.bias) &&
          s1.frequency == s2.frequency && s1.gamma == s2.gamma && s1.alpha == s2.alpha &&
          s1.noise_amp == s2.noise_amp && s1.drude == s2.drude &&
          s1.saturated_gyrotropy == s2.saturated_gyrotropy && s1.is_file == s2.is_file);
}

bool susceptibility_list_equal(const susceptibility_list &s1, const susceptibility_list &s2) {
  if (s1.size() != s2.size()) return false;
  for (size_t i = 0; i < s1.size(); ++i) {
    if (!susceptibility_equal(s1[i], s2[i])) return false;
  }
  return true;
}

bool medium_struct_equal(const medium_struct *m1, const medium_struct *m2) {
  return (vector3_equal(m1->epsilon_diag, m2->epsilon_diag) &&
          cvector3_equal(m1->epsilon_offdiag, m2->epsilon_offdiag) &&
          vector3_equal(m1->mu_diag, m2->mu_diag) &&
          cvector3_equal(m1->mu_offdiag, m2->mu_offdiag) &&
          vector3_equal(m1->E_chi2_diag, m2->E_chi2_diag) &&
          vector3_equal(m1->E_chi3_diag, m2->E_chi3_diag) &&
          vector3_equal(m1->H_chi2_diag, m2->H_chi2_diag) &&
          vector3_equal(m1->D_conductivity_diag, m2->D_conductivity_diag) &&
          vector3_equal(m1->B_conductivity_diag, m2->B_conductivity_diag) &&
          susceptibility_list_equal(m1->E_susceptibilities, m2->E_susceptibilities) &&
          susceptibility_list_equal(m1->H_susceptibilities, m2->H_susceptibilities));
}

bool material_grid_equal(const material_data *m1, const material_data *m2) {
  // a rigorous method for comapring material grids
  int n1, n2;
  n1 = m1->grid_size.x * m1->grid_size.y * m1->grid_size.z;
  n2 = m2->grid_size.x * m2->grid_size.y * m2->grid_size.z;
  if (n1 != n2) return false;
  for (int i = 0; i < n1; i++)
    if (m1->epsilon_data[i] != m2->epsilon_data[i]) return false;

  return (medium_struct_equal(&(m1->medium), &(m2->medium)) &&
          medium_struct_equal(&(m1->medium_1), &(m2->medium_1)) &&
          medium_struct_equal(&(m1->medium_2), &(m2->medium_2)));
}

// garbage collection for material structures: called to deallocate memory
// allocated for susceptibilities in user-defined materials.
// TODO
void material_gc(material_type m) {
  if (!m || m->which_subclass != material_data::MATERIAL_USER) return;
  m->medium.E_susceptibilities.clear();
  m->medium.H_susceptibilities.clear();
  m->medium_1.E_susceptibilities.clear();
  m->medium_1.H_susceptibilities.clear();
  m->medium_2.E_susceptibilities.clear();
  m->medium_2.H_susceptibilities.clear();
}

void material_free(material_type m) {
  if (!m) return;

  m->medium.E_susceptibilities.clear();
  m->medium.H_susceptibilities.clear();
  m->medium_1.E_susceptibilities.clear();
  m->medium_1.H_susceptibilities.clear();
  m->medium_2.E_susceptibilities.clear();
  m->medium_2.H_susceptibilities.clear();

  // NOTE: We do not delete the user_data field here since it is an opaque/void
  // object so will assume that the caller keeps track of its lifetime.
  delete[] m->epsilon_data;
  m->epsilon_data = NULL;

  delete[] m->weights;
  m->weights = NULL;
}

bool material_type_equal(const material_type m1, const material_type m2) {
  if (m1 == m2) return true;
  if (m1->which_subclass != m2->which_subclass) return false;
  switch (m1->which_subclass) {
    case material_data::MATERIAL_FILE:
    case material_data::PERFECT_METAL: return true;
    case material_data::MATERIAL_USER:
      return m1->user_func == m2->user_func && m1->user_data == m2->user_data;
    case material_data::MATERIAL_GRID:
    case material_data::MEDIUM: return medium_struct_equal(&(m1->medium), &(m2->medium));
    default: return false;
  }
}

/***************************************************************/
/***************************************************************/
/***************************************************************/
typedef struct {
  double m00, m01, m02, m11, m12, m22;
} symmetric_matrix;

/* rotate A by a unitary (real) rotation matrix R:
      RAR = transpose(R) * A * R
*/
void sym_matrix_rotate(symmetric_matrix *RAR, const symmetric_matrix *A_, const double R[3][3]) {
  int i, j;
  double A[3][3], AR[3][3];
  A[0][0] = A_->m00;
  A[1][1] = A_->m11;
  A[2][2] = A_->m22;
  A[0][1] = A[1][0] = A_->m01;
  A[0][2] = A[2][0] = A_->m02;
  A[1][2] = A[2][1] = A_->m12;
  for (i = 0; i < 3; ++i)
    for (j = 0; j < 3; ++j)
      AR[i][j] = A[i][0] * R[0][j] + A[i][1] * R[1][j] + A[i][2] * R[2][j];
  for (i = 0; i < 3; ++i)
    for (j = i; j < 3; ++j)
      A[i][j] = R[0][i] * AR[0][j] + R[1][i] * AR[1][j] + R[2][i] * AR[2][j];
  RAR->m00 = A[0][0];
  RAR->m11 = A[1][1];
  RAR->m22 = A[2][2];
  RAR->m01 = A[0][1];
  RAR->m02 = A[0][2];
  RAR->m12 = A[1][2];
}

/* Set Vinv = inverse of V, where both V and Vinv are real-symmetric matrices.*/
void sym_matrix_invert(symmetric_matrix *Vinv, const symmetric_matrix *V) {
  double m00 = V->m00, m11 = V->m11, m22 = V->m22;
  double m01 = V->m01, m02 = V->m02, m12 = V->m12;

  if (m01 == 0.0 && m02 == 0.0 && m12 == 0.0) {
    /* optimize common case of a diagonal matrix: */
    Vinv->m00 = 1.0 / m00;
    Vinv->m11 = 1.0 / m11;
    Vinv->m22 = 1.0 / m22;
    Vinv->m01 = Vinv->m02 = Vinv->m12 = 0.0;
  }
  else {
    double detinv;

    /* compute the determinant: */
    detinv = m00 * m11 * m22 - m02 * m11 * m02 + 2.0 * m01 * m12 * m02 - m01 * m01 * m22 -
             m12 * m12 * m00;

    if (detinv == 0.0) meep::abort("singular 3x3 matrix");

    detinv = 1.0 / detinv;

    Vinv->m00 = detinv * (m11 * m22 - m12 * m12);
    Vinv->m11 = detinv * (m00 * m22 - m02 * m02);
    Vinv->m22 = detinv * (m11 * m00 - m01 * m01);

    Vinv->m02 = detinv * (m01 * m12 - m11 * m02);
    Vinv->m01 = detinv * (m12 * m02 - m01 * m22);
    Vinv->m12 = detinv * (m01 * m02 - m00 * m12);
  }
}

/* Returns whether or not V is positive-definite. */
int sym_matrix_positive_definite(symmetric_matrix *V) {
  double det2, det3;
  double m00 = V->m00, m11 = V->m11, m22 = V->m22;

#if defined(WITH_HERMITIAN_EPSILON)
  scalar_complex m01 = V->m01, m02 = V->m02, m12 = V->m12;

  det2 = m00 * m11 - CSCALAR_NORMSQR(m01);
  det3 = det2 * m22 - m11 * CSCALAR_NORMSQR(m02) - CSCALAR_NORMSQR(m12) * m00 +
         2.0 * ((m01.re * m12.re - m01.im * m12.im) * m02.re +
                (m01.re * m12.im + m01.im * m12.re) * m02.im);
#else  /* real matrix */
  double m01 = V->m01, m02 = V->m02, m12 = V->m12;

  det2 = m00 * m11 - m01 * m01;
  det3 = det2 * m22 - m02 * m11 * m02 + 2.0 * m01 * m12 * m02 - m12 * m12 * m00;
#endif /* real matrix */

  return (m00 > 0.0 && det2 > 0.0 && det3 > 0.0);
}

/***************************************************************/
/***************************************************************/
/***************************************************************/
static meep::ndim dim = meep::D3;
void set_dimensions(int dims) {
  if (dims == CYLINDRICAL) { dim = meep::Dcyl; }
  else {
    dim = meep::ndim(dims - 1);
  }
}

vector3 vec_to_vector3(const meep::vec &pt) {
  vector3 v3;

  switch (pt.dim) {
    case meep::D1:
      v3.x = 0;
      v3.y = 0;
      v3.z = pt.z();
      break;
    case meep::D2:
      v3.x = pt.x();
      v3.y = pt.y();
      v3.z = 0;
      break;
    case meep::D3:
      v3.x = pt.x();
      v3.y = pt.y();
      v3.z = pt.z();
      break;
    case meep::Dcyl:
      v3.x = pt.r();
      v3.y = 0;
      v3.z = pt.z();
      break;
  }
  return v3;
}

meep::vec vector3_to_vec(const vector3 v3) {
  switch (dim) {
    case meep::D1: return meep::vec(v3.z);
    case meep::D2: return meep::vec(v3.x, v3.y);
    case meep::D3: return meep::vec(v3.x, v3.y, v3.z);
    case meep::Dcyl: return meep::veccyl(v3.x, v3.z);
    default: meep::abort("unknown dimensionality in vector3_to_vec");
  }
}

geom_box gv2box(const meep::volume &v) {
  geom_box box;
  box.low = vec_to_vector3(v.get_min_corner());
  box.high = vec_to_vector3(v.get_max_corner());
  return box;
}

bool is_material_grid(material_type mt) {
  return (mt->which_subclass == material_data::MATERIAL_GRID);
}
bool is_material_grid(void *md) { return is_material_grid((material_type)md); }

// return whether mt is spatially varying
bool is_variable(material_type mt) {
  return (mt && ((mt->which_subclass == material_data::MATERIAL_USER) ||
                 (mt->which_subclass == material_data::MATERIAL_GRID) ||
                 (mt->which_subclass == material_data::MATERIAL_FILE)));
}
bool is_variable(void *md) { return is_variable((material_type)md); }

bool is_medium(material_type md, medium_struct **m) {
  if (md->which_subclass == material_data::MEDIUM) {
    *m = &(md->medium);
    return true;
  };
  return false;
}

bool is_medium(void *md, medium_struct **m) { return is_medium((material_type)md, m); }

// note: is_metal assumes that eval_material_pt has already been called for variable materials
bool is_metal(meep::field_type ft, const material_type *material) {
  material_data *md = *material;
  if (ft == meep::E_stuff) switch (md->which_subclass) {
      case material_data::MEDIUM:
      case material_data::MATERIAL_USER:
      case material_data::MATERIAL_FILE:
      case material_data::MATERIAL_GRID:
        return (md->medium.epsilon_diag.x < 0 || md->medium.epsilon_diag.y < 0 ||
                md->medium.epsilon_diag.z < 0);
      case material_data::PERFECT_METAL: return true;
      default: meep::abort("unknown material type in is_metal"); return false;
    }
  else
    switch (md->which_subclass) {
      case material_data::MEDIUM:
      case material_data::MATERIAL_USER:
      case material_data::MATERIAL_FILE:
      case material_data::MATERIAL_GRID:
        return (md->medium.mu_diag.x < 0 || md->medium.mu_diag.y < 0 || md->medium.mu_diag.z < 0);
      case material_data::PERFECT_METAL:
        return false; // is an electric conductor, but not a magnetic conductor
      default: meep::abort("unknown material type in is_metal"); return false;
    }
}

// computes the vector-Jacobian product of the gradient of the matgrid_val function v
// with the Jacobian of the to_geom_box_coords function for geometric_object o
vector3 to_geom_object_coords_VJP(vector3 v, const geometric_object *o) {
  if (!o) { meep::abort("must pass a geometric_object to to_geom_object_coords_VJP.\n"); }

  switch (o->which_subclass) {
    default: {
      vector3 po = {0, 0, 0};
      return po;
    }
    case geometric_object::SPHERE: {
      number radius = o->subclass.sphere_data->radius;
      return vector3_scale(0.5 / radius, v);
    }
    /* case geometric_object::CYLINDER:
       NOT YET IMPLEMENTED */
    case geometric_object::BLOCK: {
      vector3 size = o->subclass.block_data->size;
      if (size.x != 0.0) v.x /= size.x;
      if (size.y != 0.0) v.y /= size.y;
      if (size.z != 0.0) v.z /= size.z;
      return matrix3x3_transpose_vector3_mult(o->subclass.block_data->projection_matrix, v);
    }
    /* case geometric_object::PRISM:
       NOT YET IMPLEMENTED */
  }
}

meep::vec material_grid_grad(vector3 p, material_data *md, const geometric_object *o) {
  if (!is_material_grid(md)) { meep::abort("Invalid material grid detected.\n"); }

  meep::vec gradient(zero_vec(dim));
  double *data = md->weights;
  int nx = md->grid_size.x;
  int ny = md->grid_size.y;
  int nz = md->grid_size.z;
  double rx = p.x;
  double ry = p.y;
  double rz = p.z;
  int stride = 1;
  int x1, y1, z1, x2, y2, z2;
  double dx, dy, dz;
  bool signflip_dx = false, signflip_dy = false, signflip_dz = false;

  meep::map_coordinates(rx, ry, rz, nx, ny, nz,
                        x1, y1, z1, x2, y2, z2,
                        dx, dy, dz,
                        false /* do_fabs */);

  if (dx != fabs(dx)) {
    dx = fabs(dx);
    signflip_dx = true;
  }
  if (dy != fabs(dy)) {
    dy = fabs(dy);
    signflip_dy = true;
  }
  if (dz != fabs(dz)) {
    dz = fabs(dz);
    signflip_dz = true;
  }

  /* define a macro to give us data(x,y,z) on the grid,
     in row-major order: */
#define D(x, y, z) (data[(((x)*ny + (y)) * nz + (z)) * stride])

  double du_dx = (signflip_dx ? -1.0 : 1.0) *
    (((-D(x1, y1, z1) + D(x2, y1, z1)) * (1.0 - dy) +
      (-D(x1, y2, z1) + D(x2, y2, z1)) * dy) * (1.0 - dz) +
     ((-D(x1, y1, z2) + D(x2, y1, z2)) * (1.0 - dy) +
      (-D(x1, y2, z2) + D(x2, y2, z2)) * dy) * dz);
  double du_dy = (signflip_dy ? -1.0 : 1.0) *
    ((-(D(x1, y1, z1) * (1.0 - dx) + D(x2, y1, z1) * dx) +
      (D(x1, y2, z1) * (1.0 - dx) + D(x2, y2, z1) * dx)) * (1.0 - dz) +
     (-(D(x1, y1, z2) * (1.0 - dx) + D(x2, y1, z2) * dx) +
      (D(x1, y2, z2) * (1.0 - dx) + D(x2, y2, z2) * dx)) * dz);
  double du_dz = (signflip_dz ? -1.0 : 1.0) *
    (-((D(x1, y1, z1) * (1.0 - dx) + D(x2, y1, z1) * dx) * (1.0 - dy) +
       (D(x1, y2, z1) * (1.0 - dx) + D(x2, y2, z1) * dx) * dy) +
     ((D(x1, y1, z2) * (1.0 - dx) + D(x2, y1, z2) * dx) * (1.0 - dy) +
      (D(x1, y2, z2) * (1.0 - dx) + D(x2, y2, z2) * dx) * dy));

#undef D

  // [du_dx,du_dy,du_dz] is the gradient ∇u with respect to the transformed coordinate
  // r1 of the matgrid_val function but what we want is the gradient of u(g(r2)) with
  // respect to r2 where g(r2) is the to_geom_object_coords function (in libctl/utils/geom.c).
  // computing this quantity involves using the chain rule and thus the vector-Jacobian product
  // ∇u J where J is the Jacobian matrix of g.
  vector3 grad_u;
  grad_u.x = du_dx * nx;
  grad_u.y = du_dy * ny;
  grad_u.z = du_dz * nz;
  if (o != NULL) {
    vector3 grad_u_J = to_geom_object_coords_VJP(grad_u, o);
    gradient.set_direction(meep::X, grad_u_J.x);
    gradient.set_direction(meep::Y, grad_u_J.y);
    gradient.set_direction(meep::Z, grad_u_J.z);
  }
  else {
    gradient.set_direction(meep::X, geometry_lattice.size.x == 0 ? 0 : grad_u.x / geometry_lattice.size.x);
    gradient.set_direction(meep::Y, geometry_lattice.size.y == 0 ? 0 : grad_u.y / geometry_lattice.size.y);
    gradient.set_direction(meep::Z, geometry_lattice.size.z == 0 ? 0 : grad_u.z / geometry_lattice.size.z);
  }

  return gradient;
}

void map_lattice_coordinates(double &px, double &py, double &pz) {
  px = geometry_lattice.size.x == 0 ? 0
    : 0.5 + (px - geometry_center.x) / geometry_lattice.size.x;
  py = geometry_lattice.size.y == 0 ? 0
    : 0.5 + (py - geometry_center.y) / geometry_lattice.size.y;
  pz = geometry_lattice.size.z == 0 ? 0
    : 0.5 + (pz - geometry_center.z) / geometry_lattice.size.z;
}

meep::vec matgrid_grad(vector3 p, geom_box_tree tp, int oi, material_data *md) {
  if (md->material_grid_kinds == material_data::U_MIN ||
      md->material_grid_kinds == material_data::U_PROD)
    meep::abort("%s:%i:matgrid_grad does not support overlapping grids with U_MIN or U_PROD\n",__FILE__,__LINE__);

  meep::vec gradient(zero_vec(dim));
  int matgrid_val_count = 0;

  // iterate through object tree at current point
  if (tp) {
    do {
      gradient += material_grid_grad(to_geom_box_coords(p, &tp->objects[oi]),
                                     (material_data *)tp->objects[oi].o->material,
                                     tp->objects[oi].o);
      if (md->material_grid_kinds == material_data::U_DEFAULT) break;
      ++matgrid_val_count;
      tp = geom_tree_search_next(p, tp, &oi);
    } while (tp && is_material_grid((material_data *)tp->objects[oi].o->material));
  }
  // perhaps there is no object tree and the default material is a material grid
  if (!tp && is_material_grid(default_material)) {
    map_lattice_coordinates(p.x,p.y,p.z);
    gradient = material_grid_grad(p, (material_data *)default_material, NULL /* geometric_object *o */);
    ++matgrid_val_count;
  }

  if (md->material_grid_kinds == material_data::U_MEAN)
    gradient = gradient * 1.0/matgrid_val_count;

  return gradient;
}

double material_grid_val(vector3 p, material_data *md) {
  // given the relative location, p, interpolate the material grid point.

  if (!is_material_grid(md)) { meep::abort("Invalid material grid detected.\n"); }
  return meep::linear_interpolate(p.x, p.y, p.z, md->weights, md->grid_size.x,
                                  md->grid_size.y, md->grid_size.z, 1);

}

static double tanh_projection(double u, double beta, double eta) {
  if (beta == 0) return u;
  double tanh_beta_eta = tanh(beta*eta);
  return (tanh_beta_eta + tanh(beta*(u-eta))) /
    (tanh_beta_eta + tanh(beta*(1-eta)));
}

double matgrid_val(vector3 p, geom_box_tree tp, int oi, material_data *md) {
  double uprod = 1.0, umin = 1.0, usum = 0.0, udefault = 0.0, u;
  int matgrid_val_count = 0;

  // iterate through object tree at current point
  if (tp) {
    do {
      u = material_grid_val(to_geom_box_coords(p, &tp->objects[oi]),
                            (material_data *)tp->objects[oi].o->material);
      if (md->material_grid_kinds == material_data::U_DEFAULT) {
        udefault = u;
        break;
      }
      if (u < umin) umin = u;
      uprod *= u;
      usum += u;
      ++matgrid_val_count;
      tp = geom_tree_search_next(p, tp, &oi);
    } while (tp && is_material_grid((material_data *)tp->objects[oi].o->material));
  }
  // perhaps there is no object tree and the default material is a material grid
  if (!tp && is_material_grid(default_material)) {
    map_lattice_coordinates(p.x,p.y,p.z);
    u = material_grid_val(p, (material_data *)default_material);
    if (matgrid_val_count == 0) udefault = u;
    if (u < umin) umin = u;
    uprod *= u;
    usum += u;
    ++matgrid_val_count;
  }

  return (md->material_grid_kinds == material_data::U_MIN
          ? umin
          : (md->material_grid_kinds == material_data::U_PROD
             ? uprod
             : (md->material_grid_kinds == material_data::U_MEAN ? usum / matgrid_val_count
                : udefault)));
}
static void cinterp_tensors(vector3 diag_in_1, cvector3 offdiag_in_1, vector3 diag_in_2,
                            cvector3 offdiag_in_2, vector3 *diag_out, cvector3 *offdiag_out,
                            double u) {
  /* convenience routine to interpolate material tensors with real and imaginary components */
  diag_out->x = diag_in_1.x + u * (diag_in_2.x - diag_in_1.x);
  diag_out->y = diag_in_1.y + u * (diag_in_2.y - diag_in_1.y);
  diag_out->z = diag_in_1.z + u * (diag_in_2.z - diag_in_1.z);
  offdiag_out->x.re = offdiag_in_1.x.re + u * (offdiag_in_2.x.re - offdiag_in_1.x.re);
  offdiag_out->x.im = offdiag_in_1.x.im + u * (offdiag_in_2.x.im - offdiag_in_1.x.im);
  offdiag_out->y.re = offdiag_in_1.y.re + u * (offdiag_in_2.y.re - offdiag_in_1.y.re);
  offdiag_out->y.im = offdiag_in_1.y.im + u * (offdiag_in_2.y.im - offdiag_in_1.y.im);
  offdiag_out->z.re = offdiag_in_1.z.re + u * (offdiag_in_2.z.re - offdiag_in_1.z.re);
  offdiag_out->z.im = offdiag_in_1.z.im + u * (offdiag_in_2.z.im - offdiag_in_1.z.im);
}

static void interp_tensors(vector3 diag_in_1, vector3 offdiag_in_1, vector3 diag_in_2,
                           vector3 offdiag_in_2, vector3 *diag_out, vector3 *offdiag_out,
                           double u) {
  /* convenience routine to interpolate material tensors with all real components */
  diag_out->x = diag_in_1.x + u * (diag_in_2.x - diag_in_1.x);
  diag_out->y = diag_in_1.y + u * (diag_in_2.y - diag_in_1.y);
  diag_out->z = diag_in_1.z + u * (diag_in_2.z - diag_in_1.z);
  offdiag_out->x = offdiag_in_1.x + u * (offdiag_in_2.x - offdiag_in_1.x);
  offdiag_out->y = offdiag_in_1.y + u * (offdiag_in_2.y - offdiag_in_1.y);
  offdiag_out->z = offdiag_in_1.z + u * (offdiag_in_2.z - offdiag_in_1.z);
}
// return material of the point p from the material grid
void epsilon_material_grid(material_data *md, double u) {
  // NOTE: assume p lies on normalized grid within (0,1)

  if (!(md->weights)) meep::abort("material params were not initialized!");

  medium_struct *mm = &(md->medium);
  medium_struct *m1 = &(md->medium_1);
  medium_struct *m2 = &(md->medium_2);

  // Linearly interpolate dc epsilon values
  cinterp_tensors(m1->epsilon_diag, m1->epsilon_offdiag, m2->epsilon_diag, m2->epsilon_offdiag,
                  &mm->epsilon_diag, &mm->epsilon_offdiag, u);

  // Interpolate resonant strength from d.p.
  vector3 zero_vec;
  zero_vec.x = zero_vec.y = zero_vec.z = 0;
  for (size_t i = 0; i < m1->E_susceptibilities.size(); i++) {
    // iterate through medium1 sus list first
    interp_tensors(zero_vec, zero_vec, m1->E_susceptibilities[i].sigma_diag,
                   m1->E_susceptibilities[i].sigma_offdiag,
                   &mm->E_susceptibilities[i].sigma_diag,
                   &mm->E_susceptibilities[i].sigma_offdiag, (1 - u));
  }
  for (size_t i = 0; i < m2->E_susceptibilities.size(); i++) {
    // iterate through medium2 sus list next
    size_t j = i + m1->E_susceptibilities.size();
    interp_tensors(zero_vec, zero_vec, m2->E_susceptibilities[i].sigma_diag,
                   m2->E_susceptibilities[i].sigma_offdiag,
                   &mm->E_susceptibilities[j].sigma_diag,
                   &mm->E_susceptibilities[j].sigma_offdiag, u);
  }

  // Linearly interpolate electric conductivity
  vector3 zero_offdiag;
  interp_tensors(m1->D_conductivity_diag, zero_vec, m2->D_conductivity_diag, zero_vec,
                 &mm->D_conductivity_diag, &zero_offdiag, u);

  // Add damping factor if we have dispersion.
  // This prevents instabilities when interpolating between sus. profiles.
  if ((m1->E_susceptibilities.size() + m2->E_susceptibilities.size()) > 0.0) {
    // calculate mean harmonic frequency
    double omega_mean = 0;
    for (const susceptibility &m1_sus : m1->E_susceptibilities) {
      omega_mean += m1_sus.frequency;
    }
    for (const susceptibility &m2_sus : m2->E_susceptibilities) {
      omega_mean += m2_sus.frequency;
    }
    omega_mean = omega_mean / (m1->E_susceptibilities.size() + m2->E_susceptibilities.size());

    // assign interpolated, nondimensionalized conductivity term
    // TODO: dampen the lorentzians to improve stability
    // mm->D_conductivity_diag.x = mm->D_conductivity_diag.y = mm->D_conductivity_diag.z = u*(1-u) *
    // omega_mean;
  }
}

// return material of the point p from the file (assumed already read)
void epsilon_file_material(material_data *md, vector3 p) {
  set_default_material(md);

  if (md->which_subclass != material_data::MATERIAL_FILE)
    meep::abort("epsilon-input-file only works with a type=file default-material");

  if (!(md->epsilon_data)) return;
  medium_struct *mm = &(md->medium);
  double rx =
      geometry_lattice.size.x == 0 ? 0 : 0.5 + (p.x - geometry_center.x) / geometry_lattice.size.x;
  double ry =
      geometry_lattice.size.y == 0 ? 0 : 0.5 + (p.y - geometry_center.y) / geometry_lattice.size.y;
  double rz =
      geometry_lattice.size.z == 0 ? 0 : 0.5 + (p.z - geometry_center.z) / geometry_lattice.size.z;
  mm->epsilon_diag.x = mm->epsilon_diag.y = mm->epsilon_diag.z =
      meep::linear_interpolate(rx, ry, rz, md->epsilon_data, md->epsilon_dims[0],
                               md->epsilon_dims[1], md->epsilon_dims[2], 1);
  mm->epsilon_offdiag.x.re = mm->epsilon_offdiag.y.re = mm->epsilon_offdiag.z.re = 0;
}

struct pol {
  susceptibility user_s;
  struct pol *next;
};

// structure to hold a conductivity profile (for scalar absorbing layers)
struct cond_profile {
  double L;     // thickness
  int N;        // number of points prof[n] from 0..N corresponding to 0..L
  double *prof; // (NULL if none)
};

class geom_epsilon : public meep::material_function {
  geometric_object_list geometry;
  geom_box_tree geometry_tree;
  geom_box_tree restricted_tree;

  cond_profile cond[5][2]; // [direction][side]

public:
  geom_epsilon(geometric_object_list g, material_type_list mlist, const meep::volume &v);
  virtual ~geom_epsilon();

  virtual void set_cond_profile(meep::direction, meep::boundary_side, double L, double dx,
                                double (*prof)(int, double *, void *), void *, double R);

  virtual void set_volume(const meep::volume &v);
  virtual void unset_volume(void);

  bool has_chi(meep::component c, int p);
  virtual bool has_chi3(meep::component c);
  virtual bool has_chi2(meep::component c);

  double chi(meep::component c, const meep::vec &r, int p);
  virtual double chi3(meep::component c, const meep::vec &r);
  virtual double chi2(meep::component c, const meep::vec &r);

  virtual bool has_mu();

  virtual bool has_conductivity(meep::component c);
  virtual double conductivity(meep::component c, const meep::vec &r);

  virtual double chi1p1(meep::field_type ft, const meep::vec &r);
  virtual void eff_chi1inv_row(meep::component c, double chi1inv_row[3], const meep::volume &v,
                               double tol, int maxeval);

  void eff_chi1inv_matrix(meep::component c, symmetric_matrix *chi1inv_matrix,
                          const meep::volume &v, double tol, int maxeval, bool &fallback);

  void fallback_chi1inv_row(meep::component c, double chi1inv_row[3], const meep::volume &v,
                            double tol, int maxeval);

  virtual void sigma_row(meep::component c, double sigrow[3], const meep::vec &r);
  void add_susceptibilities(meep::structure *s);
  void add_susceptibilities(meep::field_type ft, meep::structure *s);

private:
  void get_material_pt(material_type &material, const meep::vec &r);
  void eval_material_pt(material_type &material, vector3 p);

  material_type_list extra_materials;
  pol *current_pol;
};

/***********************************************************************/

geom_epsilon::geom_epsilon(geometric_object_list g, material_type_list mlist,
                           const meep::volume &v) {
  geometry = g; // don't bother making a copy, only used in one place
  extra_materials = mlist;
  current_pol = NULL;

  FOR_DIRECTIONS(d) FOR_SIDES(b) { cond[d][b].prof = NULL; }

  if (meep::am_master()) {
    int num_print = meep::verbosity > 2
                        ? geometry.num_items
                        : std::min(geometry.num_items, meep::verbosity > 0 ? 10 : 0);
    for (int i = 0; i < geometry.num_items; ++i) {

      if (i < num_print) display_geometric_object_info(5, geometry.items[i]);

      medium_struct *mm;
      if (is_medium(geometry.items[i].material, &mm)) {
        mm->check_offdiag_im_zero_or_abort();
        if (i < num_print)
          master_printf("%*sdielectric constant epsilon diagonal "
                        "= (%g,%g,%g)\n",
                        5 + 5, "", mm->epsilon_diag.x, mm->epsilon_diag.y, mm->epsilon_diag.z);
      }
    }
    if (num_print < geometry.num_items && meep::verbosity > 0)
      master_printf("%*s...(+ %d objects not shown)...\n", 5, "", geometry.num_items - num_print);
  }
  geom_fix_object_list(geometry);
  geom_box box = gv2box(v);
  geometry_tree = create_geom_box_tree0(geometry, box);
  if (meep::verbosity > 2 && meep::am_master()) {
    master_printf("Geometric-object bounding-box tree:\n");
    display_geom_box_tree(5, geometry_tree);

    int tree_depth, tree_nobjects;
    geom_box_tree_stats(geometry_tree, &tree_depth, &tree_nobjects);
    master_printf("Geometric object tree has depth %d "
                  "and %d object nodes (vs. %d actual objects)\n",
                  tree_depth, tree_nobjects, geometry.num_items);
  }

  restricted_tree = geometry_tree;
}

geom_epsilon::~geom_epsilon() {
  unset_volume();
  destroy_geom_box_tree(geometry_tree);
  FOR_DIRECTIONS(d) FOR_SIDES(b) {
    if (cond[d][b].prof) delete[] cond[d][b].prof;
  }
}

void geom_epsilon::set_cond_profile(meep::direction dir, meep::boundary_side side, double L,
                                    double dx, double (*P)(int, double *, void *), void *data,
                                    double R) {
  if (cond[dir][side].prof) delete[] cond[dir][side].prof;

  int N = int(L / dx + 0.5);
  cond[dir][side].L = L;
  cond[dir][side].N = N;
  double *prof = cond[dir][side].prof = new double[N + 1];

  double umin = 0, umax = 1, esterr;
  int errflag;
  double prof_int =
      adaptive_integration(P, &umin, &umax, 1, data, 1e-9, 1e-4, 50000, &esterr, &errflag);

  double prefac = (-log(R)) / (4 * L * prof_int);
  for (int i = 0; i <= N; ++i) {
    double u = double(i) / N;
    prof[i] = prefac * P(1, &u, data);
  }
}

void geom_epsilon::unset_volume(void) {
  if (restricted_tree != geometry_tree) {
    destroy_geom_box_tree(restricted_tree);
    restricted_tree = geometry_tree;
  }
}

void geom_epsilon::set_volume(const meep::volume &v) {
  unset_volume();

  geom_box box = gv2box(v);
  restricted_tree = create_geom_box_tree0(geometry, box);
}

static void material_epsmu(meep::field_type ft, material_type material, symmetric_matrix *epsmu,
                           symmetric_matrix *epsmu_inv) {

  material_data *md = material;
  if (ft == meep::E_stuff) switch (md->which_subclass) {

      case material_data::MEDIUM:
      case material_data::MATERIAL_FILE:
      case material_data::MATERIAL_USER:
      case material_data::MATERIAL_GRID:
        epsmu->m00 = md->medium.epsilon_diag.x;
        epsmu->m11 = md->medium.epsilon_diag.y;
        epsmu->m22 = md->medium.epsilon_diag.z;
        epsmu->m01 = md->medium.epsilon_offdiag.x.re;
        epsmu->m02 = md->medium.epsilon_offdiag.y.re;
        epsmu->m12 = md->medium.epsilon_offdiag.z.re;
        sym_matrix_invert(epsmu_inv, epsmu);
        break;

      case material_data::PERFECT_METAL:
        epsmu->m00 = -meep::infinity;
        epsmu->m11 = -meep::infinity;
        epsmu->m22 = -meep::infinity;
        epsmu_inv->m00 = -0.0;
        epsmu_inv->m11 = -0.0;
        epsmu_inv->m22 = -0.0;
        epsmu->m01 = epsmu->m02 = epsmu->m12 = 0.0;
        epsmu_inv->m01 = epsmu_inv->m02 = epsmu_inv->m12 = 0.0;
        break;

      default: meep::abort("unknown material type in epsmu");
    }
  else
    switch (md->which_subclass) {
      case material_data::MEDIUM:
      case material_data::MATERIAL_FILE:
      case material_data::MATERIAL_USER:
      case material_data::MATERIAL_GRID:
        epsmu->m00 = md->medium.mu_diag.x;
        epsmu->m11 = md->medium.mu_diag.y;
        epsmu->m22 = md->medium.mu_diag.z;
        epsmu->m01 = md->medium.mu_offdiag.x.re;
        epsmu->m02 = md->medium.mu_offdiag.y.re;
        epsmu->m12 = md->medium.mu_offdiag.z.re;
        sym_matrix_invert(epsmu_inv, epsmu);
        break;

      case material_data::PERFECT_METAL:
        epsmu->m00 = 1.0;
        epsmu->m11 = 1.0;
        epsmu->m22 = 1.0;
        epsmu_inv->m00 = 1.0;
        epsmu_inv->m11 = 1.0;
        epsmu_inv->m22 = 1.0;
        epsmu->m01 = epsmu->m02 = epsmu->m12 = 0.0;
        epsmu_inv->m01 = epsmu_inv->m02 = epsmu_inv->m12 = 0.0;
        break;

      default: meep::abort("unknown material type in epsmu");
    }
}

// the goal of this routine is to fill in the 'medium' field
// within the material structure as appropriate for the
// material properties at r.
void geom_epsilon::get_material_pt(material_type &material, const meep::vec &r) {
  vector3 p = vec_to_vector3(r);
  boolean inobject;
  material =
      (material_type)material_of_unshifted_point_in_tree_inobject(p, restricted_tree, &inobject);
  eval_material_pt(material, p);
}

// evaluate the material at the given point p if necessary — this is needed if
// the material is variable (a grid, function, or file); otherwise it is a no-op.
void geom_epsilon::eval_material_pt(material_type &material, vector3 p) {
  switch (material->which_subclass) {
    // material grid: interpolate onto user specified material grid to get properties at r
    case material_data::MATERIAL_GRID: {
      double u;
      int oi;
      geom_box_tree tp;
      tp = geom_tree_search(p, restricted_tree, &oi);
      // interpolate and project onto material grid
      u = tanh_projection(matgrid_val(p, tp, oi, material), material->beta, material->eta);
      // interpolate material from material grid point
      epsilon_material_grid(material, u);

      return;
    }
    // material read from file: interpolate to get properties at r
    case material_data::MATERIAL_FILE:
      if (material->epsilon_data)
        epsilon_file_material(material, p);
      else
        material = (material_type)default_material;
      return;

    // material specified by user-supplied function: call user
    // function to get properties at r.
    // Note that we initialize the medium to vacuum, so that
    // the user's function only needs to fill in whatever is
    // different from vacuum.
    case material_data::MATERIAL_USER:
      material->medium = medium_struct();
      material->user_func(p, material->user_data, &(material->medium));
      material->medium.check_offdiag_im_zero_or_abort();
      return;

    // position-independent material or metal: there is nothing to do
    case material_data::MEDIUM:
    case material_data::PERFECT_METAL: return;

    default: meep::abort("unknown material type in eval_material_pt");
  };
}

// returns trace of the tensor diagonal
double geom_epsilon::chi1p1(meep::field_type ft, const meep::vec &r) {
  symmetric_matrix chi1p1, chi1p1_inv;

#ifdef DEBUG
  vector3 p = vec_to_vector3(r);
  if (p.x < restricted_tree->b.low.x || p.y < restricted_tree->b.low.y ||
      p.z < restricted_tree->b.low.z || p.x > restricted_tree->b.high.x ||
      p.y > restricted_tree->b.high.y || p.z > restricted_tree->b.high.z)
    meep::abort("invalid point (%g,%g,%g)\n", p.x, p.y, p.z);
#endif

  material_type material;
  get_material_pt(material, r);
  material_epsmu(ft, material, &chi1p1, &chi1p1_inv);
  material_gc(material);

  return (chi1p1.m00 + chi1p1.m11 + chi1p1.m22) / 3;
}

/* Find frontmost object in v, along with the constant material behind it.
   Returns false if more than two objects & materials intersect the pixel.

   Requires moderately horrifying logic to figure things out properly,
   stolen from MPB. */
static bool get_front_object(const meep::volume &v, geom_box_tree geometry_tree, vector3 &pcenter,
                             const geometric_object **o_front, vector3 &shiftby_front,
                             material_type &mat_front, material_type &mat_behind,
                             vector3 &p_front, vector3 &p_behind) {
  vector3 p;
  const geometric_object *o1 = 0, *o2 = 0;
  vector3 shiftby1 = {0, 0, 0}, shiftby2 = {0, 0, 0}, p1 = {0,0,0}, p2 = {0,0,0};
  geom_box pixel;
  material_type mat1 = vacuum, mat2 = vacuum;
  int id1 = -1, id2 = -1;
  const int num_neighbors[3] = {3, 5, 9};
  const int neighbors[3][9][3] = {{{0, 0, 0},
                                   {0, 0, -1},
                                   {0, 0, 1},
                                   {0, 0, 0},
                                   {0, 0, 0},
                                   {0, 0, 0},
                                   {0, 0, 0},
                                   {0, 0, 0},
                                   {0, 0, 0}},
                                  {{0, 0, 0},
                                   {-1, -1, 0},
                                   {1, 1, 0},
                                   {-1, 1, 0},
                                   {1, -1, 0},
                                   {0, 0, 0},
                                   {0, 0, 0},
                                   {0, 0, 0},
                                   {0, 0, 0}},
                                  {{0, 0, 0},
                                   {1, 1, 1},
                                   {1, 1, -1},
                                   {1, -1, 1},
                                   {1, -1, -1},
                                   {-1, 1, 1},
                                   {-1, 1, -1},
                                   {-1, -1, 1},
                                   {-1, -1, -1}}};
  pixel = gv2box(v);
  pcenter = p = vec_to_vector3(v.center());
  double d1, d2, d3;
  d1 = (pixel.high.x - pixel.low.x) * 0.5;
  d2 = (pixel.high.y - pixel.low.y) * 0.5;
  d3 = (pixel.high.z - pixel.low.z) * 0.5;
  int dimension_index = meep::number_of_directions(dim) - 1;
  for (int i = 0; i < num_neighbors[dimension_index]; ++i) {
    const geometric_object *o;
    material_type mat;
    vector3 q, shiftby;
    int id;
    q.x = p.x + neighbors[dimension_index][i][0] * d1;
    q.y = p.y + neighbors[dimension_index][i][1] * d2;
    q.z = p.z + neighbors[dimension_index][i][2] * d3;
    o = object_of_point_in_tree(q, geometry_tree, &shiftby, &id);
    if ((id == id1 && vector3_equal(shiftby, shiftby1)) ||
        (id == id2 && vector3_equal(shiftby, shiftby2)))
      continue;

    mat = (material_type)default_material;
    if (o) {
      material_data *md = (material_data *)o->material;
      if (md->which_subclass != material_data::MATERIAL_FILE) mat = md;
    }
    if (id1 == -1) {
      o1 = o;
      shiftby1 = shiftby;
      id1 = id;
      mat1 = mat;
      p1 = q;
    }
    else if (id2 == -1 ||
             ((id >= id1 && id >= id2) && (id1 == id2 || material_type_equal(mat1, mat2)))) {
      o2 = o;
      shiftby2 = shiftby;
      id2 = id;
      mat2 = mat;
      p2 = q;
    }
    else if (!(id1 < id2 && (id1 == id || material_type_equal(mat1, mat))) &&
             !(id2 < id1 && (id2 == id || material_type_equal(mat2, mat))))
      return false;
  }

  // CHECK(id1 > -1, "bug in object_of_point_in_tree?");
  if (id2 == -1) { /* only one nearby object/material */
    id2 = id1;
    o2 = o1;
    mat2 = mat1;
    p2 = p1;
    shiftby2 = shiftby1;
  }

  if (id1 >= id2) {
    *o_front = o1;
    shiftby_front = shiftby1;
    mat_front = mat1;
    p_front = p1;
    if (id1 == id2) {
      mat_behind = mat1;
      p_behind = p1;
    }
    else {
      mat_behind = mat2;
      p_behind = p2;
    }
  }
  if (id2 > id1) {
    *o_front = o2;
    shiftby_front = shiftby2;
    mat_front = mat2;
    p_front = p2;
    mat_behind = mat1;
    p_behind = p1;
  }
  return true;
}

void geom_epsilon::eff_chi1inv_row(meep::component c, double chi1inv_row[3], const meep::volume &v,
                                   double tol, int maxeval) {
  symmetric_matrix meps_inv;
  bool fallback;
  eff_chi1inv_matrix(c, &meps_inv, v, tol, maxeval, fallback);

  if (fallback) { fallback_chi1inv_row(c, chi1inv_row, v, tol, maxeval); }
  else {
    switch (component_direction(c)) {
      case meep::X:
      case meep::R:
        chi1inv_row[0] = meps_inv.m00;
        chi1inv_row[1] = meps_inv.m01;
        chi1inv_row[2] = meps_inv.m02;
        break;
      case meep::Y:
      case meep::P:
        chi1inv_row[0] = meps_inv.m01;
        chi1inv_row[1] = meps_inv.m11;
        chi1inv_row[2] = meps_inv.m12;
        break;
      case meep::Z:
        chi1inv_row[0] = meps_inv.m02;
        chi1inv_row[1] = meps_inv.m12;
        chi1inv_row[2] = meps_inv.m22;
        break;
      case meep::NO_DIRECTION: chi1inv_row[0] = chi1inv_row[1] = chi1inv_row[2] = 0; break;
    }
  }
}

void geom_epsilon::eff_chi1inv_matrix(meep::component c, symmetric_matrix *chi1inv_matrix,
                                      const meep::volume &v, double tol, int maxeval,
                                      bool &fallback) {
  const geometric_object *o;
  material_type mat, mat_behind;
  vector3 p_mat, p_mat_behind;
  symmetric_matrix meps;
  vector3 p, shiftby, normal;
  fallback = false;

  if (maxeval == 0 ||
      !get_front_object(v, geometry_tree, p, &o, shiftby, mat, mat_behind, p_mat, p_mat_behind)) {
  noavg:
    get_material_pt(mat, v.center());
  trivial:
    material_epsmu(meep::type(c), mat, &meps, chi1inv_matrix);
    material_gc(mat);
    return;
  }

  // for variable materials with do_averaging == true, switch over to slow fallback integration method
  if ((is_variable(mat) && mat->do_averaging) || (is_variable(mat_behind) && mat_behind->do_averaging)) {
    fallback = true;
    return;
  }

  /* check for trivial case of only one object/material */
  if (material_type_equal(mat, mat_behind)) {
    if (is_variable(mat)) eval_material_pt(mat, vec_to_vector3(v.center()));
    goto trivial;
  }

  /* Evaluate materials in case they are variable.  This allows us to do fast subpixel averaging
     at the boundary of an object with a variable material, while remaining accurate enough if the
     material is continuous over the pixel.  (We make a first-order error by averaging as if the material
     were constant, but only in a boundary layer of thickness 1/resolution, so the net effect should
     still be second-order.) */
  eval_material_pt(mat, p_mat);
  eval_material_pt(mat_behind, p_mat_behind);
  if (material_type_equal(mat, mat_behind)) goto trivial;

  // it doesn't make sense to average metals (electric or magnetic)
  if (is_metal(meep::type(c), &mat) || is_metal(meep::type(c), &mat_behind)) goto noavg;

  normal = unit_vector3(normal_to_fixed_object(vector3_minus(p, shiftby), *o));
  if (normal.x == 0 && normal.y == 0 && normal.z == 0)
    goto noavg; // couldn't get normal vector for this point, punt
  geom_box pixel = gv2box(v);
  pixel.low = vector3_minus(pixel.low, shiftby);
  pixel.high = vector3_minus(pixel.high, shiftby);

  double fill = box_overlap_with_object(pixel, *o, tol, maxeval);

  material_epsmu(meep::type(c), mat, &meps, chi1inv_matrix);
  symmetric_matrix eps2, epsinv2;
  symmetric_matrix eps1, delta;
  double Rot[3][3];
  material_epsmu(meep::type(c), mat_behind, &eps2, &epsinv2);
  eps1 = meps;

  Rot[0][0] = normal.x;
  Rot[1][0] = normal.y;
  Rot[2][0] = normal.z;
  if (fabs(normal.x) > 1e-2 || fabs(normal.y) > 1e-2) {
    Rot[0][2] = normal.y;
    Rot[1][2] = -normal.x;
    Rot[2][2] = 0;
  }
  else { /* n is not parallel to z direction, use (x x n) instead */
    Rot[0][2] = 0;
    Rot[1][2] = -normal.z;
    Rot[2][2] = normal.y;
  }
  { /* normalize second column */
    double s = Rot[0][2] * Rot[0][2] + Rot[1][2] * Rot[1][2] + Rot[2][2] * Rot[2][2];
    s = 1.0 / sqrt(s);
    Rot[0][2] *= s;
    Rot[1][2] *= s;
    Rot[2][2] *= s;
  }
  /* 1st column is 2nd column x 0th column */
  Rot[0][1] = Rot[1][2] * Rot[2][0] - Rot[2][2] * Rot[1][0];
  Rot[1][1] = Rot[2][2] * Rot[0][0] - Rot[0][2] * Rot[2][0];
  Rot[2][1] = Rot[0][2] * Rot[1][0] - Rot[1][2] * Rot[0][0];

  /* rotate epsilon tensors to surface parallel/perpendicular axes */
  sym_matrix_rotate(&eps1, &eps1, Rot);
  sym_matrix_rotate(&eps2, &eps2, Rot);

#define AVG (fill * (EXPR(eps1)) + (1 - fill) * (EXPR(eps2)))
#define SQR(x) ((x) * (x))

#define EXPR(eps) (-1 / eps.m00)
  delta.m00 = AVG;
#undef EXPR
#define EXPR(eps) (eps.m11 - SQR(eps.m01) / eps.m00)
  delta.m11 = AVG;
#undef EXPR
#define EXPR(eps) (eps.m22 - SQR(eps.m02) / eps.m00)
  delta.m22 = AVG;
#undef EXPR

#define EXPR(eps) (eps.m01 / eps.m00)
  delta.m01 = AVG;
#undef EXPR
#define EXPR(eps) (eps.m02 / eps.m00)
  delta.m02 = AVG;
#undef EXPR
#define EXPR(eps) (eps.m12 - eps.m02 * eps.m01 / eps.m00)
  delta.m12 = AVG;
#undef EXPR

  meps.m00 = -1 / delta.m00;
  meps.m11 = delta.m11 - SQR(delta.m01) / delta.m00;
  meps.m22 = delta.m22 - SQR(delta.m02) / delta.m00;
  meps.m01 = -delta.m01 / delta.m00;
  meps.m02 = -delta.m02 / delta.m00;
  meps.m12 = delta.m12 - (delta.m02 * delta.m01) / delta.m00;

#undef SQR

#define SWAP(a, b)                                                                                 \
  {                                                                                                \
    double xxx = a;                                                                                \
    a = b;                                                                                         \
    b = xxx;                                                                                       \
  }
  /* invert rotation matrix = transpose */
  SWAP(Rot[0][1], Rot[1][0]);
  SWAP(Rot[0][2], Rot[2][0]);
  SWAP(Rot[2][1], Rot[1][2]);
  sym_matrix_rotate(&meps, &meps, Rot); /* rotate back */
#undef SWAP

#ifdef DEBUG
  if (!sym_matrix_positive_definite(&meps))
    meep::abort("negative mean epsilon from Kottke algorithm");
#endif

  sym_matrix_invert(chi1inv_matrix, &meps);
}

static int eps_ever_negative = 0;
static meep::field_type func_ft = meep::E_stuff;

struct matgrid_volavg {
  meep::ndim dim;        // dimensionality of voxel
  double rad;            // (spherical) voxel radius
  double uval;           // bilinearly-interpolated weight at voxel center
  double ugrad_abs;      // magnitude of gradient of uval
  double beta;           // thresholding bias
  double eta;            // thresholding erosion/dilation
  double eps1;           // trace of epsilon tensor from medium 1
  double eps2;           // trace of epsilon tensor from medium 2
};

static void get_uproj_w(const matgrid_volavg *mgva, double x0, double &u_proj, double &w) {
  // use a linear approximation for the material grid weights around the Yee grid point
  u_proj = tanh_projection(mgva->uval + mgva->ugrad_abs*x0, mgva->beta, mgva->eta);
  if (mgva->dim == meep::D1)
    w = 1/(2*mgva->rad);
  else if (mgva->dim == meep::D2 || mgva->dim == meep::Dcyl)
    w = 2*sqrt(mgva->rad*mgva->rad - x0*x0)/(meep::pi * mgva->rad*mgva->rad);
  else if (mgva->dim == meep::D3)
    w = meep::pi*(mgva->rad*mgva->rad - x0*x0)/(4/3 * meep::pi * mgva->rad*mgva->rad*mgva->rad);
}

#ifdef CTL_HAS_COMPLEX_INTEGRATION
static cnumber matgrid_ceps_func(int n, number *x, void *mgva_) {
  double u_proj = 0, w = 0;
  matgrid_volavg *mgva = (matgrid_volavg *)mgva_;
  get_uproj_w(mgva, x[0], u_proj, w);
  cnumber ret;
  ret.re = (1-u_proj)*mgva->eps1 + u_proj*mgva->eps2;
  ret.im = (1-u_proj)/mgva->eps1 + u_proj/mgva->eps2;
  return ret * w;
}
#else
static number matgrid_eps_func(int n, number *x, void *mgva_) {
  double u_proj = 0, w = 0;
  matgrid_volavg *mgva = (matgrid_volavg *)mgva_;
  get_uproj_w(mgva, x[0], u_proj, w);
  return w * ((1-u_proj)*mgva->eps1 + u_proj*mgva->eps2);
}
static number matgrid_inveps_func(int n, number *x, void *mgva_) {
  double u_proj = 0, w = 0;
  matgrid_volavg *mgva = (matgrid_volavg *)mgva_;
  get_uproj_w(mgva, x[0], u_proj, w);
  return w * ((1-u_proj)/mgva->eps1 + u_proj/mgva->eps2);
}
#endif

#ifdef CTL_HAS_COMPLEX_INTEGRATION
static cnumber ceps_func(int n, number *x, void *geomeps_) {
  geom_epsilon *geomeps = (geom_epsilon *)geomeps_;
  vector3 p = {0, 0, 0};
  p.x = x[0];
  p.y = n > 1 ? x[1] : 0;
  p.z = n > 2 ? x[2] : 0;
  double s = 1;
  if (dim == meep::Dcyl) {
    double py = p.y;
    p.y = p.z;
    p.z = py;
    s = p.x;
  }
  cnumber ret;
  double ep = geomeps->chi1p1(func_ft, vector3_to_vec(p));
  if (ep < 0) eps_ever_negative = 1;
  ret.re = ep * s;
  ret.im = s / ep;
  return ret;
}
#else
static number eps_func(int n, number *x, void *geomeps_) {
  geom_epsilon *geomeps = (geom_epsilon *)geomeps_;
  vector3 p = {0, 0, 0};
  double s = 1;
  p.x = x[0];
  p.y = n > 1 ? x[1] : 0;
  p.z = n > 2 ? x[2] : 0;
  if (dim == meep::Dcyl) {
    double py = p.y;
    p.y = p.z;
    p.z = py;
    s = p.x;
  }
  double ep = geomeps->chi1p1(func_ft, vector3_to_vec(p));
  if (ep < 0) eps_ever_negative = 1;
  return ep * s;
}
static number inveps_func(int n, number *x, void *geomeps_) {
  geom_epsilon *geomeps = (geom_epsilon *)geomeps_;
  vector3 p = {0, 0, 0};
  double s = 1;
  p.x = x[0];
  p.y = n > 1 ? x[1] : 0;
  p.z = n > 2 ? x[2] : 0;
  if (dim == meep::Dcyl) {
    double py = p.y;
    p.y = p.z;
    p.z = py;
    s = p.x;
  }
  double ep = geomeps->chi1p1(func_ft, vector3_to_vec(p));
  if (ep < 0) eps_ever_negative = 1;
  return s / ep;
}
#endif

// fallback meaneps using libctl's adaptive cubature routine
void geom_epsilon::fallback_chi1inv_row(meep::component c, double chi1inv_row[3],
                                        const meep::volume &v, double tol, int maxeval) {

  symmetric_matrix chi1p1, chi1p1_inv;
  material_type material;
  vector3 p = vec_to_vector3(v.center());
  boolean inobject;
  material =
      (material_type)material_of_unshifted_point_in_tree_inobject(p, restricted_tree, &inobject);
  material_data *md = material;
  meep::vec gradient(zero_vec(v.dim));
  double uval = 0;

  if (md->which_subclass == material_data::MATERIAL_GRID) {
    geom_box_tree tp;
    int oi;
    tp = geom_tree_search(p, restricted_tree, &oi);
    gradient = matgrid_grad(p, tp, oi, md);
    uval = matgrid_val(p, tp, oi, md);
  }
  else {
    gradient = normal_vector(meep::type(c), v);
  }

  get_material_pt(material, v.center());
  material_epsmu(meep::type(c), material, &chi1p1, &chi1p1_inv);
  material_gc(material);
  if (chi1p1.m01 != 0 || chi1p1.m02 != 0 || chi1p1.m12 != 0 || chi1p1.m00 != chi1p1.m11 ||
      chi1p1.m11 != chi1p1.m22 || chi1p1.m00 != chi1p1.m22 || meep::abs(gradient) < 1e-8) {
    int rownum = meep::component_direction(c) % 3;
    if (rownum == 0) {
      chi1inv_row[0] = chi1p1_inv.m00;
      chi1inv_row[1] = chi1p1_inv.m01;
      chi1inv_row[2] = chi1p1_inv.m02;
    }
    else if (rownum == 1) {
      chi1inv_row[0] = chi1p1_inv.m01;
      chi1inv_row[1] = chi1p1_inv.m11;
      chi1inv_row[2] = chi1p1_inv.m12;
    }
    else {
      chi1inv_row[0] = chi1p1_inv.m02;
      chi1inv_row[1] = chi1p1_inv.m12;
      chi1inv_row[2] = chi1p1_inv.m22;
    }
    return;
  }
  number esterr;
  integer errflag;
  double meps, minveps;

  if (md->which_subclass == material_data::MATERIAL_GRID) {
    number xmin[1], xmax[1];
    matgrid_volavg mgva;
    mgva.dim = v.dim;
    mgva.ugrad_abs = meep::abs(gradient);
    mgva.uval = uval;
    mgva.rad = v.diameter()/2;
    mgva.beta = md->beta;
    mgva.eta = md->eta;
    mgva.eps1 = (md->medium_1.epsilon_diag.x+md->medium_1.epsilon_diag.y+md->medium_1.epsilon_diag.z)/3;
    mgva.eps2 = (md->medium_2.epsilon_diag.x+md->medium_2.epsilon_diag.y+md->medium_2.epsilon_diag.z)/3;
    xmin[0] = -v.diameter()/2;
    xmax[0] = v.diameter()/2;
#ifdef CTL_HAS_COMPLEX_INTEGRATION
    cnumber ret = cadaptive_integration(matgrid_ceps_func, xmin, xmax, 1, (void *)&mgva, 0, tol, maxeval,
                                        &esterr, &errflag);
    meps = ret.re;
    minveps = ret.im;
#else
    meps = adaptive_integration(matgrid_eps_func, xmin, xmax, 1, (void *)&mgva, 0, tol, maxeval, &esterr,
                                &errflag);
    minveps = adaptive_integration(matgrid_inveps_func, xmin, xmax, 1, (void *)&mgva, 0, tol, maxeval, &esterr,
                                   &errflag);
#endif
  }
  else {
    integer n;
    number xmin[3], xmax[3];
    vector3 gvmin, gvmax;
    gvmin = vec_to_vector3(v.get_min_corner());
    gvmax = vec_to_vector3(v.get_max_corner());
    xmin[0] = gvmin.x;
    xmax[0] = gvmax.x;
    if (dim == meep::Dcyl) {
      xmin[1] = gvmin.z;
      xmin[2] = gvmin.y;
      xmax[1] = gvmax.z;
      xmax[2] = gvmax.y;
    }
    else {
      xmin[1] = gvmin.y;
      xmin[2] = gvmin.z;
      xmax[1] = gvmax.y;
      xmax[2] = gvmax.z;
    }
    if (xmin[2] == xmax[2])
      n = xmin[1] == xmax[1] ? 1 : 2;
    else
      n = 3;
    double vol = 1;
    for (int i = 0; i < n; ++i)
      vol *= xmax[i] - xmin[i];
    if (dim == meep::Dcyl) vol *= (xmin[0] + xmax[0]) * 0.5;
    eps_ever_negative = 0;
    func_ft = meep::type(c);
#ifdef CTL_HAS_COMPLEX_INTEGRATION
    cnumber ret = cadaptive_integration(ceps_func, xmin, xmax, n, (void *)this, 0, tol, maxeval,
                                        &esterr, &errflag);
    meps = ret.re / vol;
    minveps = ret.im / vol;
#else
    meps = adaptive_integration(eps_func, xmin, xmax, n, (void *)this, 0, tol, maxeval, &esterr,
                                &errflag) / vol;
    minveps = adaptive_integration(inveps_func, xmin, xmax, n, (void *)this, 0, tol, maxeval, &esterr,
                                   &errflag) / vol;
#endif
  }
  if (eps_ever_negative) // averaging negative eps causes instability
    minveps = 1.0 / (meps = eps(v.center()));
  {
    double n[3] = {0, 0, 0};
    double nabsinv = 1.0 / meep::abs(gradient);
    LOOP_OVER_DIRECTIONS(gradient.dim, k) { n[k % 3] = gradient.in_direction(k) * nabsinv; }
    int rownum = meep::component_direction(c) % 3;
    for (int i = 0; i < 3; ++i)
      chi1inv_row[i] = n[rownum] * n[i] * (minveps - 1 / meps);
    chi1inv_row[rownum] += 1 / meps;
  }
}

static double get_chi3(meep::component c, const medium_struct *m) {
  switch (c) {
    case meep::Er:
    case meep::Ex: return m->E_chi3_diag.x;
    case meep::Ep:
    case meep::Ey: return m->E_chi3_diag.y;
    case meep::Ez: return m->E_chi3_diag.z;
    case meep::Hr:
    case meep::Hx: return m->H_chi3_diag.x;
    case meep::Hp:
    case meep::Hy: return m->H_chi3_diag.y;
    case meep::Hz: return m->H_chi3_diag.z;
    default: return 0;
  }
}

static double get_chi2(meep::component c, const medium_struct *m) {
  switch (c) {
    case meep::Er:
    case meep::Ex: return m->E_chi2_diag.x;
    case meep::Ep:
    case meep::Ey: return m->E_chi2_diag.y;
    case meep::Ez: return m->E_chi2_diag.z;
    case meep::Hr:
    case meep::Hx: return m->H_chi2_diag.x;
    case meep::Hp:
    case meep::Hy: return m->H_chi2_diag.y;
    case meep::Hz: return m->H_chi2_diag.z;
    default: return 0;
  }
}

static double get_chi(meep::component c, const medium_struct *m, int p) {
  return ((p == 2) ? get_chi2(c, m) : get_chi3(c, m));
}

// the following routines consolidate the former has_chi2, has_chi3 and chi2, chi3
// p=2,3 for chi2,chi3
bool geom_epsilon::has_chi(meep::component c, int p) {
  medium_struct *mm;

  for (int i = 0; i < geometry.num_items; ++i)
    if (is_medium(geometry.items[i].material, &mm))
      if (get_chi(c, mm, p) != 0) return true;

  for (int i = 0; i < extra_materials.num_items; ++i)
    if (is_medium(extra_materials.items[i], &mm))
      if (get_chi(c, mm, p) != 0) return true;

  return (is_medium(default_material, &mm) && get_chi(c, mm, p) != 0);
}

bool geom_epsilon::has_chi3(meep::component c) { return has_chi(c, 3); }

bool geom_epsilon::has_chi2(meep::component c) { return has_chi(c, 2); }

double geom_epsilon::chi(meep::component c, const meep::vec &r, int p) {
  material_type material;
  get_material_pt(material, r);

  double chi_val;

  material_data *md = material;
  switch (md->which_subclass) {
    case material_data::MEDIUM:
    case material_data::MATERIAL_GRID:
    case material_data::MATERIAL_USER: chi_val = get_chi(c, &(md->medium), p); break;

    default: chi_val = 0;
  };

  material_gc(material);
  return chi_val;
}

double geom_epsilon::chi3(meep::component c, const meep::vec &r) { return chi(c, r, 3); }

double geom_epsilon::chi2(meep::component c, const meep::vec &r) { return chi(c, r, 2); }

static bool mu_not_1(material_type m) {
  medium_struct *mm;
  return (is_medium(m, &mm) &&
          (mm->mu_diag.x != 1 || mm->mu_diag.y != 1 || mm->mu_diag.z != 1 ||
           mm->mu_offdiag.x.re != 0 || mm->mu_offdiag.y.re != 0 || mm->mu_offdiag.z.re != 0));
}
static bool mu_not_1(void *m) { return mu_not_1((material_type)m); }

bool geom_epsilon::has_mu() {
  for (int i = 0; i < geometry.num_items; ++i)
    if (mu_not_1(geometry.items[i].material)) return true;

  for (int i = 0; i < extra_materials.num_items; ++i)
    if (mu_not_1(extra_materials.items[i])) return true;

  return (mu_not_1(default_material));
}

/* a global scalar conductivity to add to all materials; this
   is mostly for the convenience of Casimir calculations where
   the global conductivity corresponds to a rotation to
   complex frequencies */
static double global_D_conductivity = 0, global_B_conductivity = 0;

static double get_cnd(meep::component c, const medium_struct *m) {
  switch (c) {
    case meep::Dr:
    case meep::Dx: return m->D_conductivity_diag.x + global_D_conductivity;
    case meep::Dp:
    case meep::Dy: return m->D_conductivity_diag.y + global_D_conductivity;
    case meep::Dz: return m->D_conductivity_diag.z + global_D_conductivity;
    case meep::Br:
    case meep::Bx: return m->B_conductivity_diag.x + global_B_conductivity;
    case meep::Bp:
    case meep::By: return m->B_conductivity_diag.y + global_B_conductivity;
    case meep::Bz: return m->B_conductivity_diag.z + global_B_conductivity;
    default: return 0;
  }
}

bool geom_epsilon::has_conductivity(meep::component c) {
  medium_struct *mm;

  FOR_DIRECTIONS(d) FOR_SIDES(b) {
    if (cond[d][b].prof) return true;
  }

  for (int i = 0; i < geometry.num_items; ++i)
    if (is_medium(geometry.items[i].material, &mm) && get_cnd(c, mm)) return true;

  for (int i = 0; i < extra_materials.num_items; ++i)
    if (is_medium(extra_materials.items[i], &mm) && get_cnd(c, mm)) return true;

  return (is_medium(default_material, &mm) && get_cnd(c, mm) != 0);
}

static meep::vec geometry_edge; // geometry_lattice.size / 2
// TODO
double geom_epsilon::conductivity(meep::component c, const meep::vec &r) {
  material_type material;
  get_material_pt(material, r);

  double cond_val;
  material_data *md = material;
  switch (md->which_subclass) {
    case material_data::MEDIUM:
    case material_data::MATERIAL_GRID:
    case material_data::MATERIAL_USER: cond_val = get_cnd(c, &(md->medium)); break;
    default: cond_val = 0;
  }
  material_gc(material);

  // if the user specified scalar absorbing layers, add their conductivities
  // to cond_val (isotropically, for both magnetic and electric conductivity).
  LOOP_OVER_DIRECTIONS(r.dim, d) {
    double x = r.in_direction(d);
    double edge = geometry_edge.in_direction(d) - cond[d][meep::High].L;
    if (cond[d][meep::High].prof && x >= edge) {
      int N = cond[d][meep::High].N;
      double ui = N * (x - edge) / cond[d][meep::High].L;
      int i = int(ui);
      if (i >= N)
        cond_val += cond[d][meep::High].prof[N];
      else {
        double di = ui - i;
        cond_val += cond[d][meep::High].prof[i] * (1 - di) + cond[d][meep::High].prof[i + 1] * di;
      }
    }
    edge = cond[d][meep::Low].L - geometry_edge.in_direction(d);
    if (cond[d][meep::Low].prof && x <= edge) {
      int N = cond[d][meep::Low].N;
      double ui = N * (edge - x) / cond[d][meep::Low].L;
      int i = int(ui);
      if (i >= N)
        cond_val += cond[d][meep::Low].prof[N];
      else {
        double di = ui - i;
        cond_val += cond[d][meep::Low].prof[i] * (1 - di) + cond[d][meep::Low].prof[i + 1] * di;
      }
    }
  }

  return cond_val;
}

/* like susceptibility_equal in ctl-io.cpp, but ignores sigma and id
   (must be updated manually, re-copying from ctl-io.cpp), if we
   add new susceptibility subclasses) */
static bool susceptibility_equiv(const susceptibility &o0, const susceptibility &o) {
  if (!vector3_equal(o0.bias, o.bias)) return false;
  if (o0.frequency != o.frequency) return false;
  if (o0.gamma != o.gamma) return false;
  if (o0.alpha != o.alpha) return false;
  if (o0.noise_amp != o.noise_amp) return false;
  if (o0.drude != o.drude) return false;
  if (o0.saturated_gyrotropy != o.saturated_gyrotropy) return false;
  if (o0.is_file != o.is_file) return false;

  if (o0.transitions != o.transitions) return false;
  if (o0.initial_populations != o.initial_populations) return false;

  return true;
}

void geom_epsilon::sigma_row(meep::component c, double sigrow[3], const meep::vec &r) {

  vector3 p = vec_to_vector3(r);

  boolean inobject;
  material_type mat =
      (material_type)material_of_unshifted_point_in_tree_inobject(p, restricted_tree, &inobject);

  if (mat->which_subclass == material_data::MATERIAL_USER) {
    mat->medium = medium_struct();
    mat->user_func(p, mat->user_data, &(mat->medium));
    mat->medium.check_offdiag_im_zero_or_abort();
  }

  if (mat->which_subclass == material_data::MATERIAL_GRID) {
    double u;
    int oi;
    geom_box_tree tp;

    tp = geom_tree_search(p, restricted_tree, &oi);
    // interpolate and project onto material grid
    u = tanh_projection(matgrid_val(p, tp, oi, mat), mat->beta, mat->eta);
    epsilon_material_grid(mat, u);   // interpolate material from material grid point
    mat->medium.check_offdiag_im_zero_or_abort();
  }

  sigrow[0] = sigrow[1] = sigrow[2] = 0.0;

  if (mat->which_subclass == material_data::MATERIAL_USER ||
      mat->which_subclass == material_data::MATERIAL_GRID ||
      mat->which_subclass == material_data::MEDIUM) {

    const susceptibility_list &slist =
        type(c) == meep::E_stuff ? mat->medium.E_susceptibilities : mat->medium.H_susceptibilities;
    for (const susceptibility &susc : slist) {
      if (susceptibility_equiv(susc, current_pol->user_s)) {
        int ic = meep::component_index(c);
        switch (ic) { // which row of the sigma tensor to return
          case 0:
            sigrow[0] = susc.sigma_diag.x;
            sigrow[1] = susc.sigma_offdiag.x;
            sigrow[2] = susc.sigma_offdiag.y;
            break;
          case 1:
            sigrow[0] = susc.sigma_offdiag.x;
            sigrow[1] = susc.sigma_diag.y;
            sigrow[2] = susc.sigma_offdiag.z;
            break;
          default: // case 2:
            sigrow[0] = susc.sigma_offdiag.y;
            sigrow[1] = susc.sigma_offdiag.z;
            sigrow[2] = susc.sigma_diag.z;
            break;
        }
        break;
      }
    }
  }
  material_gc(mat);
}

/* make multilevel_susceptibility from python input data */
static meep::susceptibility *make_multilevel_sus(const susceptibility_struct *d) {
  if (!d || d->transitions.size() == 0) return NULL;

  // the user can number the levels however she wants, but we
  // will renumber them to 0...(L-1)
  int minlev = d->transitions[0].to_level;
  int maxlev = minlev;
  for (size_t t = 0; t < d->transitions.size(); ++t) {
    if (minlev > d->transitions[t].from_level) minlev = d->transitions[t].from_level;
    if (minlev > d->transitions[t].to_level) minlev = d->transitions[t].to_level;
    if (maxlev < d->transitions[t].from_level) maxlev = d->transitions[t].from_level;
    if (maxlev < d->transitions[t].to_level) maxlev = d->transitions[t].to_level;
  }
  size_t L = maxlev - minlev + 1; // number of atom levels

  // count number of radiative transitions
  int T = 0;
  for (size_t t = 0; t < d->transitions.size(); ++t)
    if (d->transitions[t].frequency != 0) ++T;
  if (T == 0) return NULL; // don't bother if there is no radiative coupling

  // non-radiative transition-rate matrix Gamma
  meep::realnum *Gamma = new meep::realnum[L * L];
  memset(Gamma, 0, sizeof(meep::realnum) * (L * L));
  for (size_t t = 0; t < d->transitions.size(); ++t) {
    int i = d->transitions[t].from_level - minlev;
    int j = d->transitions[t].to_level - minlev;
    Gamma[i * L + i] += +d->transitions[t].transition_rate + d->transitions[t].pumping_rate;
    Gamma[j * L + i] -= +d->transitions[t].transition_rate + d->transitions[t].pumping_rate;
  }

  // initial populations of each level
  meep::realnum *N0 = new meep::realnum[L];
  memset(N0, 0, sizeof(meep::realnum) * L);
  for (size_t p = 0; p < d->initial_populations.size() && p < L; ++p)
    N0[p] = d->initial_populations[p];

  meep::realnum *alpha = new meep::realnum[L * T];
  memset(alpha, 0, sizeof(meep::realnum) * (L * T));
  meep::realnum *omega = new meep::realnum[T];
  meep::realnum *gamma = new meep::realnum[T];
  meep::realnum *sigmat = new meep::realnum[T * 5];

  const double pi = 3.14159265358979323846264338327950288; // need pi below.

  for (size_t t = 0, tr = 0; t < d->transitions.size(); ++t)
    if (d->transitions[t].frequency != 0) {
      omega[tr] = d->transitions[t].frequency; // no 2*pi here
      gamma[tr] = d->transitions[t].gamma;
      if (dim == meep::Dcyl) {
        sigmat[5 * tr + meep::R] = d->transitions[t].sigma_diag.x;
        sigmat[5 * tr + meep::P] = d->transitions[t].sigma_diag.y;
        sigmat[5 * tr + meep::Z] = d->transitions[t].sigma_diag.z;
      }
      else {
        sigmat[5 * tr + meep::X] = d->transitions[t].sigma_diag.x;
        sigmat[5 * tr + meep::Y] = d->transitions[t].sigma_diag.y;
        sigmat[5 * tr + meep::Z] = d->transitions[t].sigma_diag.z;
      }
      int i = d->transitions[t].from_level - minlev;
      int j = d->transitions[t].to_level - minlev;
      alpha[i * T + tr] = -1.0 / (2 * pi * omega[tr]); // but we *do* need the 2*pi here. -- AWC
      alpha[j * T + tr] = +1.0 / (2 * pi * omega[tr]);
      ++tr;
    }

  meep::multilevel_susceptibility *s =
      new meep::multilevel_susceptibility(L, T, Gamma, N0, alpha, omega, gamma, sigmat);

  delete[] Gamma;
  delete[] N0;
  delete[] alpha;
  delete[] omega;
  delete[] gamma;
  delete[] sigmat;

  return s;
}

// add a polarization to the list if it is not already there
static pol *add_pol(pol *pols, const susceptibility &user_s) {
  struct pol *p = pols;
  while (p && !susceptibility_equiv(user_s, p->user_s))
    p = p->next;
  if (!p) {
    p = new pol;
    p->user_s = user_s;
    p->next = pols;
    pols = p;
  }
  return pols;
}

static pol *add_pols(pol *pols, const susceptibility_list &slist) {
  for (const susceptibility &susc : slist) {
    pols = add_pol(pols, susc);
  }
  return pols;
}

void geom_epsilon::add_susceptibilities(meep::structure *s) {
  add_susceptibilities(meep::E_stuff, s);
  add_susceptibilities(meep::H_stuff, s);
}

void geom_epsilon::add_susceptibilities(meep::field_type ft, meep::structure *s) {
  pol *pols = 0;
  medium_struct *mm;

  // construct a list of the unique susceptibilities in the geometry:
  for (int i = 0; i < geometry.num_items; ++i)
    if (is_medium(geometry.items[i].material, &mm))
      pols = add_pols(pols, ft == meep::E_stuff ? mm->E_susceptibilities : mm->H_susceptibilities);

  for (int i = 0; i < extra_materials.num_items; ++i)
    if (is_medium(extra_materials.items[i], &mm))
      pols = add_pols(pols, ft == meep::E_stuff ? mm->E_susceptibilities : mm->H_susceptibilities);

  if (is_medium(default_material, &mm))
    pols = add_pols(pols, ft == meep::E_stuff ? mm->E_susceptibilities : mm->H_susceptibilities);

  for (struct pol *p = pols; p; p = p->next) {

    susceptibility *ss = &(p->user_s);
    if (ss->is_file) meep::abort("unknown susceptibility");
    bool noisy = (ss->noise_amp != 0.0);
    bool gyrotropic =
        (ss->saturated_gyrotropy || ss->bias.x != 0.0 || ss->bias.y != 0.0 || ss->bias.z != 0.0);
    meep::susceptibility *sus;

    if (ss->transitions.size() != 0 || ss->initial_populations.size() != 0) {
      // multilevel atom
      sus = make_multilevel_sus(ss);
      if (meep::verbosity > 0) master_printf("multilevel atom susceptibility\n");
    }
    else {
      if (noisy) {
        sus = new meep::noisy_lorentzian_susceptibility(ss->noise_amp, ss->frequency, ss->gamma,
                                                        ss->drude);
      }
      else if (gyrotropic) {
        meep::gyrotropy_model model =
            ss->saturated_gyrotropy
                ? meep::GYROTROPIC_SATURATED
                : ss->drude ? meep::GYROTROPIC_DRUDE : meep::GYROTROPIC_LORENTZIAN;
        sus = new meep::gyrotropic_susceptibility(meep::vec(ss->bias.x, ss->bias.y, ss->bias.z),
                                                  ss->frequency, ss->gamma, ss->alpha, model);
      }
      else {
        sus = new meep::lorentzian_susceptibility(ss->frequency, ss->gamma, ss->drude);
      }
      if (meep::verbosity > 0) {
        master_printf("%s%s susceptibility: frequency=%g, gamma=%g",
                      noisy ? "noisy " : gyrotropic ? "gyrotropic " : "",
                      ss->saturated_gyrotropy ? "Landau-Lifshitz-Gilbert-type"
                                              : ss->drude ? "drude" : "lorentzian",
                      ss->frequency, ss->gamma);
        if (noisy) master_printf(", amp=%g ", ss->noise_amp);
        if (gyrotropic) {
          if (ss->saturated_gyrotropy) master_printf(", alpha=%g", ss->alpha);
          master_printf(", bias=(%g,%g,%g)", ss->bias.x, ss->bias.y, ss->bias.z);
        }
        master_printf("\n");
      }
    }

    current_pol = p;
    if (sus) {
      s->add_susceptibility(*this, ft, *sus);
      delete sus;
    }
  }
  current_pol = NULL;

  while (pols) {
    struct pol *p = pols;
    pols = pols->next;
    delete p;
  }
}

typedef struct pml_profile_thunk {
  meep::pml_profile_func func;
  void *func_data;
} pml_profile_thunk;

double pml_profile_wrapper(int dim, double *u, void *user_data) {
  (void)dim; // unused
  pml_profile_thunk *mythunk = (pml_profile_thunk *)user_data;
  return mythunk->func(u[0], mythunk->func_data);
}

/***************************************************************/
/* mechanism for allowing users to specify non-PML absorbing   */
/* layers.                                                     */
/***************************************************************/
absorber_list create_absorber_list() {
  absorber_list alist = new absorber_list_type;
  return alist;
}

void destroy_absorber_list(absorber_list alist) { delete alist; }

void add_absorbing_layer(absorber_list alist, double thickness, int direction, int side,
                         double R_asymptotic, double mean_stretch, meep::pml_profile_func func,
                         void *func_data) {
  absorber myabsorber;
  myabsorber.thickness = thickness;
  myabsorber.direction = direction;
  myabsorber.side = side;
  myabsorber.R_asymptotic = R_asymptotic;
  myabsorber.mean_stretch = mean_stretch;
  myabsorber.pml_profile = func;
  myabsorber.pml_profile_data = func_data;

  if (alist == 0) meep::abort("invalid absorber_list in add_absorbing_layer");

  alist->push_back(myabsorber);
}

/***************************************************************/
/***************************************************************/
/***************************************************************/
void set_materials_from_geometry(meep::structure *s, geometric_object_list g, vector3 center,
                                 bool use_anisotropic_averaging, double tol, int maxeval,
                                 bool _ensure_periodicity, material_type _default_material,
                                 absorber_list alist, material_type_list extra_materials) {
  // set global variables in libctlgeom based on data fields in s
  geom_initialize();
  geometry_center = center;

  if (_default_material->which_subclass != material_data::MATERIAL_USER &&
      _default_material->which_subclass != material_data::PERFECT_METAL) {
    _default_material->medium.check_offdiag_im_zero_or_abort();
  }
  set_default_material(_default_material);
  ensure_periodicity = _ensure_periodicity;
  meep::grid_volume gv = s->gv;
  double resolution = gv.a;

  int sim_dims = 3;
  vector3 size = {0.0, 0.0, 0.0};
  switch (s->user_volume.dim) {
    case meep::D1:
      sim_dims = 1;
      size.z = s->user_volume.nz() / resolution;
      break;
    case meep::D2:
      sim_dims = 2;
      size.x = s->user_volume.nx() / resolution;
      size.y = s->user_volume.ny() / resolution;
      break;
    case meep::D3:
      sim_dims = 3;
      size.x = s->user_volume.nx() / resolution;
      size.y = s->user_volume.ny() / resolution;
      size.z = s->user_volume.nz() / resolution;
      break;
    case meep::Dcyl:
      sim_dims = CYLINDRICAL;
      size.x = s->user_volume.nr() / resolution;
      size.z = s->user_volume.nz() / resolution;
      break;
  };

  set_dimensions(sim_dims);

  geometry_lattice.size = size;
  geometry_edge = vector3_to_vec(size) * 0.5;

  if (meep::verbosity > 0) {
    master_printf("Working in %s dimensions.\n", meep::dimension_name(s->gv.dim));
    master_printf("Computational cell is %g x %g x %g with resolution %g\n", size.x, size.y, size.z,
                  resolution);
  }

  geom_epsilon geps(g, extra_materials, gv.pad().surroundings());

  /***************************************************************/
  /***************************************************************/
  /***************************************************************/
  if (alist) {
    for (absorber_list_type::iterator layer = alist->begin(); layer != alist->end(); layer++) {
      LOOP_OVER_DIRECTIONS(gv.dim, d) {
        if (layer->direction != ALL_DIRECTIONS && layer->direction != d) continue;
        FOR_SIDES(b) {
          if (layer->side != ALL_SIDES && layer->side != b) continue;
          pml_profile_thunk mythunk;
          mythunk.func = layer->pml_profile;
          mythunk.func_data = layer->pml_profile_data;
          geps.set_cond_profile(d, b, layer->thickness, gv.inva * 0.5, pml_profile_wrapper,
                                (void *)&mythunk, layer->R_asymptotic);
        }
      }
    }
  }

  /***************************************************************/
  /***************************************************************/
  /***************************************************************/
  s->set_materials(geps, use_anisotropic_averaging, tol, maxeval);
  s->remove_susceptibilities();
  geps.add_susceptibilities(s);

  if (meep::verbosity > 0) master_printf("-----------\n");
}

/***************************************************************/
/* convenience routines for creating materials of various types*/
/***************************************************************/
material_type make_dielectric(double epsilon) {
  material_data *md = new material_data();
  md->medium.epsilon_diag.x = epsilon;
  md->medium.epsilon_diag.y = epsilon;
  md->medium.epsilon_diag.z = epsilon;
  return md;
}

material_type make_user_material(user_material_func user_func, void *user_data, bool do_averaging) {
  material_data *md = new material_data();
  md->which_subclass = material_data::MATERIAL_USER;
  md->user_func = user_func;
  md->user_data = user_data;
  md->do_averaging = do_averaging;
  return md;
}

// this routine subsumes the content of the old
// 'read_epsilon_file' routine
material_type make_file_material(const char *eps_input_file) {
  material_data *md = new material_data();
  md->which_subclass = material_data::MATERIAL_FILE;
  md->do_averaging = false;

  md->epsilon_dims[0] = md->epsilon_dims[1] = md->epsilon_dims[2] = 1;
  if (eps_input_file && eps_input_file[0]) { // file specified
    char *fname = new char[strlen(eps_input_file) + 1];
    strcpy(fname, eps_input_file);
    // parse epsilon-input-file as "fname.h5:dataname"
    char *dataname = strrchr(fname, ':');
    if (dataname) *(dataname++) = 0;
    meep::h5file eps_file(fname, meep::h5file::READONLY, false);
    int rank; // ignored since rank < 3 is equivalent to singleton dims
    md->epsilon_data = (double *)eps_file.read(dataname, &rank, md->epsilon_dims, 3, false);
    if (meep::verbosity > 0)
      master_printf("read in %zdx%zdx%zd epsilon-input-file \"%s\"\n", md->epsilon_dims[0],
                    md->epsilon_dims[1], md->epsilon_dims[2], eps_input_file);
    delete[] fname;
  }

  return md;
}

/******************************************************************************/
/* Material grid functions                                                    */
/******************************************************************************/
material_type make_material_grid(bool do_averaging, double beta, double eta) {
  material_data *md = new material_data();
  md->which_subclass = material_data::MATERIAL_GRID;
  md->do_averaging = do_averaging;
  md->beta = beta;
  md->eta = eta;
  return md;
}

void update_weights(material_type matgrid, double *weights) {
  size_t N = matgrid->grid_size.x * matgrid->grid_size.y * matgrid->grid_size.z;
  memcpy(matgrid->weights, weights, N * sizeof(double));
}

/******************************************************************************/
/* Helpers from  libctl/utils/geom.c                                          */
/******************************************************************************/
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

static void geom_box_intersection(geom_box *bi, const geom_box *b1, const geom_box *b2) {
  bi->low.x = MAX(b1->low.x, b2->low.x);
  bi->low.y = MAX(b1->low.y, b2->low.y);
  bi->low.z = MAX(b1->low.z, b2->low.z);
  bi->high.x = MIN(b1->high.x, b2->high.x);
  bi->high.y = MIN(b1->high.y, b2->high.y);
  bi->high.z = MIN(b1->high.z, b2->high.z);
}

static int geom_boxes_intersect(const geom_box *b1, const geom_box *b2) {
#define BETWEEN(x, low, high) ((x) >= (low) && (x) <= (high))
  /* true if the x, y, and z ranges all intersect. */
  return (
      (BETWEEN(b1->low.x, b2->low.x, b2->high.x) || BETWEEN(b1->high.x, b2->low.x, b2->high.x) ||
       BETWEEN(b2->low.x, b1->low.x, b1->high.x)) &&
      (BETWEEN(b1->low.y, b2->low.y, b2->high.y) || BETWEEN(b1->high.y, b2->low.y, b2->high.y) ||
       BETWEEN(b2->low.y, b1->low.y, b1->high.y)) &&
      (BETWEEN(b1->low.z, b2->low.z, b2->high.z) || BETWEEN(b1->high.z, b2->low.z, b2->high.z) ||
       BETWEEN(b2->low.z, b1->low.z, b1->high.z)));
}

/******************************************************************************/
/* Fragment Statistics                                                        */
/******************************************************************************/

double fragment_stats::tol = 0;
int fragment_stats::maxeval = 0;
int fragment_stats::resolution = 0;
meep::ndim fragment_stats::dims = meep::D1;
geometric_object_list fragment_stats::geom = {};
std::vector<dft_data> fragment_stats::dft_data_list;
std::vector<meep::volume> fragment_stats::pml_1d_vols;
std::vector<meep::volume> fragment_stats::pml_2d_vols;
std::vector<meep::volume> fragment_stats::pml_3d_vols;
std::vector<meep::volume> fragment_stats::absorber_vols;
material_type_list fragment_stats::extra_materials = material_type_list();
bool fragment_stats::split_chunks_evenly = false;
bool fragment_stats::eps_averaging = false;

static geom_box make_box_from_cell(vector3 cell_size) {
  double edgex = cell_size.x / 2;
  double edgey = cell_size.y / 2;
  double edgez = cell_size.z / 2;
  vector3 low = {-edgex, -edgey, -edgez};
  vector3 high = {edgex, edgey, edgez};
  geom_box result = {low, high};

  return result;
}

static size_t get_pixels_in_box(geom_box *b, int empty_pixel = 1) {
  int empty_x = b->low.x == b->high.x;
  int empty_y = b->low.y == b->high.y;
  int empty_z = b->low.z == b->high.z;

  double total_pixels =
      ((empty_x ? empty_pixel : (b->high.x - b->low.x) * fragment_stats::resolution) *
       (empty_y ? empty_pixel : (b->high.y - b->low.y) * fragment_stats::resolution) *
       (empty_z ? empty_pixel : (b->high.z - b->low.z) * fragment_stats::resolution));

  return (size_t)ceil(total_pixels);
}

static void center_box(geom_box *b) {
  b->low = vector3_plus(geometry_center, b->low);
  b->high = vector3_plus(geometry_center, b->high);
}

static fragment_stats init_stats(geom_box &box, double tol, int maxeval, meep::grid_volume *gv) {
  fragment_stats::tol = tol;
  fragment_stats::maxeval = maxeval;
  fragment_stats::resolution = gv->a;
  fragment_stats::dims = gv->dim;

  center_box(&box);
  fragment_stats result(box);
  return result;
}

fragment_stats compute_fragment_stats(
    geometric_object_list geom_, meep::grid_volume *gv, vector3 cell_size, vector3 cell_center,
    material_type default_mat, std::vector<dft_data> dft_data_list_,
    std::vector<meep::volume> pml_1d_vols_, std::vector<meep::volume> pml_2d_vols_,
    std::vector<meep::volume> pml_3d_vols_, std::vector<meep::volume> absorber_vols_,
    material_type_list extra_materials_, double tol, int maxeval, bool ensure_per,
    bool eps_averaging) {

  fragment_stats::geom = geom_;
  fragment_stats::dft_data_list = dft_data_list_;
  fragment_stats::pml_1d_vols = pml_1d_vols_;
  fragment_stats::pml_2d_vols = pml_2d_vols_;
  fragment_stats::pml_3d_vols = pml_3d_vols_;
  fragment_stats::absorber_vols = absorber_vols_;
  fragment_stats::extra_materials = extra_materials_;
  fragment_stats::eps_averaging = eps_averaging;

  init_libctl(default_mat, ensure_per, gv, cell_size, cell_center, &geom_);
  geom_box box = make_box_from_cell(cell_size);
  fragment_stats stats = init_stats(box, tol, maxeval, gv);
  stats.compute();
  return stats;
}

fragment_stats::fragment_stats(geom_box &bx)
    : num_anisotropic_eps_pixels(0), num_anisotropic_mu_pixels(0), num_nonlinear_pixels(0),
      num_susceptibility_pixels(0), num_nonzero_conductivity_pixels(0), num_1d_pml_pixels(0),
      num_2d_pml_pixels(0), num_3d_pml_pixels(0), num_dft_pixels(0), num_pixels_in_box(0), box(bx) {

  num_pixels_in_box = get_pixels_in_box(&bx);
}

void init_libctl(material_type default_mat, bool ensure_per, meep::grid_volume *gv,
                 vector3 cell_size, vector3 cell_center,
                 geometric_object_list *geom_) {
  geom_initialize();
  set_default_material(default_mat);
  ensure_periodicity = ensure_per;
  geometry_center = cell_center;
  dimensions = meep::number_of_directions(gv->dim);
  geometry_lattice.size = cell_size;
  geom_fix_object_list(*geom_);
}

bool fragment_stats::has_non_medium_material() {
  for (int i = 0; i < geom.num_items; ++i) {
    material_type mat = (material_type)geom.items[i].material;
    if (mat->which_subclass != material_data::MEDIUM) { return true; }
  }
  if (((material_type)default_material)->which_subclass != material_data::MEDIUM) { return true; }
  return false;
}

void fragment_stats::update_stats_from_material(material_type mat, size_t pixels,
                                                bool anisotropic_pixels_already_added) {
  switch (mat->which_subclass) {
    case material_data::MEDIUM: {
      medium_struct *med = &mat->medium;
      if (!anisotropic_pixels_already_added) { count_anisotropic_pixels(med, pixels); }
      count_nonlinear_pixels(med, pixels);
      count_susceptibility_pixels(med, pixels);
      count_nonzero_conductivity_pixels(med, pixels);
      break;
    }
    case material_data::MATERIAL_USER: {
      bool anisotropic_pixels_extmat_already_added = false;
      bool nonlinear_pixels_extmat_already_added = false;
      bool susceptibility_pixels_extmat_already_added = false;
      bool nonzero_conductivity_pixels_extmat_already_added = false;
      for (int i = 0; i < extra_materials.num_items; ++i) {
        medium_struct *med = &extra_materials.items[i]->medium;
        if (!anisotropic_pixels_already_added && !anisotropic_pixels_extmat_already_added) {
          anisotropic_pixels_extmat_already_added = count_anisotropic_pixels(med, pixels);
        }
        if (!nonlinear_pixels_extmat_already_added) {
          nonlinear_pixels_extmat_already_added = count_nonlinear_pixels(med, pixels);
        }
        if (!susceptibility_pixels_extmat_already_added) {
          susceptibility_pixels_extmat_already_added = count_susceptibility_pixels(med, pixels);
        }
        if (!nonzero_conductivity_pixels_extmat_already_added) {
          nonzero_conductivity_pixels_extmat_already_added =
              count_nonzero_conductivity_pixels(med, pixels);
        }
      }
      break;
    }
    default:
      // Only Medium and Material Function are supported
      return;
  }
}

void fragment_stats::compute_stats() {

  if (geom.num_items == 0) {
    // If there is no geometry, count the default material for the whole fragment
    update_stats_from_material((material_type)default_material, num_pixels_in_box);
  }

  for (int i = 0; i < geom.num_items; ++i) {
    geometric_object *go = &geom.items[i];
    // tolerance and max number of function evaluations of numerical quadrature are increased and decreased
    // from default values of 0.0001 and 100000, respectively, to obtain fast, approximate result
    double overlap = box_overlap_with_object(box, *go, 0.05 /* tol */, 1000 /* maxeval */);

    bool anisotropic_pixels_already_added = false;
    if (eps_averaging) {
      // If the object doesn't overlap the entire box, that implies that
      // an object interface intercepts the box, which means we treat
      // the entire box as anisotropic. This method could give some false
      // positives if there is another object with the same material behind
      // the current object, but in practice it is probably reasonable to
      // assume that there is a material interface somwhere in the box so
      // we won't worry about fancier edge-detection methods for now.
      if (overlap != 1.0) {
        anisotropic_pixels_already_added = true;
        num_anisotropic_eps_pixels += num_pixels_in_box;
        if (mu_not_1(go->material)) { num_anisotropic_mu_pixels += num_pixels_in_box; }
      }
    }

    // Count contributions from material of object
    size_t pixels = (size_t)ceil(overlap * num_pixels_in_box);
    if (pixels > 0) {
      material_type mat = (material_type)go->material;
      update_stats_from_material(mat, pixels, anisotropic_pixels_already_added);
    }

    // Count contributions from default_material
    size_t default_material_pixels = num_pixels_in_box - pixels;
    if (default_material_pixels > 0) {
      update_stats_from_material((material_type)default_material, default_material_pixels,
                                 anisotropic_pixels_already_added);
    }
  }
}

bool fragment_stats::count_anisotropic_pixels(medium_struct *med, size_t pixels) {
  size_t eps_offdiag_elements = 0;
  size_t mu_offdiag_elements = 0;

  if (med->epsilon_offdiag.x.re != 0) { eps_offdiag_elements++; }
  if (med->epsilon_offdiag.y.re != 0) { eps_offdiag_elements++; }
  if (med->epsilon_offdiag.z.re != 0) { eps_offdiag_elements++; }
  if (med->mu_offdiag.x.re != 0) { mu_offdiag_elements++; }
  if (med->mu_offdiag.y.re != 0) { mu_offdiag_elements++; }
  if (med->mu_offdiag.z.re != 0) { mu_offdiag_elements++; }

  num_anisotropic_eps_pixels += eps_offdiag_elements * pixels;
  num_anisotropic_mu_pixels += mu_offdiag_elements * pixels;
  return (eps_offdiag_elements != 0) || (mu_offdiag_elements != 0);
}

bool fragment_stats::count_nonlinear_pixels(medium_struct *med, size_t pixels) {
  size_t nonzero_chi_elements = 0;

  if (med->E_chi2_diag.x != 0) { nonzero_chi_elements++; }
  if (med->E_chi2_diag.y != 0) { nonzero_chi_elements++; }
  if (med->E_chi2_diag.z != 0) { nonzero_chi_elements++; }
  if (med->E_chi3_diag.x != 0) { nonzero_chi_elements++; }
  if (med->E_chi3_diag.y != 0) { nonzero_chi_elements++; }
  if (med->E_chi3_diag.z != 0) { nonzero_chi_elements++; }
  if (med->H_chi2_diag.x != 0) { nonzero_chi_elements++; }
  if (med->H_chi2_diag.y != 0) { nonzero_chi_elements++; }
  if (med->H_chi2_diag.z != 0) { nonzero_chi_elements++; }
  if (med->H_chi3_diag.x != 0) { nonzero_chi_elements++; }
  if (med->H_chi3_diag.y != 0) { nonzero_chi_elements++; }
  if (med->H_chi3_diag.z != 0) { nonzero_chi_elements++; }

  num_nonlinear_pixels += nonzero_chi_elements * pixels;
  return nonzero_chi_elements != 0;
}

bool fragment_stats::count_susceptibility_pixels(medium_struct *med, size_t pixels) {
  num_susceptibility_pixels += med->E_susceptibilities.size() * pixels;
  num_susceptibility_pixels += med->H_susceptibilities.size() * pixels;
  return (med->E_susceptibilities.size() != 0) || (med->H_susceptibilities.size() != 0);
}

bool fragment_stats::count_nonzero_conductivity_pixels(medium_struct *med, size_t pixels) {
  size_t nonzero_conductivity_elements = 0;

  if (med->D_conductivity_diag.x != 0) { nonzero_conductivity_elements++; }
  if (med->D_conductivity_diag.y != 0) { nonzero_conductivity_elements++; }
  if (med->D_conductivity_diag.z != 0) { nonzero_conductivity_elements++; }
  if (med->B_conductivity_diag.x != 0) { nonzero_conductivity_elements++; }
  if (med->B_conductivity_diag.y != 0) { nonzero_conductivity_elements++; }
  if (med->B_conductivity_diag.z != 0) { nonzero_conductivity_elements++; }

  num_nonzero_conductivity_pixels += nonzero_conductivity_elements * pixels;
  return nonzero_conductivity_elements != 0;
}

void fragment_stats::compute_dft_stats() {

  for (size_t i = 0; i < dft_data_list.size(); ++i) {
    for (size_t j = 0; j < dft_data_list[i].vols.size(); ++j) {
      geom_box dft_box = gv2box(dft_data_list[i].vols[j]);

      if (geom_boxes_intersect(&dft_box, &box)) {
        geom_box overlap_box;
        geom_box_intersection(&overlap_box, &dft_box, &box);

        // Note: Since geom_boxes_intersect returns true if two planes share a line or two volumes
        // share a line or plane, there are cases where some pixels are counted multiple times.
        size_t overlap_pixels = get_pixels_in_box(&overlap_box, 2);
        num_dft_pixels +=
            overlap_pixels * dft_data_list[i].num_freqs * dft_data_list[i].num_components;
      }
    }
  }
}

void fragment_stats::compute_pml_stats() {

  const std::vector<meep::volume> *pml_vols[] = {&pml_1d_vols, &pml_2d_vols, &pml_3d_vols};
  size_t *pml_pixels[] = {&num_1d_pml_pixels, &num_2d_pml_pixels, &num_3d_pml_pixels};

  for (int j = 0; j < 3; ++j) {
    for (size_t i = 0; i < pml_vols[j]->size(); ++i) {
      geom_box pml_box = gv2box((*pml_vols[j])[i]);

      if (geom_boxes_intersect(&pml_box, &box)) {
        geom_box overlap_box;
        geom_box_intersection(&overlap_box, &pml_box, &box);
        size_t overlap_pixels = get_pixels_in_box(&overlap_box, 1);
        *pml_pixels[j] += overlap_pixels;
      }
    }
  }
}

void fragment_stats::compute_absorber_stats() {

  for (size_t i = 0; i < absorber_vols.size(); ++i) {
    geom_box absorber_box = gv2box(absorber_vols[i]);

    if (geom_boxes_intersect(&absorber_box, &box)) {
      geom_box overlap_box;
      geom_box_intersection(&overlap_box, &absorber_box, &box);
      size_t overlap_pixels = get_pixels_in_box(&overlap_box, 1);
      num_nonzero_conductivity_pixels += overlap_pixels;
    }
  }
}

void fragment_stats::compute() {
  compute_stats();
  compute_dft_stats();
  compute_pml_stats();
  compute_absorber_stats();
}

// Return the estimated time in seconds this fragment will take to run
// based on a cost function obtained via linear regression on a dataset
// of random simulations.
double fragment_stats::cost() const {
  return (num_anisotropic_eps_pixels * 1.15061674e-04 + num_anisotropic_mu_pixels * 1.26843801e-04 +
          num_nonlinear_pixels * 1.67029547e-04 + num_susceptibility_pixels * 2.24790864e-04 +
          num_nonzero_conductivity_pixels * 4.61260934e-05 + num_dft_pixels * 1.47283950e-04 +
          num_1d_pml_pixels * 9.92955372e-05 + num_2d_pml_pixels * 1.36901107e-03 +
          num_3d_pml_pixels * 6.63939607e-04 + num_pixels_in_box * 3.46518274e-04);
}

void fragment_stats::print_stats() const {
  master_printf("Fragment stats\n");
  master_printf("  anisotropic_eps: %zu\n", num_anisotropic_eps_pixels);
  master_printf("  anisotropic_mu: %zu\n", num_anisotropic_mu_pixels);
  master_printf("  nonlinear: %zu\n", num_nonlinear_pixels);
  master_printf("  susceptibility: %zu\n", num_susceptibility_pixels);
  master_printf("  conductivity: %zu\n", num_nonzero_conductivity_pixels);
  master_printf("  pml_1d: %zu\n", num_1d_pml_pixels);
  master_printf("  pml_2d: %zu\n", num_2d_pml_pixels);
  master_printf("  pml_3d: %zu\n", num_3d_pml_pixels);
  master_printf("  dft: %zu\n", num_dft_pixels);
  master_printf("  pixels_in_box: %zu\n", num_pixels_in_box);
  master_printf("  box.low:  {%f, %f, %f}\n", box.low.x, box.low.y, box.low.z);
  master_printf("  box.high: {%f, %f, %f}\n\n", box.high.x, box.high.y, box.high.z);
}

dft_data::dft_data(int freqs, int components, std::vector<meep::volume> volumes)
    : num_freqs(freqs), num_components(components), vols(volumes) {}

/***************************************************************/
// Gradient calculation routines needed for material grid
/***************************************************************/

geom_box_tree calculate_tree(const meep::volume &v, geometric_object_list g) {
  geom_fix_object_list(g);
  geom_box box = gv2box(v);
  geom_box_tree geometry_tree = create_geom_box_tree0(g, box);
  return geometry_tree;
}

/* convenience routine to get element of material tensors */
static std::complex<double> cvec_to_value(vector3 diag, cvector3 offdiag, int idx) {
  std::complex<double> val = std::complex<double>(0, 0);
  switch (idx) {
    case 0:
      val = std::complex<double>(diag.x, 0);
      break;
    case 1:
      val = std::complex<double>(offdiag.x.re, offdiag.x.im);
      break;
    case 2:
      val = std::complex<double>(offdiag.y.re, offdiag.y.im);
      break;
    case 3:
      val = std::complex<double>(offdiag.x.re, -offdiag.x.im);
      break;
    case 4:
      val = std::complex<double>(diag.y, 0);
      break;
    case 5:
      val = std::complex<double>(offdiag.z.re, offdiag.z.im);
      break;
    case 6:
      val = std::complex<double>(offdiag.y.re, -offdiag.y.im);
      break;
    case 7:
      val = std::complex<double>(offdiag.z.re, -offdiag.z.im);
      break;
    case 8:
      val = std::complex<double>(diag.z, 0);
      break;
    default:
      meep::abort("Invalid value in switch statement.");
  }
  return val;
}

/* convenience routine to get element of material tensors */
double vec_to_value(vector3 diag, vector3 offdiag, int idx) {
  double val = 0.0;
  switch (idx) {
    case 0:
      val = diag.x;
      break;
    case 1:
      val = offdiag.x;
      break;
    case 2:
      val = offdiag.y;
      break;
    case 3:
      val = offdiag.x;
      break;
    case 4:
      val = diag.y;
      break;
    case 5:
      val = offdiag.z;
      break;
    case 6:
      val = offdiag.y;
      break;
    case 7:
      val = offdiag.z;
      break;
    case 8:
      val = diag.z;
      break;
    default:
      meep::abort("Invalid value in switch statement.");
  }
  return val;
}

void get_material_tensor(const medium_struct *mm, double freq,
                         std::complex<double> *tensor) {
  /*
  Note that the current implementation assumes that any dispersion
  is either a lorentzian or drude profile.
  */
  for (int i = 0; i < 9; i++) {
    std::complex<double> a = std::complex<double>(1, 0);
    std::complex<double> b = std::complex<double>(0, 0);
    // compute first part containing conductivity
    vector3 dummy;
    dummy.x = dummy.y = dummy.z = 0.0;
    double conductivityCur = vec_to_value(mm->D_conductivity_diag, dummy, i);
    a += std::complex<double>(0.0, conductivityCur / freq);

    // compute lorentzian component
    b = cvec_to_value(mm->epsilon_diag, mm->epsilon_offdiag, i);
    for (const auto &mm_susc: mm->E_susceptibilities) {
      meep::lorentzian_susceptibility sus = meep::lorentzian_susceptibility(
          mm_susc.frequency, mm_susc.gamma, mm_susc.drude);
      double sigma = vec_to_value(mm_susc.sigma_diag, mm_susc.sigma_offdiag, i);
      b += sus.chi1(freq, sigma);
    }

    // elementwise multiply
    tensor[i] = a * b;
  }
}

double get_material_gradient(
    double u,                       // material parameter at current point
    std::complex<double> fields_a,  // adjoint field at current point
    std::complex<double> fields_f,  // forward field at current point
    double freq,                    // frequency
    material_data *md,              // material
    meep::component field_dir,      // current field component
    double du                       // step size
) {
  /* Note that the current implementation assumes that
  no materials have off-diag components and that if a material
  has dispersion, it is either a lorentzian or drude profile.
  */

  const medium_struct *mm = &(md->medium);
  const medium_struct *m1 = &(md->medium_1);
  const medium_struct *m2 = &(md->medium_2);

  // trivial case
  if ((mm->E_susceptibilities.size() == 0) && mm->D_conductivity_diag.x == 0 &&
      mm->D_conductivity_diag.y == 0 && mm->D_conductivity_diag.z == 0){
        switch (field_dir){
          case meep::Ex: return 2 * (m2->epsilon_diag.x - m1->epsilon_diag.x) * (fields_a * fields_f).real();
          case meep::Ey: return 2 * (m2->epsilon_diag.y - m1->epsilon_diag.y) * (fields_a * fields_f).real();
          case meep::Ez: return 2 * (m2->epsilon_diag.z - m1->epsilon_diag.z) * (fields_a * fields_f).real();
          default: meep::abort("Invalid field component specified in gradient.");
        }
      }

  /* For now we do a finite difference approach to estimate the
  gradient of the system matrix `A` since it's fairly accurate,
  cheap, and easy to generalize. */
  std::complex<double> dA_du_0[9] = {std::complex<double>(0, 0)};
  epsilon_material_grid(md, u - du);
  get_material_tensor(mm, freq, dA_du_0);

  std::complex<double> dA_du_1[9] = {std::complex<double>(0, 0)};
  epsilon_material_grid(md, u + du);
  get_material_tensor(mm, freq, dA_du_1);

  std::complex<double> dA_du[9] = {std::complex<double>(0, 0)};
  for (int i = 0; i < 9; i++)
    dA_du[i] = (dA_du_1[i] - dA_du_0[i]) / (2 * du);

  int dir_idx = 0;
  if (field_dir == meep::Ex)
    dir_idx = 0;
  else if (field_dir == meep::Ey)
    dir_idx = 1;
  else if (field_dir == meep::Ez)
    dir_idx = 2;
  else
    meep::abort("Invalid adjoint field component");

  std::complex<double> result = fields_a * dA_du[3 * dir_idx + dir_idx] * fields_f;
  return 2 * result.real();
}

/* add the weights from linear_interpolate (see the linear_interpolate
   function in fields.cpp) to data ... this has to be changed if
   linear_interpolate is changed!! ...also multiply by scaleby
   etc. for different gradient types */

void add_interpolate_weights(double rx, double ry, double rz,
                             double *data, int nx, int ny, int nz, int stride,
                             double scaleby, const double *udata, int ukind, double uval) {
  int x1, y1, z1, x2, y2, z2;
  double dx, dy, dz, u;

  meep::map_coordinates(rx, ry, rz, nx, ny, nz,
                        x1, y1, z1, x2, y2, z2,
                        dx, dy, dz);

/* define a macro to give us data(x,y,z) on the grid,
in row-major order (the order used by HDF5): */
#define D(x, y, z) (data[(((x)*ny + (y)) * nz + (z)) * stride])
#define U(x, y, z) (udata[(((x)*ny + (y)) * nz + (z)) * stride])

  u = (((U(x1, y1, z1) * (1.0 - dx) + U(x2, y1, z1) * dx) * (1.0 - dy) +
        (U(x1, y2, z1) * (1.0 - dx) + U(x2, y2, z1) * dx) * dy) *
           (1.0 - dz) +
       ((U(x1, y1, z2) * (1.0 - dx) + U(x2, y1, z2) * dx) * (1.0 - dy) +
        (U(x1, y2, z2) * (1.0 - dx) + U(x2, y2, z2) * dx) * dy) *
           dz);

  if (ukind == material_data::U_MIN && u != uval) return; // TODO look into this
  if (ukind == material_data::U_PROD) scaleby *= uval / u;

  D(x1, y1, z1) += (1.0 - dx) * (1.0 - dy) * (1.0 - dz) * scaleby;
  D(x2, y1, z1) += dx * (1.0 - dy) * (1.0 - dz) * scaleby;
  D(x1, y2, z1) += (1.0 - dx) * dy * (1.0 - dz) * scaleby;
  D(x2, y2, z1) += dx * dy * (1.0 - dz) * scaleby;
  D(x1, y1, z2) += (1.0 - dx) * (1.0 - dy) * dz * scaleby;
  D(x2, y1, z2) += dx * (1.0 - dy) * dz * scaleby;
  D(x1, y2, z2) += (1.0 - dx) * dy * dz * scaleby;
  D(x2, y2, z2) += dx * dy * dz * scaleby;

#undef D
#undef U
}

void material_grids_addgradient_point(double *v, std::complex<double> fields_a,
                                      std::complex<double> fields_f, meep::component field_dir,
                                      vector3 p, double scalegrad, double freq,
                                      geom_box_tree geometry_tree) {
  geom_box_tree tp;
  int oi, ois;
  material_data *mg, *mg_sum;
  double uval;
  int kind;
  tp = geom_tree_search(p, geometry_tree, &oi);

  if (tp &&
      ((material_type)tp->objects[oi].o->material)->which_subclass == material_data::MATERIAL_GRID)
    mg = (material_data *)tp->objects[oi].o->material;
  else if (!tp && ((material_type)default_material)->which_subclass == material_data::MATERIAL_GRID)
    mg = (material_data *)default_material;
  else
    return; /* no material grids at this point */

  // Calculate the number of material grids if we are averaging values
  if ((tp) && ((kind = mg->material_grid_kinds) == material_data::U_MEAN)) {
    int matgrid_val_count = 0;
    geom_box_tree tp_sum;
    tp_sum = geom_tree_search(p, geometry_tree, &ois);
    mg_sum = (material_data *)tp_sum->objects[ois].o->material;
    do {
      tp_sum = geom_tree_search_next(p, tp_sum, &ois);
      ++matgrid_val_count;
      if (tp_sum) mg_sum = (material_data *)tp_sum->objects[ois].o->material;
    } while (tp_sum && is_material_grid(mg_sum));
    scalegrad /= matgrid_val_count;
  }
  else if ((tp) && ((mg->material_grid_kinds == material_data::U_MIN) ||
                    (mg->material_grid_kinds == material_data::U_PROD))) {
    meep::abort("%s:%i:material_grids_addgradient_point does not support overlapping MATERIAL_GRIDs with U_MIN or U_PROD.\n",__FILE__,__LINE__);
  }

  // Iterate through grids and add weights as needed
  if (tp) {
    /*NOTE in the future, it may be nice to be able to have *different*
    material grids in a particular design region. This would require checking each
    material grid here and iterating to the next spot in a large array of
    design parameters (see MPB).

    For now, it's actually easier just to assign each "unique" material grid its
    own design region. Unlike MPB, we are only iterating over the grid points inside
    that design region. Note that we can still have multiple copies of
    the same design grid (i.e. for symmetries). Thats why we are looping in this
    fashion. Since we aren't checking if each design grid is unique, however,
    it's up to the user to only have one unique design grid in this volume.*/
    vector3 sz = mg->grid_size;
    double *vcur = v;
    double *ucur = mg->weights;
    uval = matgrid_val(p, tp, oi, mg);
    do {
      vector3 pb = to_geom_box_coords(p, &tp->objects[oi]);
      add_interpolate_weights(
          pb.x, pb.y, pb.z, vcur, sz.x, sz.y, sz.z, 1,
          get_material_gradient(uval, fields_a, fields_f, freq, mg, field_dir, 1e-6) * scalegrad,
          ucur, kind, uval);
      if (kind == material_data::U_DEFAULT) break;
      tp = geom_tree_search_next(p, tp, &oi);
    } while (tp && is_material_grid((material_data *)tp->objects[oi].o->material));
  }
  // no object tree -- the whole domain is the material grid
  if (!tp && is_material_grid(default_material)) {
    map_lattice_coordinates(p.x,p.y,p.z);
    vector3 sz = mg->grid_size;
    double *vcur = v;
    double *ucur = mg->weights;
    uval = tanh_projection(material_grid_val(p, mg), mg->beta, mg->eta);
    add_interpolate_weights(p.x, p.y, p.z, vcur, sz.x, sz.y, sz.z, 1,
                            get_material_gradient(uval, fields_a, fields_f, freq, mg, field_dir) *
                                scalegrad,
                            ucur, kind, uval);
  }
}

void material_grids_addgradient(double *v, size_t ng, std::complex<double> *fields_a,
                                std::complex<double> *fields_f, double *frequencies,
                                size_t nf, double scalegrad, const meep::volume &where,
                                geom_box_tree geometry_tree, meep::fields *f) {
  int n2, n3, n4;
  double s[3][3], cen[3][3], c1, c2, c3, s1, s2, s3;
  vector3 p;

  // calculate cell dimensions
  meep::direction dirs[3];
  meep::vec min_max_loc[2] = {meep::vec(0,0,0),meep::vec(0,0,0)}; // extremal points in subgrid
  meep::component field_dir[3] = {meep::Ex, meep::Ey, meep::Ez};
  size_t dims[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  for (int c = 0; c < 3; c++) {

    f->get_array_slice_dimensions(where, &dims[3 * c], dirs, true, false, min_max_loc,
                                  0, field_dir[c]);

    vector3 max_corner = vec_to_vector3(min_max_loc[1]);
    double max_c_array[3] = {max_corner.x, max_corner.y, max_corner.z};
    vector3 min_corner = vec_to_vector3(min_max_loc[0]);
    double min_c_array[3] = {min_corner.x, min_corner.y, min_corner.z};
    for (int ci = 0; ci < 3; ci++) {
      s[c][ci] = (max_c_array[ci] - min_c_array[ci]) == 0 ? 0 : (max_c_array[ci] - min_c_array[ci]) / (dims[3 * c + ci] - 1);
      cen[c][ci] = dims[3 * c + ci] <= 1 ? 0 : min_c_array[ci];
    }
  }

  // Loop over component, x, y, z, and frequency dimensions
  // TODO speed up with MPI (if needed)
  int xyz_index = 0;
  for (int c = 0; c < 3; c++) {             // component
    n2 = dims[c * 3]; n3 = dims[c * 3 + 1]; n4 = dims[c * 3 + 2];
    c1 = cen[c][0]; c2 = cen[c][1]; c3 = cen[c][2];
    s1 = s[c][0]; s2 = s[c][1]; s3 = s[c][2];

    for (size_t i1 = 0; i1 < nf; ++i1) {       // freq
      for (int i2 = 0; i2 < n2; ++i2) {     // x
        for (int i3 = 0; i3 < n3; ++i3) {   // y
          for (int i4 = 0; i4 < n4; ++i4) { // z
            p.x = i2 * s1 + c1; p.y = i3 * s2 + c2; p.z = i4 * s3 + c3;
            material_grids_addgradient_point(v+ ng*i1, fields_a[xyz_index], fields_f[xyz_index], field_dir[c], p,
                                             scalegrad, frequencies[i1], geometry_tree);
            xyz_index++;
          }
        }
      }
    }
  }
}

static void find_array_min_max(int n, const double *data, double &min_val, double &max_val) {
  min_val = data[0];
  max_val = data[0];
  for (int i = 1; i < n; ++i) {
    if (data[i] < min_val)
      min_val = data[i];
    if (data[i] > max_val)
      max_val = data[i];
  }
  return;
}

void get_epsilon_grid(geometric_object_list gobj_list,
                      material_type_list mlist,
                      material_type _default_material,
                      bool _ensure_periodicity,
                      meep::grid_volume gv,
                      vector3 cell_size,
                      vector3 cell_center,
                      int nx, const double *x,
                      int ny, const double *y,
                      int nz, const double *z,
                      double *grid_vals) {
  double min_val[3], max_val[3];
  for (int n = 0; n < 3; ++n) {
    int ndir = (n == 0) ? nx : ((n == 1) ? ny : nz);
    if (ndir < 1) meep::abort("get_epsilon_grid: ndir < 1.");
    const double *adir = (n == 0) ? x : ((n == 1) ? y : z);
    find_array_min_max(ndir, adir, min_val[n], max_val[n]);
  }
  const meep::volume vol(meep::vec(min_val[0],min_val[1],min_val[2]),
                         meep::vec(max_val[0],max_val[1],max_val[2]));
  init_libctl(_default_material, _ensure_periodicity, &gv,
              cell_size, cell_center, &gobj_list);
  dim = gv.dim;
  geom_epsilon geps(gobj_list, mlist, vol);
  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j)
      for (int k = 0; k < nz; ++k)
        /* obtain the trace of the \varepsilon tensor for each
           grid point in row-major order (the order used by NumPy) */
        grid_vals[k + nz*(j + ny*i)] = geps.chi1p1(meep::E_stuff, meep::vec(x[i],y[j],z[k]));
}

} // namespace meep_geom
