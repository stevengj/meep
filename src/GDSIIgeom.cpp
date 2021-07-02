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

/***************************************************************/
/* GDSII.cpp  -- libmeepgeom code to support meep geometry     */
/*            -- definitions from GDSII files                  */
/* homer reid -- 5/2018                                        */
/***************************************************************/

#include <vector>
#include <string>
#include "meepgeom.hpp"

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef HAVE_LIBGDSII
#include <libGDSII.h>
#endif

namespace meep_geom {

#ifdef HAVE_LIBGDSII

bool with_libGDSII() { return true; }

void get_polygon_bounding_box(dVec vertex_coordinates, meep::vec &max_corner,
                              meep::vec &min_corner) {
  double xmax = vertex_coordinates[2 * 0 + 0], xmin = xmax;
  double ymax = vertex_coordinates[2 * 0 + 1], ymin = ymax;
  for (size_t nv = 0; nv < vertex_coordinates.size() / 2; nv++) {
    double x = vertex_coordinates[2 * nv + 0], y = vertex_coordinates[2 * nv + 1];
    xmax = fmax(xmax, x);
    ymax = fmax(ymax, y);
    xmin = fmin(xmin, x);
    ymin = fmin(ymin, y);
  }
  max_corner.set_direction(meep::X, xmax);
  max_corner.set_direction(meep::Y, ymax);
  min_corner.set_direction(meep::X, xmin);
  min_corner.set_direction(meep::Y, ymin);
}

void get_polygon_center_size(dVec vertex_coordinates, meep::vec &center, meep::vec &size) {
  meep::vec max_corner, min_corner;
  get_polygon_bounding_box(vertex_coordinates, max_corner, min_corner);

  center.set_direction(meep::X,
                       0.5 * (max_corner.in_direction(meep::X) + min_corner.in_direction(meep::X)));
  center.set_direction(meep::Y,
                       0.5 * (max_corner.in_direction(meep::Y) + min_corner.in_direction(meep::Y)));
  center.set_direction(meep::Z, 0.0);

  size.set_direction(meep::X, max_corner.in_direction(meep::X) - min_corner.in_direction(meep::X));
  size.set_direction(meep::Y, max_corner.in_direction(meep::Y) - min_corner.in_direction(meep::Y));
  size.set_direction(meep::Z, 0.0);
}

/*******************************************************************/
// Search the geometry for a polygon on a given layer containing   */
// (the reference point of) a given text label.                    */
// If Text==NULL, find any polygon on the given layer.             */
// If Layer==-1, search all layers.                                */
// If multiple matching polygons are found, choose one arbitrarily.*/
/*******************************************************************/
dVec get_polygon(const char *GDSIIFile, const char *Text, int Layer = -1) {
  PolygonList polygons = libGDSII::GetPolygons(GDSIIFile, Text, Layer);

  char Description[100];
  if (Text)
    snprintf(Description, 100, "with label %s", Text);
  else
    snprintf(Description, 100, "on layer %i", Layer);

  if (polygons.size() == 0) meep::abort("%s: found no polygons %s", GDSIIFile, Description);
  if (polygons.size() > 1)
    fprintf(stderr, "warning: %s: found multiple polygons %s (choosing arbitrarily)\n", GDSIIFile,
            Description);

  return polygons[0];
}

/*******************************************************************/
/* find a polygon on the given GDSII layer and set the libctlgeom  */
/* geometry to the size of its bounding box.                       */
/* if Text is non-null, only polygons containing the reference     */
/* point of a GDSII text element with content Text will be         */
/* considered.                                                     */
/*******************************************************************/
meep::grid_volume set_geometry_from_GDSII(double resolution, const char *GDSIIFile,
                                          const char *Text, int Layer, double zsize) {
  dVec polygon = get_polygon(GDSIIFile, Text, Layer);

  meep::vec center, size;
  get_polygon_center_size(polygon, center, size);

  geometry_lattice.size.x = size.in_direction(meep::X);
  geometry_lattice.size.y = size.in_direction(meep::Y);
  geometry_lattice.size.z = zsize;
  meep::grid_volume gv =
      zsize == 0.0
          ? meep::vol2d(geometry_lattice.size.x, geometry_lattice.size.y, resolution)
          : meep::vol3d(geometry_lattice.size.x, geometry_lattice.size.y, zsize, resolution);
  gv.center_origin();
  return gv;
}

meep::grid_volume set_geometry_from_GDSII(double resolution, const char *GDSIIFile, int Layer,
                                          double zsize) {
  return set_geometry_from_GDSII(resolution, GDSIIFile, 0, Layer, zsize);
}

/*******************************************************************/
/* find all polygons on a given GDSII layer and return a list of   */
/* geometric_objects describing prisms, all with the same material */
/* and thickness.                                                  */
/*******************************************************************/
geometric_object_list get_GDSII_prisms(material_type material, const char *GDSIIFile, int Layer,
                                       double zmin, double zmax) {
  geometric_object_list prisms = {0, 0};

  // fetch all polygons on the given GDSII layer
  PolygonList polygons = libGDSII::GetPolygons(GDSIIFile, Layer);
  int num_prisms = polygons.size();
  if (num_prisms == 0) return prisms; // no polygons found; TODO: print warning?

  // create a prism for each polygon in the list
  prisms.num_items = num_prisms;
  prisms.items = new geometric_object[num_prisms];
  for (int np = 0; np < num_prisms; np++) {
    dVec polygon = polygons[np];
    int num_vertices = polygon.size() / 2;
    vector3 *vertices = new vector3[num_vertices];
    for (int nv = 0; nv < num_vertices; nv++) {
      vertices[nv].x = polygon[2 * nv + 0];
      vertices[nv].y = polygon[2 * nv + 1];
      vertices[nv].z = zmin;
    }
    double height = zmax - zmin;
    vector3 zaxis = {0, 0, 1};
    prisms.items[np] = make_prism(material, vertices, num_vertices, height, zaxis);
    delete[] vertices; // make_prism copies the data
  }
  return prisms;
}

/*******************************************************************/
/* like the previous routine, but creates only a single prism,     */
/* optionally identified by Text; if non-null, only polygons       */
/* containing the reference point of a GDSII text string with      */
/* content Text will be considered. if there are still multiple    */
/* choices of polygon, one will be chosen arbitrarily.             */
/*******************************************************************/
geometric_object get_GDSII_prism(material_type material, const char *GDSIIFile, const char *Text,
                                 int Layer, double zmin, double zmax) {
  dVec polygon = get_polygon(GDSIIFile, Text, Layer);

  int num_vertices = polygon.size() / 2;
  vector3 *vertices = new vector3[num_vertices];
  for (int nv = 0; nv < num_vertices; nv++) {
    vertices[nv].x = polygon[2 * nv + 0];
    vertices[nv].y = polygon[2 * nv + 1];
    vertices[nv].z = zmin;
  }

  double height = zmax - zmin;
  vector3 zaxis = {0, 0, 1};
  return make_prism(material, vertices, num_vertices, height, zaxis);
}

geometric_object get_GDSII_prism(material_type material, const char *GDSIIFile, int Layer,
                                 double zmin, double zmax) {
  return get_GDSII_prism(material, GDSIIFile, 0, Layer, zmin, zmax);
}

/*******************************************************************/
/* create a meep::volume from a GDSII polygon and optional z-size; */
/* useful for defining flux regions, source volumes ,etc.          */
/*******************************************************************/
meep::volume get_GDSII_volume(const char *GDSIIFile, const char *Text, int Layer, double zmin,
                              double zmax) {
  dVec polygon = get_polygon(GDSIIFile, Text, Layer);
  meep::ndim di = ((((float)zmin) == 0.0 && ((float)zmax) == 0.0) ? meep::D2 : meep::D3);
  meep::vec max_corner = meep::zero_vec(di), min_corner = meep::zero_vec(di);
  get_polygon_bounding_box(polygon, max_corner, min_corner);
  max_corner.set_direction(meep::Z, zmax);
  min_corner.set_direction(meep::Z, zmin);
  return meep::volume(max_corner, min_corner);
}

meep::volume get_GDSII_volume(const char *GDSIIFile, int Layer, double zmin, double zmax) {
  return get_GDSII_volume(GDSIIFile, 0, Layer, zmin, zmax);
}

/***************************************************************/
/* stubs for compilation without libGDSII **********************/
/***************************************************************/
#else // HAVE_LIBGDSII

bool with_libGDSII() { return false; }

void GDSIIError(const char *Routine) {
  meep::abort("Meep must be configured/compiled with libGDSII for %s", Routine);
}

meep::grid_volume set_geometry_from_GDSII(double resolution, const char *GDSIIFile,
                                          const char *Text, int Layer, double zsize) {
  (void)resolution;
  (void)GDSIIFile;
  (void)Text;
  (void)Layer;
  (void)zsize;
  GDSIIError("set_geometry_from_GDSII");
  return meep::grid_volume();
}
meep::grid_volume set_geometry_from_GDSII(double resolution, const char *GDSIIFile, int Layer,
                                          double zsize) {
  (void)resolution;
  (void)GDSIIFile;
  (void)Layer;
  (void)zsize;
  GDSIIError("set_geometry_from_GDSII");
  return meep::grid_volume();
}

geometric_object_list get_GDSII_prisms(material_type material, const char *GDSIIFile, int Layer,
                                       double zmin, double zmax) {
  (void)material;
  (void)GDSIIFile;
  (void)Layer;
  (void)zmin;
  (void)zmax;
  GDSIIError("get_GDSII_prisms");
  geometric_object_list prisms = {0, 0};
  return prisms;
}

geometric_object get_GDSII_prism(material_type material, const char *GDSIIFile, const char *Text,
                                 int Layer, double zmin, double zmax) {
  (void)material;
  (void)GDSIIFile;
  (void)Text;
  (void)Layer;
  (void)zmin;
  (void)zmax;
  GDSIIError("get_GDSII_prism");
  vector3 center = {0.0, 0.0, 0.0};
  return make_sphere(0, center, 0.0);
}
geometric_object get_GDSII_prism(material_type material, const char *GDSIIFile, int Layer,
                                 double zmin, double zmax) {
  (void)material;
  (void)GDSIIFile;
  (void)Layer;
  (void)zmin;
  (void)zmax;
  GDSIIError("get_GDSII_prism");
  vector3 center = {0.0, 0.0, 0.0};
  return make_sphere(0, center, 0.0);
}
meep::volume get_GDSII_volume(const char *GDSIIFile, const char *Text, int Layer, double zmin,
                              double zmax) {
  (void)GDSIIFile;
  (void)Text;
  (void)Layer;
  (void)zmin;
  (void)zmax;
  GDSIIError("get_GDSII_volume");
  return meep::volume(meep::vec());
}
meep::volume get_GDSII_volume(const char *GDSIIFile, int Layer, double zmin, double zmax) {
  (void)GDSIIFile;
  (void)Layer;
  (void)zmin;
  (void)zmax;
  GDSIIError("get_GDSII_volume");
  return meep::volume(meep::vec());
}

#endif // HAVE_LIBGDSII

std::vector<int> get_GDSII_layers(const char *GDSIIFile) {
#if defined(HAVE_LIBGDSII) && defined(HAVE_GDSII_GETLAYERS)
  return libGDSII::GetLayers(GDSIIFile);
#else
  (void)GDSIIFile;
  meep::abort(
      "get_GDSII_layers needs Meep to be configured/compiled with libGDSII version 0.21 or later");
  std::vector<int> layers;
  return layers;
#endif
}

} // namespace meep_geom
