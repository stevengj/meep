// -*- C++ -*-
/* These are functions for the libctl front-end which are exported
   via SWIG. */

#ifndef MEEP_CTL_SWIG_HPP
#define MEEP_CTL_SWIG_HPP 1

struct kpoint_list {
  meep::vec *kpoints;
  size_t n;
  meep::vec *kdom;
  size_t num_bands;
};

vector3 vec_to_vector3(const meep::vec &);
meep::vec vector3_to_vec(const vector3 v3);
void set_dimensions(int dims);

meep::structure *make_structure(int dims, vector3 size, vector3 center, double resolution,
                                bool enable_averaging, double subpixel_tol, int subpixel_maxeval,
                                bool ensure_periodicity_p, ctlio::geometric_object_list geometry,
                                ctlio::material_type_list extra_materials,
                                ctlio::material_type default_mat, const char *eps_input_file,
                                ctlio::pml_list pml_layers, ctlio::symmetry_list symmetries,
                                int num_chunks, double Courant, double global_D_conductivity_diag_,
                                double global_B_conductivity_diag_);

ctlio::cvector3_list do_harminv(ctlio::cnumber_list vals, double dt, double fmin, double fmax,
                                int maxbands, double spectral_density, double Q_thresh,
                                double rel_err_thresh, double err_thresh, double rel_amp_thresh,
                                double amp_thresh);

kpoint_list do_get_eigenmode_coefficients(meep::fields *f, meep::dft_flux flux,
                                          const meep::volume &eig_vol, int *bands, int num_bands,
                                          int parity, std::complex<double> *coeffs, double *vgrp,
                                          double eig_resolution, double eigensolver_tol,
                                          meep::kpoint_func user_kpoint_func,
                                          void *user_kpoint_data, int dir);

ctlio::number_list dft_flux_flux(meep::dft_flux *f);
ctlio::number_list dft_energy_electric(meep::dft_energy *f);
ctlio::number_list dft_energy_magnetic(meep::dft_energy *f);
ctlio::number_list dft_energy_total(meep::dft_energy *f);
ctlio::number_list dft_force_force(meep::dft_force *f);
ctlio::number_list dft_ldos_ldos(meep::dft_ldos *f);
ctlio::cnumber_list dft_ldos_F(meep::dft_ldos *f);
ctlio::cnumber_list dft_ldos_J(meep::dft_ldos *f);
ctlio::cnumber_list dft_near2far_farfield(meep::dft_near2far *f, const meep::vec &x);
ctlio::number_list dft_near2far_flux(meep::dft_near2far *f, meep::direction df,
                                     const meep::volume &where, double resolution);

ctlio::cnumber_list make_casimir_g(double T, double dt, double sigma, meep::field_type ft,
                                   std::complex<double> (*eps_func)(std::complex<double> omega) = 0,
                                   double Tfft = 0);

ctlio::cnumber_list make_casimir_g_kz(double T, double dt, double sigma, meep::field_type ft);

ctlio::integer_list GDSII_layers(const char *fname);
meep::volume GDSII_vol(const char *fname, int layer = -1, double zmin = 0, double zmax = 0);
SCM get_GDSII_prism_data(const char *GDSIIFile, int Layer = -1, double zmin = 0.0, double zmax = 0.0);

// wrapper around constructor to fool SWIG
meep::volume_list *make_volume_list(const meep::volume &v, int c, std::complex<double> weight,
                                    meep::volume_list *next);

ctlio::number_list std_vector_double_to_scm(std::vector<double> *v);

#endif // MEEP_CTL_SWIG_HPP
