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
#ifndef MEEP_H
#define MEEP_H

#include <stdio.h>
#include <stddef.h>
#include <math.h>

#include "meep/vec.hpp"
#include "meep/mympi.hpp"

#include <vector>

namespace meep {

/* The (time-domain) fields arrays of the fields_chunk class as well
   as the material arrays of the structure_chunk class chi1inv,
   chi3, sigma, etc. can be stored using single-precision floating
   point rather than double precision (the default). The reduced
   precision can provide for up to a factor of 2X improvement in the
   time-stepping rate with generally negligible loss in accuracy. */
#define MEEP_SINGLE 0 // 1 for single precision, 0 for double
#if MEEP_SINGLE
typedef float realnum;
#else
typedef double realnum;
#endif

#define MEEP_MIN_OUTPUT_TIME 4.0 // output no more often than this many seconds

extern int
    verbosity; // if 0, suppress all non-error messages from Meep; 1 is default, 2 is debug output

const double pi = 3.141592653589793238462643383276;

const double infinity = HUGE_VAL;

#ifdef NAN
const double nan = NAN;
#else
const double nan = -7.0415659787563146e103; // ideally, a value never encountered in practice
#endif

class h5file;

// Defined in monitor.cpp
void matrix_invert(std::complex<double> (&Vinv)[9], std::complex<double> (&V)[9]);

// Defined in dft.cpp
std::vector<double> linspace(double freq_min, double freq_max, size_t Nfreq);

double pml_quadratic_profile(double, void *);

/* generic base class, only used by subclassing: represents susceptibility
   polarizability vector P = chi(omega) W  (where W = E or H). */
template<class T>
class susceptibility {
public:
  susceptibility() {
    id = cur_id++;
    ntot = 0;
    next = NULL;
    FOR_COMPONENTS(c) FOR_DIRECTIONS(d) {
      sigma[c][d] = NULL;
      trivial_sigma[c][d] = true;
    }
  }
  susceptibility(const susceptibility &s) {
    id = s.id;
    ntot = s.ntot;
    next = NULL;
    FOR_COMPONENTS(c) FOR_DIRECTIONS(d) {
      sigma[c][d] = NULL;
      trivial_sigma[c][d] = true;
    }
  }
  virtual susceptibility *clone() const;
  virtual ~susceptibility() {
    FOR_COMPONENTS(c) FOR_DIRECTIONS(d) { delete[] sigma[c][d]; }
    delete next;
  }

  int get_id() const { return id; }
  bool operator==(const susceptibility &s) const { return id == s.id; };

  // Returns the 1st order nonlinear susceptibility (generic)
  virtual std::complex<T> chi1(T freq, T sigma = 1);

  // update all of the internal polarization state given the W field
  // at the current time step, possibly the previous field W_prev, etc.
  virtual void update_P(T *W[NUM_FIELD_COMPONENTS][2],
                        T *W_prev[NUM_FIELD_COMPONENTS][2], T dt, const grid_volume &gv,
                        void *P_internal_data) const {
    (void)P;
    (void)W;
    (void)W_prev;
    (void)dt;
    (void)gv;
    (void)P_internal_data; // avoid warnings for unused params
  }

  // subtract all of the internal polarizations from the given f_minus_p
  // field.  Also given the fields array if it is needed for some reason.
  // Only update for ft fields.
  virtual void subtract_P(field_type ft, T *f_minus_p[NUM_FIELD_COMPONENTS][2],
                          void *P_internal_data) const {
    (void)ft;
    (void)f_minus_p;
    (void)P_internal_data;
  }

  // whether, for the given field W, Meep needs to allocate P[c]
  virtual bool needs_P(component c, int cmp, T *W[NUM_FIELD_COMPONENTS][2]) const;

  // whether update_P will need the notowned part of W for this c
  // (which means that Meep will need to communicate it between chunks)
  virtual bool needs_W_notowned(component c, T *W[NUM_FIELD_COMPONENTS][2]) const;

  // whether update_P needs the W_prev field (from the previous timestep)
  virtual bool needs_W_prev() const { return false; }

  /* A susceptibility may be associated with any amount of internal
     data needed to update the polarization field.  This includes the
     polarization field(s) itself.  It may also, for example, store
     the polarization field from previous timesteps, atomic-level
     populations, or other data.  These routines return the size of
     this internal-data array and initialize it. */
  virtual void *new_internal_data(T *W[NUM_FIELD_COMPONENTS][2],
                                  const grid_volume &gv) const {
    (void)W;
    (void)gv;
    return 0;
  }
  virtual void delete_internal_data(void *data) const;
  virtual void init_internal_data(T *W[NUM_FIELD_COMPONENTS][2], T dt,
                                  const grid_volume &gv, void *data) const {
    (void)W;
    (void)dt;
    (void)gv;
    (void)data;
  }
  virtual void *copy_internal_data(void *data) const {
    (void)data;
    return 0;
  }

  /* The following methods are used in boundaries.cpp to set up any
     extra communications that may be necessary at chunk boundaries
     for the internal data of a susceptibility's polarization
     state. */

  /* the number of notowned fields/data in the internal data that
     are needed by update_P for the c Yee grid (note: we assume that we only
     have internal data for c's where we have external polarizations) */
  virtual int num_internal_notowned_needed(component c, void *P_internal_data) const {
    (void)c;
    (void)P_internal_data;
    return 0;
  }
  /* the offset into the internal data of the n'th Yee-grid point in
     the c Yee grid for the inotowned internal field, where
     0 <= inotowned < size_internal_notowned_needed. */
  virtual T *internal_notowned_ptr(int inotowned, component c, int n,
                                   void *P_internal_data) const {
    (void)inotowned;
    (void)n;
    (void)c;
    (void)P_internal_data;
    return 0;
  }

  /* same thing as above, except this gives (possibly complex)
     internal fields that need to be multiplied by the same phase
     factor as the fields at boundaries.  Note: we assume internal fields
     are complex if and only if !is_real (i.e. if EM fields are complex) */
  virtual int num_cinternal_notowned_needed(component c, void *P_internal_data) const {
    (void)c;
    (void)P_internal_data;
    return 0;
  }
  // real/imaginary parts offsets for cmp = 0/1
  virtual T *cinternal_notowned_ptr(int inotowned, component c, int cmp, int n,
                                    void *P_internal_data) const {
    (void)inotowned;
    (void)n;
    (void)c;
    (void)cmp;
    (void)P_internal_data;
    return 0;
  }

  virtual void dump_params(h5file *h5f, size_t *start) {
    (void)h5f;
    (void)start;
  }
  virtual int get_num_params() { return 0; }
  // This should only be used when dumping and loading susceptibility data to hdf5
  void set_id(int new_id) { id = new_id; };

  susceptibility<T> *next;
  size_t ntot;
  T *sigma[NUM_FIELD_COMPONENTS][5];

  /* trivial_sigma[c][d] is true only if *none* of the processes has a
     nontrivial sigma (c,d) component.  This differs, from sigma,
     which is non-NULL only if *this* process needs a nontrivial sigma
     (c,d).  Coordinated between processes at add_susceptibility, no
     communication elsewhere.  (We need this for boundary
     communcations between chunks, where one chunk might have sigma ==
     0 and the other != 0.) */
  bool trivial_sigma[NUM_FIELD_COMPONENTS][5];

private:
  static int cur_id; // unique id to assign to next susceptibility object
  int id;            // id for this object and its clones, for comparison purposes
};

/* a Lorentzian susceptibility
   \chi(\omega) = sigma * omega_0^2 / (\omega_0^2 - \omega^2 - i\gamma \omega)
  If no_omega_0_denominator is true, then we omit the omega_0^2 factor in the
  denominator to obtain a Drude model. */
template<class T>
class lorentzian_susceptibility : public susceptibility<T> {
public:
  lorentzian_susceptibility(T omega_0, T gamma, bool no_omega_0_denominator = false)
      : omega_0(omega_0), gamma(gamma), no_omega_0_denominator(no_omega_0_denominator) {}
  virtual susceptibility<T> *clone() const { return new lorentzian_susceptibility(*this); }
  virtual ~lorentzian_susceptibility() {}

  // Returns the 1st order nonlinear susceptibility
  virtual std::complex<T> chi1(T freq, T sigma = 1);

  virtual void update_P(T *W[NUM_FIELD_COMPONENTS][2],
                        T *W_prev[NUM_FIELD_COMPONENTS][2], T dt, const grid_volume &gv,
                        void *P_internal_data) const;

  virtual void subtract_P(field_type ft, T *f_minus_p[NUM_FIELD_COMPONENTS][2],
                          void *P_internal_data) const;

  virtual void *new_internal_data(T *W[NUM_FIELD_COMPONENTS][2], const grid_volume &gv) const;
  virtual void init_internal_data(T *W[NUM_FIELD_COMPONENTS][2], T dt,
                                  const grid_volume &gv, void *data) const;
  virtual void *copy_internal_data(void *data) const;

  virtual int num_cinternal_notowned_needed(component c, void *P_internal_data) const;
  virtual T *cinternal_notowned_ptr(int inotowned, component c, int cmp, int n,
                                    void *P_internal_data) const;

  virtual void dump_params(h5file *h5f, size_t *start);
  virtual int get_num_params() { return 4; }

protected:
  T omega_0, gamma;
  bool no_omega_0_denominator;
};

/* like a Lorentzian susceptibility, but the polarization equation
   includes white noise with a specified amplitude */
template<class T>
class noisy_lorentzian_susceptibility : public lorentzian_susceptibility<T> {
public:
  noisy_lorentzian_susceptibility(T noise_amp, T omega_0, T gamma,
                                  bool no_omega_0_denominator = false)
    : lorentzian_susceptibility<T>::lorentzian_susceptibility(omega_0, gamma, no_omega_0_denominator), noise_amp(noise_amp) {}

  virtual susceptibility<T> *clone() const { return new noisy_lorentzian_susceptibility(*this); }

  virtual void update_P(T *W[NUM_FIELD_COMPONENTS][2],
                        T *W_prev[NUM_FIELD_COMPONENTS][2], T dt, const grid_volume &gv,
                        void *P_internal_data) const;

  virtual void dump_params(h5file *h5f, size_t *start);
  virtual int get_num_params() { return 5; }

protected:
  T noise_amp;
};

typedef enum { GYROTROPIC_LORENTZIAN, GYROTROPIC_DRUDE, GYROTROPIC_SATURATED } gyrotropy_model;

/* gyrotropic susceptibility */
template<class T>
class gyrotropic_susceptibility : public susceptibility<T> {
public:
  gyrotropic_susceptibility(const vec &bias, T omega_0, T gamma, T alpha = 0.0,
                            gyrotropy_model model = GYROTROPIC_LORENTZIAN);
  virtual susceptibility<T> *clone() const { return new gyrotropic_susceptibility(*this); }

  virtual void *new_internal_data(T *W[NUM_FIELD_COMPONENTS][2], const grid_volume &gv) const;
  virtual void init_internal_data(T *W[NUM_FIELD_COMPONENTS][2], T dt,
                                  const grid_volume &gv, void *data) const;
  virtual void *copy_internal_data(void *data) const;

  virtual bool needs_P(component c, int cmp, T *W[NUM_FIELD_COMPONENTS][2]) const;
  virtual void update_P(T *W[NUM_FIELD_COMPONENTS][2],
                        T *W_prev[NUM_FIELD_COMPONENTS][2], T dt, const grid_volume &gv,
                        void *P_internal_data) const;
  virtual void subtract_P(field_type ft, T *f_minus_p[NUM_FIELD_COMPONENTS][2],
                          void *P_internal_data) const;

  virtual int num_cinternal_notowned_needed(component c, void *P_internal_data) const;
  virtual T *cinternal_notowned_ptr(int inotowned, component c, int cmp, int n,
                                    void *P_internal_data) const;

  virtual void dump_params(h5file *h5f, size_t *start);
  virtual int get_num_params() { return 8; }
  virtual bool needs_W_notowned(component c, T *W[NUM_FIELD_COMPONENTS][2]) const {
    (void)c;
    (void)W;
    return true;
  }

protected:
  T gyro_tensor[3][3];
  T omega_0, gamma, alpha;
  gyrotropy_model model;
};

class multilevel_susceptibility : public susceptibility<double> {
public:
  multilevel_susceptibility() : L(0), T(0), Gamma(0), N0(0), alpha(0), omega(0), gamma(0) {}
  multilevel_susceptibility(int L, int T, const realnum *Gamma, const realnum *N0,
                            const realnum *alpha, const realnum *omega, const realnum *gamma,
                            const realnum *sigmat);
  multilevel_susceptibility(const multilevel_susceptibility &from);
  virtual susceptibility *clone() const { return new multilevel_susceptibility(*this); }
  virtual ~multilevel_susceptibility();

  virtual void update_P(realnum *W[NUM_FIELD_COMPONENTS][2],
                        realnum *W_prev[NUM_FIELD_COMPONENTS][2], realnum dt, const grid_volume &gv,
                        void *P_internal_data) const;

  virtual void subtract_P(field_type ft, realnum *f_minus_p[NUM_FIELD_COMPONENTS][2],
                          void *P_internal_data) const;

  virtual void *new_internal_data(realnum *W[NUM_FIELD_COMPONENTS][2], const grid_volume &gv) const;
  virtual void init_internal_data(realnum *W[NUM_FIELD_COMPONENTS][2], realnum dt,
                                  const grid_volume &gv, void *data) const;
  virtual void *copy_internal_data(void *data) const;
  virtual void delete_internal_data(void *data) const;

  virtual int num_cinternal_notowned_needed(component c, void *P_internal_data) const;
  virtual realnum *cinternal_notowned_ptr(int inotowned, component c, int cmp, int n,
                                          void *P_internal_data) const;

  // always need notowned W and W_prev for E dot dP/dt terms
  virtual bool needs_W_notowned(component c, realnum *W[NUM_FIELD_COMPONENTS][2]) const {
    (void)c;
    (void)W;
    return true;
  }
  virtual bool needs_W_prev() const { return true; }

protected:
  int L;           // number of atom levels
  int T;           // number of optical transitions
  realnum *Gamma;  // LxL matrix of relaxation rates Gamma[i*L+j] from i -> j
  realnum *N0;     // L initial populations
  realnum *alpha;  // LxT matrix of transition coefficients 1/omega
  realnum *omega;  // T transition frequencies
  realnum *gamma;  // T optical loss rates
  realnum *sigmat; // 5*T transition-specific sigma-diagonal factors
};

// h5file.cpp: HDF5 file I/O.  Most users, if they use this
// class at all, will only use the constructor to open the file, and
// will otherwise use the fields::output_hdf5 functions.
class h5file {
public:
  typedef enum { READONLY, READWRITE, WRITE } access_mode;

  h5file(const char *filename_, access_mode m = READWRITE, bool parallel_ = true);
  ~h5file(); // closes the files (and any open dataset)

  bool ok();

  void *read(const char *dataname, int *rank, size_t *dims, int maxrank,
             bool single_precision = true);
  void write(const char *dataname, int rank, const size_t *dims, void *data,
             bool single_precision = true);
  char *read(const char *dataname);
  void write(const char *dataname, const char *data);

  void create_data(const char *dataname, int rank, const size_t *dims, bool append_data = false,
                   bool single_precision = true);
  void extend_data(const char *dataname, int rank, const size_t *dims);
  void create_or_extend_data(const char *dataname, int rank, const size_t *dims, bool append_data,
                             bool single_precision = true);
  void write_chunk(int rank, const size_t *chunk_start, const size_t *chunk_dims, float *data);
  void write_chunk(int rank, const size_t *chunk_start, const size_t *chunk_dims, double *data);
  void write_chunk(int rank, const size_t *chunk_start, const size_t *chunk_dims, size_t *data);
  void done_writing_chunks();

  void read_size(const char *dataname, int *rank, size_t *dims, int maxrank);
  void read_chunk(int rank, const size_t *chunk_start, const size_t *chunk_dims, float *data);
  void read_chunk(int rank, const size_t *chunk_start, const size_t *chunk_dims, double *data);
  void read_chunk(int rank, const size_t *chunk_start, const size_t *chunk_dims, size_t *data);

  void remove();
  void remove_data(const char *dataname);

  const char *file_name() const { return filename; }

  void prevent_deadlock(); // hackery for exclusive mode
  bool dataset_exists(const char *name);

private:
  access_mode mode;
  char *filename;
  bool parallel;

  bool is_cur(const char *dataname);
  void unset_cur();
  void set_cur(const char *dataname, void *data_id);
  char *cur_dataname;

  /* store hid_t values as hid_t* cast to void*, so that
     files including meep.h don't need hdf5.h */
  void *id;     /* file */
  void *cur_id; /* dataset, if any */

  void *get_id(); // get current (file) id, opening/creating file if needed
  void close_id();

public:
  /* linked list to keep track of which datasets we are extending...
     this is necessary so that create_or_extend_data can know whether
     to create (overwrite) a dataset or extend it. */
  struct extending_s {
    int dindex;
    char *dataname;
    struct extending_s *next;
  } * extending;
  extending_s *get_extending(const char *dataname) const;
};

typedef double (*pml_profile_func)(double u, void *func_data);

#define DEFAULT_SUBPIXEL_TOL 1e-4
#define DEFAULT_SUBPIXEL_MAXEVAL 100000

/* This class is used to compute position-dependent material properties
   like the dielectric function, permeability (mu), polarizability sigma,
   nonlinearities, et cetera.  Simple cases of stateless functions are
   handled by canned subclasses below, but more complicated cases
   can be handled by creating a user-defined subclass of material_function.
   It is useful to group different properties into one class because
   it is likely that complicated implementations will share state between
   properties. */
class material_function {
  material_function(const material_function &ef) { (void)ef; } // prevent copying
public:
  material_function() {}
  virtual ~material_function() {}

  /* Specify a restricted grid_volume: all subsequent eps/sigma/etc
     calls will be for points inside v, until the next set_volume. */
  virtual void set_volume(const volume &v) { (void)v; }
  virtual void unset_volume(void) {} // unrestrict the grid_volume

  virtual double chi1p1(field_type ft, const vec &r) {
    (void)ft;
    (void)r;
    return 1.0;
  }

  /* scalar dielectric function */
  virtual double eps(const vec &r) { return chi1p1(E_stuff, r); }

  /* scalar permeability function */
  virtual bool has_mu() { return false; } /* true if mu != 1 */
  virtual double mu(const vec &r) { return chi1p1(H_stuff, r); }

  /* scalar conductivity function */
  virtual bool has_conductivity(component c) {
    (void)c;
    return false;
  }
  virtual double conductivity(component c, const vec &r) {
    (void)c;
    (void)r;
    return 0.0;
  }

  // fallback routine based on spherical quadrature
  vec normal_vector(field_type ft, const volume &v);

  /* Return c'th row of effective 1/(1+chi1) tensor in the given grid_volume v
     ... virtual so that e.g. libctl can override with more-efficient
     libctlgeom-based routines.  maxeval == 0 if no averaging desired. */
  virtual void eff_chi1inv_row(component c, double chi1inv_row[3], const volume &v,
                               double tol = DEFAULT_SUBPIXEL_TOL,
                               int maxeval = DEFAULT_SUBPIXEL_MAXEVAL);

  /* polarizability sigma function: return c'th row of tensor */
  virtual void sigma_row(component c, double sigrow[3], const vec &r) {
    (void)c;
    (void)r;
    sigrow[0] = sigrow[1] = sigrow[2] = 0.0;
  }

  // Nonlinear susceptibilities
  virtual bool has_chi3(component c) {
    (void)c;
    return false;
  }
  virtual double chi3(component c, const vec &r) {
    (void)c;
    (void)r;
    return 0.0;
  }
  virtual bool has_chi2(component c) {
    (void)c;
    return false;
  }
  virtual double chi2(component c, const vec &r) {
    (void)c;
    (void)r;
    return 0.0;
  }
};

class simple_material_function : public material_function {
  double (*f)(const vec &);

public:
  simple_material_function(double (*func)(const vec &)) { f = func; }

  virtual ~simple_material_function() {}

  virtual double chi1p1(field_type ft, const vec &r) {
    (void)ft;
    return f(r);
  }
  virtual double eps(const vec &r) { return f(r); }
  virtual double mu(const vec &r) { return f(r); }
  virtual double conductivity(component c, const vec &r) {
    (void)c;
    return f(r);
  }
  virtual void sigma_row(component c, double sigrow[3], const vec &r) {
    sigrow[0] = sigrow[1] = sigrow[2] = 0.0;
    sigrow[component_index(c)] = f(r);
  }
  virtual double chi3(component c, const vec &r) {
    (void)c;
    return f(r);
  }
  virtual double chi2(component c, const vec &r) {
    (void)c;
    return f(r);
  }
};

template<class T>
class structure;

template<class T>
class structure_chunk {
public:
  double a, Courant, dt; // resolution a, Courant number, and timestep dt=Courant/a
  T *chi3[NUM_FIELD_COMPONENTS], *chi2[NUM_FIELD_COMPONENTS];
  T *chi1inv[NUM_FIELD_COMPONENTS][5];
  bool trivial_chi1inv[NUM_FIELD_COMPONENTS][5];
  T *conductivity[NUM_FIELD_COMPONENTS][5];
  T *condinv[NUM_FIELD_COMPONENTS][5]; // cache of 1/(1+conduct*dt/2)
  bool condinv_stale;                  // true if condinv needs to be recomputed
  T *sig[6], *kap[6], *siginv[6];      // conductivity array for uPML
  int sigsize[6];                      // conductivity array size
  grid_volume gv; // integer grid_volume that could be bigger than non-overlapping v below
  volume v;
  susceptibility<T> *chiP[NUM_FIELD_TYPES]; // only E_stuff and H_stuff are used
  double
      cost; // The cost of this chunk's grid_volume as computed by split_by_cost and fragment_stats

  int refcount; // reference count of objects using this structure_chunk

  ~structure_chunk();
  structure_chunk(const grid_volume &gv, const volume &vol_limit, double Courant, int proc_num);
  structure_chunk(const structure_chunk *);
  void set_chi1inv(component c, material_function &eps, bool use_anisotropic_averaging, double tol,
                   int maxeval);
  bool has_chi(component c, direction d) const;
  bool has_chisigma(component c, direction d) const;
  bool has_chi1inv(component c, direction d) const;
  void set_conductivity(component c, material_function &eps);
  void update_condinv();
  void set_chi3(component c, material_function &eps);
  void set_chi2(component c, material_function &eps);
  void use_pml(direction, double dx, double boundary_loc, double Rasymptotic, double mean_stretch,
               pml_profile_func pml_profile, void *pml_profile_data, double pml_profile_integral,
               double pml_profile_integral_u);

  void add_susceptibility(material_function &sigma, field_type ft, const susceptibility<T> &sus);

  void mix_with(const structure_chunk<T> *, double);

  int n_proc() const { return the_proc; } // Says which proc owns me!
  int is_mine() const { return the_is_mine; }

  void remove_susceptibilities();

  // monitor.cpp
  std::complex<double> get_chi1inv_at_pt(component, direction, int idx, double frequency = 0) const;
  std::complex<double> get_chi1inv(component, direction, const ivec &iloc,
                                   double frequency = 0) const;
  std::complex<double> get_inveps(component c, direction d, const ivec &iloc,
                                  double frequency = 0) const {
    return get_chi1inv(c, d, iloc, frequency);
  }
  double max_eps() const;

private:
  double pml_fmin;
  int the_proc;
  int the_is_mine;
};

// linked list of descriptors for boundary regions (currently just for PML)
template<class T>
class boundary_region {
public:
  typedef enum { NOTHING_SPECIAL, PML } boundary_region_kind;

  boundary_region()
      : kind(NOTHING_SPECIAL), thickness(0.0), Rasymptotic(1e-16), mean_stretch(1.0),
        pml_profile(NULL), pml_profile_data(NULL), pml_profile_integral(1.0),
        pml_profile_integral_u(1.0), d(NO_DIRECTION), side(Low), next(0) {}
  boundary_region(boundary_region_kind kind, double thickness, double Rasymptotic,
                  double mean_stretch, pml_profile_func pml_profile, void *pml_profile_data,
                  double pml_profile_integral, double pml_profile_integral_u, direction d,
                  boundary_side side, boundary_region *next = 0)
      : kind(kind), thickness(thickness), Rasymptotic(Rasymptotic), mean_stretch(mean_stretch),
        pml_profile(pml_profile), pml_profile_data(pml_profile_data),
        pml_profile_integral(pml_profile_integral), pml_profile_integral_u(pml_profile_integral_u),
        d(d), side(side), next(next) {}

  boundary_region(const boundary_region &r)
      : kind(r.kind), thickness(r.thickness), Rasymptotic(r.Rasymptotic),
        mean_stretch(r.mean_stretch), pml_profile(r.pml_profile),
        pml_profile_data(r.pml_profile_data), pml_profile_integral(r.pml_profile_integral),
        pml_profile_integral_u(r.pml_profile_integral_u), d(r.d), side(r.side) {
    next = r.next ? new boundary_region(*r.next) : 0;
  }

  ~boundary_region() {
    if (next) delete next;
  }

  void operator=(const boundary_region &r) {
    kind = r.kind;
    thickness = r.thickness;
    Rasymptotic = r.Rasymptotic;
    mean_stretch = r.mean_stretch;
    pml_profile = r.pml_profile;
    pml_profile_data = r.pml_profile_data;
    pml_profile_integral = r.pml_profile_integral;
    pml_profile_integral_u = r.pml_profile_integral_u;
    d = r.d;
    side = r.side;
    if (next) delete next;
    next = r.next ? new boundary_region(*r.next) : 0;
  }
  boundary_region operator+(const boundary_region &r0) const {
    boundary_region r(*this), *cur = &r;
    while (cur->next)
      cur = cur->next;
    cur->next = new boundary_region(r0);
    return r;
  }

  boundary_region operator*(double strength_mult) const {
    boundary_region r(*this), *cur = &r;
    while (cur) {
      cur->Rasymptotic = pow(cur->Rasymptotic, strength_mult);
      cur = cur->next;
    }
    return r;
  }

  void apply(structure<T> *s) const;
  void apply(const structure<T> *s, structure_chunk<T> *sc) const;
  bool check_ok(const grid_volume &gv) const;

private:
  boundary_region_kind kind;
  double thickness, Rasymptotic, mean_stretch;
  pml_profile_func pml_profile;
  void *pml_profile_data;
  double pml_profile_integral, pml_profile_integral_u;
  direction d;
  boundary_side side;
  boundary_region *next;
};

template<class T>
boundary_region<T> pml(double thickness, direction d, boundary_side side, double Rasymptotic = 1e-15,
                       double mean_stretch = 1.0);
template<class T>
boundary_region<T> pml(double thickness, direction d, double Rasymptotic = 1e-15,
                       double mean_stretch = 1.0);
template<class T>
boundary_region<T> pml(double thickness, double Rasymptotic = 1e-15, double mean_stretch = 1.0);
#define no_pml() boundary_region()

class binary_partition;

template<class T>
class structure {
public:
  structure_chunk<T> **chunks;
  int num_chunks;
  bool shared_chunks; // whether modifications to chunks will be visible to fields objects
  grid_volume gv, user_volume;
  double a, Courant, dt; // resolution a, Courant number, and timestep dt=Courant/a
  volume v;
  symmetry S;
  const char *outdir;
  grid_volume *effort_volumes;
  double *effort;
  int num_effort_volumes;

  ~structure();
  structure();
  structure(const grid_volume &gv, material_function &eps,
            const boundary_region<T> &br = boundary_region<T>(), const symmetry &s = meep::identity(),
            int num_chunks = 0, double Courant = 0.5, bool use_anisotropic_averaging = false,
            double tol = DEFAULT_SUBPIXEL_TOL, int maxeval = DEFAULT_SUBPIXEL_MAXEVAL);
  structure(const grid_volume &gv, double eps(const vec &),
            const boundary_region<T> &br = boundary_region<T>(), const symmetry &s = meep::identity(),
            int num_chunks = 0, double Courant = 0.5, bool use_anisotropic_averaging = false,
            double tol = DEFAULT_SUBPIXEL_TOL, int maxeval = DEFAULT_SUBPIXEL_MAXEVAL);
  structure(const structure<T> *);
  structure(const structure<T> &);

  void set_materials(material_function &mat, bool use_anisotropic_averaging = true,
                     double tol = DEFAULT_SUBPIXEL_TOL, int maxeval = DEFAULT_SUBPIXEL_MAXEVAL);
  void set_chi1inv(component c, material_function &eps, bool use_anisotropic_averaging = true,
                   double tol = DEFAULT_SUBPIXEL_TOL, int maxeval = DEFAULT_SUBPIXEL_MAXEVAL);
  bool has_chi(component c, direction d) const;
  void set_epsilon(material_function &eps, bool use_anisotropic_averaging = true,
                   double tol = DEFAULT_SUBPIXEL_TOL, int maxeval = DEFAULT_SUBPIXEL_MAXEVAL);
  void set_epsilon(double eps(const vec &), bool use_anisotropic_averaging = true,
                   double tol = DEFAULT_SUBPIXEL_TOL, int maxeval = DEFAULT_SUBPIXEL_MAXEVAL);
  void set_mu(material_function &eps, bool use_anisotropic_averaging = true,
              double tol = DEFAULT_SUBPIXEL_TOL, int maxeval = DEFAULT_SUBPIXEL_MAXEVAL);
  void set_mu(double mu(const vec &), bool use_anisotropic_averaging = true,
              double tol = DEFAULT_SUBPIXEL_TOL, int maxeval = DEFAULT_SUBPIXEL_MAXEVAL);
  void set_conductivity(component c, material_function &conductivity);
  void set_conductivity(component C, double conductivity(const vec &));
  void set_chi3(component c, material_function &eps);
  void set_chi3(material_function &eps);
  void set_chi3(double eps(const vec &));
  void set_chi2(component c, material_function &eps);
  void set_chi2(material_function &eps);
  void set_chi2(double eps(const vec &));

  void add_susceptibility(double sigma(const vec &), field_type c, const susceptibility<T> &sus);
  void add_susceptibility(material_function &sigma, field_type c, const susceptibility<T> &sus);
  void remove_susceptibilities();

  void set_output_directory(const char *name);
  void mix_with(const structure *, double);

  bool equal_layout(const structure<T> &) const;
  void print_layout(void) const;
  std::vector<grid_volume> get_chunk_volumes() const;
  std::vector<int> get_chunk_owners() const;

  // structure_dump.cpp
  void dump(const char *filename);
  void dump_chunk_layout(const char *filename);
  void load(const char *filename);
  void load_chunk_layout(const char *filename, boundary_region<T> &br);
  void load_chunk_layout(const binary_partition *bp, boundary_region<T> &br);
  void load_chunk_layout(const std::vector<grid_volume> &gvs,
                         const std::vector<int> &ids,
                         boundary_region<T> &br);

  // monitor.cpp
  std::complex<double> get_chi1inv(component, direction, const ivec &origloc, double frequency = 0,
                                   bool parallel = true) const;
  std::complex<double> get_chi1inv(component, direction, const vec &loc, double frequency = 0,
                                   bool parallel = true) const;
  std::complex<double> get_inveps(component c, direction d, const ivec &origloc,
                                  double frequency = 0) const {
    return get_chi1inv(c, d, origloc, frequency);
  }
  std::complex<double> get_inveps(component c, direction d, const vec &loc,
                                  double frequency = 0) const {
    return get_chi1inv(c, d, loc, frequency);
  }
  std::complex<double> get_eps(const vec &loc, double frequency = 0) const;
  std::complex<double> get_mu(const vec &loc, double frequency = 0) const;
  double max_eps() const;
  double estimated_cost(int process = my_rank());

  friend class boundary_region<T>;

private:
  void use_pml(direction d, boundary_side b, double dx);
  void add_to_effort_volumes(const grid_volume &new_effort_volume, double extra_effort);
  void choose_chunkdivision(const grid_volume &gv, int num_chunks, const boundary_region<T> &br,
                            const symmetry &s);
  void check_chunks();
  void changing_chunks();
  // Helper methods for dumping and loading susceptibilities
  void set_chiP_from_file(h5file *file, const char *dataset, field_type ft);
  void write_susceptibility_params(h5file *file, const char *dname, int EorH);
};

// defined in structure.cpp
std::vector<grid_volume> choose_chunkdivision(grid_volume &gv,
                                              volume &v,
                                              int num_chunks,
                                              const symmetry &s);

class src_vol;
template<class T>
class fields;
template<class T>
class fields_chunk;
template<class T>
class flux_vol;

// Time-dependence of a current source, intended to be overridden by
// subclasses.  current() and dipole() are be related by
// current = d(dipole)/dt (or rather, the finite-difference equivalent).
class src_time {
public:
  // the following variable specifies whether the current
  // source is specified as a current or as an integrated
  // current (a dipole moment), if possible.  In the original Meep,
  // by default electric sources are integrated and magnetic
  // sources are not, but this may change.
  bool is_integrated;

  src_time() {
    is_integrated = true;
    current_time = nan;
    current_current = 0.0;
    next = NULL;
  }
  virtual ~src_time() { delete next; }
  src_time(const src_time &t) {
    is_integrated = t.is_integrated;
    current_time = t.current_time;
    current_current = t.current_current;
    current_dipole = t.current_dipole;
    if (t.next)
      next = t.next->clone();
    else
      next = NULL;
  }

  std::complex<double> dipole() const { return current_dipole; }
  std::complex<double> current() const { return current_current; }
  void update(double time, double dt) {
    if (time != current_time) {
      current_dipole = dipole(time);
      current_current = current(time, dt);
      current_time = time;
    }
  }

  // subclasses *can* override this method in order to specify the
  // current directly rather than as the derivative of dipole.
  // in that case you would probably ignore the dt argument.
  virtual std::complex<double> current(double time, double dt) const {
    return ((dipole(time + dt) - dipole(time)) / dt);
  }

  double last_time_max() { return last_time_max(0.0); }
  double last_time_max(double after);

  src_time *add_to(src_time *others, src_time **added) const;
  src_time *next;

  // subclasses should override these methods:
  virtual std::complex<double> dipole(double time) const {
    (void)time;
    return 0;
  }
  virtual double last_time() const { return 0.0; }
  virtual src_time *clone() const { return new src_time(*this); }
  virtual bool is_equal(const src_time &t) const {
    (void)t;
    return 1;
  }
  virtual std::complex<double> frequency() const { return 0.0; }
  virtual void set_frequency(std::complex<double> f) { (void)f; }

private:
  double current_time;
  std::complex<double> current_dipole, current_current;
};

bool src_times_equal(const src_time &t1, const src_time &t2);

// Gaussian-envelope source with given frequency, width, peak-time, cutoff
class gaussian_src_time : public src_time {
public:
  gaussian_src_time(double f, double fwidth, double s = 5.0);
  gaussian_src_time(double f, double w, double start_time, double end_time);
  virtual ~gaussian_src_time() {}

  virtual std::complex<double> dipole(double time) const;
  virtual double last_time() const { return float(peak_time + cutoff); };
  virtual src_time *clone() const { return new gaussian_src_time(*this); }
  virtual bool is_equal(const src_time &t) const;
  virtual std::complex<double> frequency() const { return freq; }
  virtual void set_frequency(std::complex<double> f) { freq = real(f); }
  std::complex<double> fourier_transform(const double f);

private:
  double freq, width, peak_time, cutoff;
};

// Continuous (CW) source with (optional) slow turn-on and/or turn-off.
class continuous_src_time : public src_time {
public:
  continuous_src_time(std::complex<double> f, double w = 0.0, double st = 0.0, double et = infinity,
                      double s = 3.0)
      : freq(f), width(w), start_time(float(st)), end_time(float(et)), slowness(s) {}
  virtual ~continuous_src_time() {}

  virtual std::complex<double> dipole(double time) const;
  virtual double last_time() const { return end_time; };
  virtual src_time *clone() const { return new continuous_src_time(*this); }
  virtual bool is_equal(const src_time &t) const;
  virtual std::complex<double> frequency() const { return freq; }
  virtual void set_frequency(std::complex<double> f) { freq = f; }

private:
  std::complex<double> freq;
  double width, start_time, end_time, slowness;
};

// user-specified source function with start and end times
class custom_src_time : public src_time {
public:
  custom_src_time(std::complex<double> (*func)(double t, void *), void *data, double st = -infinity,
                  double et = infinity, std::complex<double> f = 0)
      : func(func), data(data), freq(f), start_time(float(st)), end_time(float(et)) {}
  virtual ~custom_src_time() {}

  virtual std::complex<double> current(double time, double dt) const {
    if (is_integrated)
      return src_time::current(time, dt);
    else
      return dipole(time);
  }
  virtual std::complex<double> dipole(double time) const {
    float rtime = float(time);
    if (rtime >= start_time && rtime <= end_time)
      return func(time, data);
    else
      return 0.0;
  }
  virtual double last_time() const { return end_time; };
  virtual src_time *clone() const { return new custom_src_time(*this); }
  virtual bool is_equal(const src_time &t) const;
  virtual std::complex<double> frequency() const { return freq; }
  virtual void set_frequency(std::complex<double> f) { freq = f; }

private:
  std::complex<double> (*func)(double t, void *);
  void *data;
  std::complex<double> freq;
  double start_time, end_time;
};

class monitor_point {
public:
  monitor_point();
  ~monitor_point();
  vec loc;
  double t;
  std::complex<double> f[NUM_FIELD_COMPONENTS];
  monitor_point *next;

  std::complex<double> get_component(component);
  double poynting_in_direction(direction d);
  double poynting_in_direction(vec direction_v);

  // When called with only its first four arguments, fourier_transform
  // performs an FFT on its monitor points, putting the frequencies in f
  // and the amplitudes in a.  Yes, the frequencies are trivial and
  // redundant, but this saves you the risk of making a mistake in
  // converting your units.  Note also, that in this case f is always a
  // real number, although it's stored in a complex.
  //
  // Note that in either case, fourier_transform assumes that the monitor
  // points are all equally spaced in time.
  void fourier_transform(component w, std::complex<double> **a, std::complex<double> **f,
                         int *numout, double fmin = 0.0, double fmax = 0.0, int maxbands = 100);
  // harminv works much like fourier_transform, except that it is not yet
  // implemented.
  void harminv(component w, std::complex<double> **a, std::complex<double> **f, int *numout,
               double fmin, double fmax, int maxbands);
};

// dft.cpp
// this should normally only be created with fields::add_dft
template<class T>
class dft_chunk {
public:
  dft_chunk(fields_chunk<T> *fc_, ivec is_, ivec ie_, vec s0_, vec s1_, vec e0_, vec e1_, double dV0_,
            double dV1_, component c_, bool use_centered_grid, std::complex<double> phase_factor,
            ivec shift_, const symmetry &S_, int sn_, const void *data_);
  ~dft_chunk();

  void update_dft(double time);

  void scale_dft(std::complex<double> scale);

  // chunk-by-chunk helper routine called by
  // fields::process_dft_component
  std::complex<double> process_dft_component(int rank, direction *ds, ivec min_corner,
                                             ivec max_corner, int num_freq, h5file *file,
                                             double *buffer, int reim,
                                             std::complex<double> *field_array, void *mode1_data,
                                             void *mode2_data, int ic_conjugate,
                                             bool retain_interp_weights, fields<T> *parent);

  void operator-=(const dft_chunk &chunk);

  // the frequencies to loop_in_chunks
  std::vector<double> omega;

  component c; // component to DFT (possibly transformed by symmetry)

  size_t N;                   // number of spatial points (on epsilon grid)
  std::complex<double> *dft; // N x Nomega array of DFT values.

  class dft_chunk *next_in_chunk; // per-fields_chunk list of DFT chunks
  class dft_chunk *next_in_dft;   // next for this particular DFT vol./component

  /* There are several types of weight factors associated with DFT fields: */
  /*  (a) To accelerate the computation of things like Poynting flux, it   */
  /*      is convenient to store certain DFT field components with built-in*/
  /*      constant prefactors (usually just \pm 1). For example, in a      */
  /*      dft_flux_plane normal to the Z direction the Ey component is     */
  /*      stored with a built-in minus sign, while the other components    */
  /*      (Ex, Hx, Hy) are not. This factor is already included in the     */
  /*      `scale` field, but we also need to keep track of it separately   */
  /*      so we can divide it out when looking up the values of individual */
  /*      DFT field components. So we store it as `stored_weight.`         */
  /*                                                                       */
  /*  (b) For similar reasons, it is convenient to store certain DFT field */
  /*      components with built-in volume factors to accelerate numerical  */
  /*      integrations. In this case the prefactor is not constant (it     */
  /*      varies from grid point to grid point) so we can't store it in    */
  /*      the dft_chunk structure like stored_weight; instead we store a   */
  /*      flag to indicate that it is present in the stored field          */
  /*      components. This is the include_dV_and_interp_weights flag.      */
  /*      (The sqrt_dV_and_interp_weights flag indicates that the sqrt of  */
  /*      the volume factor is stored instead.)                            */
  /*                                                                       */
  /*  (c) When computing things like -0.5*|E|^2 for the stress tensor, we  */
  /*      we cannot incorporate the minus sign into the scale factor       */
  /*      because we only ever compute |scale|^2. Thus, it is necessary    */
  /*      to store an additional weight factor with the dft_chunk to record*/
  /*      any additional negative or complex weight factor to be used in   */
  /*      in computations involving the fourier-transformed fields. This   */
  /*      is the extra_weight field. Because it is used in computations    */
  /*       involving dft[...], it needs to be public.                      */
  std::complex<double> stored_weight;
  bool include_dV_and_interp_weights;
  bool sqrt_dV_and_interp_weights;
  std::complex<double> extra_weight;

  // parameters passed from field_integrate:
  fields_chunk<T> *fc;
  ivec is, ie;
  vec s0, s1, e0, e1;
  double dV0, dV1;
  bool empty_dim[5];          // which directions correspond to empty dimensions in original volume
  std::complex<double> scale; // scale factor * phase from shift and symmetry
  ivec shift;
  symmetry S;
  int sn;

  // cache of exp(iwt) * scale, of length Nomega
  std::complex<double> *dft_phase;

  ptrdiff_t avg1, avg2; // index offsets for average to get epsilon grid

  int vc; // component descriptor from the original volume
};

template<class T>
void save_dft_hdf5(dft_chunk<T> *dft_chunks, component c, h5file *file, const char *dprefix = 0);
template<class T>
void load_dft_hdf5(dft_chunk<T> *dft_chunks, component c, h5file *file, const char *dprefix = 0);
template<class T>
void save_dft_hdf5(dft_chunk<T> *dft_chunks, const char *name, h5file *file, const char *dprefix = 0);
template<class T>
void load_dft_hdf5(dft_chunk<T> *dft_chunks, const char *name, h5file *file, const char *dprefix = 0);

// dft.cpp (normally created with fields::add_dft_flux)
template<class T>
class dft_flux {
public:
  dft_flux(const component cE_, const component cH_, dft_chunk<T> *E_, dft_chunk<T> *H_, double fmin,
           double fmax, int Nf, const volume &where_, direction normal_direction_,
           bool use_symmetry_);
  dft_flux(const component cE_, const component cH_, dft_chunk<T> *E_, dft_chunk<T> *H_,
           const std::vector<double> freq_, const volume &where_, direction normal_direction_,
           bool use_symmetry_);
  dft_flux(const component cE_, const component cH_, dft_chunk<T> *E_, dft_chunk<T> *H_,
           const double *freq_, size_t Nfreq, const volume &where_, direction normal_direction_,
           bool use_symmetry_);
  dft_flux(const dft_flux &f);

  double *flux();

  void save_hdf5(h5file *file, const char *dprefix = 0);
  void load_hdf5(h5file *file, const char *dprefix = 0);

  void operator-=(const dft_flux<T> &fl) {
    if (E && fl.E) *E -= *fl.E;
    if (H && fl.H) *H -= *fl.H;
  }

  void save_hdf5(fields<T> &f, const char *fname, const char *dprefix = 0, const char *prefix = 0);
  void load_hdf5(fields<T> &f, const char *fname, const char *dprefix = 0, const char *prefix = 0);

  void scale_dfts(std::complex<double> scale);

  void remove();

  std::vector<double> freq;
  dft_chunk<T> *E, *H;
  component cE, cH;
  volume where;
  direction normal_direction;
  bool use_symmetry;
};

// dft.cpp (normally created with fields::add_dft_energy)
template<class T>
class dft_energy {
public:
  dft_energy(dft_chunk<T> *E_, dft_chunk<T> *H_, dft_chunk<T> *D_, dft_chunk<T> *B_, double freq_min,
             double freq_max, int Nf, const volume &where_);
  dft_energy(dft_chunk<T> *E_, dft_chunk<T> *H_, dft_chunk<T> *D_, dft_chunk<T> *B_,
             const std::vector<double> freq_, const volume &where_);
  dft_energy(dft_chunk<T> *E_, dft_chunk<T> *H_, dft_chunk<T> *D_, dft_chunk<T> *B_, const double *freq_,
             size_t Nfreq, const volume &where_);
  dft_energy(const dft_energy &f);

  double *electric();
  double *magnetic();
  double *total();

  void save_hdf5(h5file *file, const char *dprefix = 0);
  void load_hdf5(h5file *file, const char *dprefix = 0);

  void operator-=(const dft_energy &fl) {
    if (E && fl.E) *E -= *fl.E;
    if (H && fl.H) *H -= *fl.H;
    if (D && fl.D) *D -= *fl.D;
    if (B && fl.B) *B -= *fl.B;
  }

  void save_hdf5(fields<T> &f, const char *fname, const char *dprefix = 0, const char *prefix = 0);
  void load_hdf5(fields<T> &f, const char *fname, const char *dprefix = 0, const char *prefix = 0);

  void scale_dfts(std::complex<double> scale);

  void remove();

  std::vector<double> freq;
  dft_chunk<T> *E, *H, *D, *B;
  volume where;
};

// stress.cpp (normally created with fields::add_dft_force)
template<class T>
class dft_force {
public:
  dft_force(dft_chunk<T> *offdiag1_, dft_chunk<T> *offdiag2_, dft_chunk<T> *diag_, double fmin, double fmax,
            int Nf, const volume &where_);
  dft_force(dft_chunk<T> *offdiag1_, dft_chunk<T> *offdiag2_, dft_chunk<T> *diag_,
            const std::vector<double> freq_, const volume &where_);
  dft_force(dft_chunk<T> *offdiag1_, dft_chunk<T> *offdiag2_, dft_chunk<T> *diag_, const double *freq_,
            size_t Nfreq, const volume &where_);
  dft_force(const dft_force &f);

  double *force();

  void save_hdf5(h5file *file, const char *dprefix = 0);
  void load_hdf5(h5file *file, const char *dprefix = 0);

  void operator-=(const dft_force &fl);

  void save_hdf5(fields<T> &f, const char *fname, const char *dprefix = 0, const char *prefix = 0);
  void load_hdf5(fields<T> &f, const char *fname, const char *dprefix = 0, const char *prefix = 0);

  void scale_dfts(std::complex<double> scale);

  void remove();

  std::vector<double> freq;
  dft_chunk<T> *offdiag1, *offdiag2, *diag;
  volume where;
};


struct sourcedata {
  component near_fd_comp;
  std::vector<ptrdiff_t> idx_arr;
  int fc_idx;
  std::vector<std::complex<double> > amp_arr;
};


// near2far.cpp (normally created with fields::add_dft_near2far)
template<class T>
class dft_near2far {
public:
  /* fourier tranforms of tangential E and H field components in a
     medium with the given scalar eps and mu */
  dft_near2far(dft_chunk<T> *F, double fmin, double fmax, int Nf, double eps, double mu,
               const volume &where_, const direction periodic_d_[2], const int periodic_n_[2],
               const double periodic_k_[2], const double period_[2]);
  dft_near2far(dft_chunk<T> *F, const std::vector<double> freq_, double eps, double mu,
               const volume &where_, const direction periodic_d_[2], const int periodic_n_[2],
               const double periodic_k_[2], const double period_[2]);
  dft_near2far(dft_chunk<T> *F, const double *freq_, size_t Nfreq, double eps, double mu,
               const volume &where_, const direction periodic_d_[2], const int periodic_n_[2],
               const double periodic_k_[2], const double period_[2]);
  dft_near2far(const dft_near2far &f);

  /* return an array (Ex,Ey,Ez,Hx,Hy,Hz) x Nfreq of the far fields at x */
  std::complex<double> *farfield(const vec &x);

  /* like farfield, but requires F to be Nfreq*6 preallocated array, and
     does *not* perform the reduction over processes...an MPI allreduce
     summation by the caller is required to get the final result ... used
     by other output routine to efficiently get far field on a grid of pts */
  void farfield_lowlevel(std::complex<double> *F, const vec &x);

  /* Return a newly allocated array with all far fields */
  double *get_farfields_array(const volume &where, int &rank, size_t *dims, size_t &N,
                              double resolution);

  /* output far fields on a grid to an HDF5 file */
  void save_farfields(const char *fname, const char *prefix, const volume &where,
                      double resolution);

  /* output Poynting flux of far fields */
  double *flux(direction df, const volume &where, double resolution);

  void save_hdf5(h5file *file, const char *dprefix = 0);
  void load_hdf5(h5file *file, const char *dprefix = 0);

  void operator-=(const dft_near2far &fl);

  void save_hdf5(fields<T> &f, const char *fname, const char *dprefix = 0, const char *prefix = 0);
  void load_hdf5(fields<T> &f, const char *fname, const char *dprefix = 0, const char *prefix = 0);

  void scale_dfts(std::complex<double> scale);

  void remove();

  std::vector<double> freq;
  dft_chunk<T> *F;
  double eps, mu;
  volume where;
  direction periodic_d[2];
  int periodic_n[2];
  double periodic_k[2], period[2];

  std::vector<sourcedata> near_sourcedata(const vec &x_0, double* farpt_list, size_t nfar_pts, std::complex<double>* dJ);
};

/* Class to compute local-density-of-states spectra: the power spectrum
   P(omega) of the work done by the sources.  Specialized to handle only
   the case where all sources have the same time dependence, which greatly
   simplifies things because then we can do the spatial integral of E*J
   *first* and then do the Fourier transform, eliminating the need to
   store the Fourier transform per point or per current. */
template<class T>
class dft_ldos {
public:
  dft_ldos(double freq_min, double freq_max, int Nfreq);
  dft_ldos(const std::vector<double> freq);
  dft_ldos(const double *freq, size_t Nfreq);
  ~dft_ldos() {
    delete[] Fdft;
    delete[] Jdft;
  }

  void update(fields<T> &f);       // to be called after each timestep
  double *ldos() const;            // returns array of Nomega values (after last timestep)
  std::complex<double> *F() const; // returns Fdft
  std::complex<double> *J() const; // returns Jdft
  std::vector<double> freq;

private:
  std::complex<double> *Fdft;  // Nomega array of field * J*(x) DFT values
  std::complex<double> *Jdft;  // Nomega array of J(t) DFT values
  double Jsum;                 // sum of |J| over all points
};

// dft.cpp (normally created with fields::add_dft_fields)
template<class T>
class dft_fields {
public:
  dft_fields(dft_chunk<T> *chunks, double freq_min, double freq_max, int Nfreq, const volume &where);
  dft_fields(dft_chunk<T> *chunks, const std::vector<double> freq_, const volume &where);
  dft_fields(dft_chunk<T> *chunks, const double *freq_, size_t Nfreq, const volume &where);

  void scale_dfts(std::complex<double> scale);

  void remove();

  std::vector<double> freq;
  dft_chunk<T> *chunks;
  volume where;
};

enum in_or_out { Incoming = 0, Outgoing };
enum connect_phase { CONNECT_PHASE = 0, CONNECT_NEGATE = 1, CONNECT_COPY = 2 };

// data for each susceptibility
typedef struct polarization_state_s {
  void *data; // internal polarization data for the susceptibility
  const susceptibility<realnum> *s;
  struct polarization_state_s *next; // linked list
} polarization_state;

template<class T>
class fields_chunk {
public:
  T *f[NUM_FIELD_COMPONENTS][2]; // fields at current time

  // auxiliary fields needed for PML (at least in some components)
  T *f_u[NUM_FIELD_COMPONENTS][2];    // integrated from D/B
  T *f_w[NUM_FIELD_COMPONENTS][2];    // E/H integrated from these
  T *f_cond[NUM_FIELD_COMPONENTS][2]; // aux field for PML+conductivity

  /* sometimes, to synchronize the E and H fields, e.g. for computing
     flux at a given time, we need to timestep H by 1/2; in this case
     we save backup copies of (some of) the fields to resume timestepping */
  T *f_backup[NUM_FIELD_COMPONENTS][2];
  T *f_u_backup[NUM_FIELD_COMPONENTS][2];
  T *f_w_backup[NUM_FIELD_COMPONENTS][2];
  T *f_cond_backup[NUM_FIELD_COMPONENTS][2];

  // W (or E/H) field from prev. timestep, only stored if needed by update_pols
  T *f_w_prev[NUM_FIELD_COMPONENTS][2];

  // used to store D-P and B-P, e.g. when P implements dispersive media
  T *f_minus_p[NUM_FIELD_COMPONENTS][2];

  T *f_rderiv_int; // cache of helper field for 1/r d(rf)/dr derivative

  dft_chunk<T> *dft_chunks;

  T **zeroes[NUM_FIELD_TYPES]; // Holds pointers to metal points.
  size_t num_zeroes[NUM_FIELD_TYPES];
  T **connections[NUM_FIELD_TYPES][CONNECT_COPY + 1][Outgoing + 1];
  size_t num_connections[NUM_FIELD_TYPES][CONNECT_COPY + 1][Outgoing + 1];
  std::complex<T> *connection_phases[NUM_FIELD_TYPES];

  int npol[NUM_FIELD_TYPES];                   // only E_stuff and H_stuff are used
  polarization_state *pol[NUM_FIELD_TYPES]; // array of npol[i] polarization_state structures

  double a, Courant, dt; // resolution a, Courant number, and timestep dt=Courant/a
  grid_volume gv;
  volume v;
  double m;                        // angular dependence in cyl. coords
  bool zero_fields_near_cylorigin; // fields=0 m pixels near r=0 for stability
  double beta;
  int is_real;
  src_vol *sources[NUM_FIELD_TYPES];
  structure_chunk<T> *new_s;
  structure_chunk<T> *s;
  const char *outdir;
  int chunk_idx;

  fields_chunk(structure_chunk<T> *, const char *outdir, double m, double beta,
               bool zero_fields_near_cylorigin, int chunkidx);

  fields_chunk(const fields_chunk<T> &, int chunkidx);
  ~fields_chunk();

  void use_real_fields();
  bool have_component(component c, bool is_complex = false) {
    switch (c) {
      case Dielectric:
      case Permeability:
      case NO_COMPONENT: return !is_complex;
      default: return (f[c][0] && f[c][is_complex]);
    }
  }

  double last_source_time();
  // monitor.cpp
  std::complex<double> get_field(component, const ivec &) const;

  std::complex<double> get_chi1inv(component, direction, const ivec &iloc,
                                   double frequency = 0) const;

  void backup_component(component c);
  void average_with_backup(component c);
  void restore_component(component c);

  void set_output_directory(const char *name);

  double count_volume(component);
  friend class fields<T>;

  int n_proc() const { return s->n_proc(); };
  int is_mine() const { return s->is_mine(); };
  // boundaries.cpp
  void zero_metal(field_type);
  bool needs_W_notowned(component c);
  // fields.cpp
  void remove_sources();
  void remove_susceptibilities(bool shared_chunks);
  void zero_fields();

  // update_eh.cpp
  bool needs_W_prev(component c) const;
  bool update_eh(field_type ft, bool skip_w_components = false);

  bool alloc_f(component c);
  void figure_out_step_plan();

  void set_solve_cw_omega(std::complex<double> omega) {
    doing_solve_cw = true;
    solve_cw_omega = omega;
  }
  void unset_solve_cw_omega() {
    doing_solve_cw = false;
    solve_cw_omega = 0.0;
  }

private:
  // we set a flag during cw_solve to replace some
  // time-dependent stuff with the analogous frequency-domain operation
  bool doing_solve_cw;                 // true when inside solve_cw
  std::complex<double> solve_cw_omega; // current omega for solve_cw

  // fields.cpp
  bool have_plus_deriv[NUM_FIELD_COMPONENTS], have_minus_deriv[NUM_FIELD_COMPONENTS];
  component plus_component[NUM_FIELD_COMPONENTS], minus_component[NUM_FIELD_COMPONENTS];
  direction plus_deriv_direction[NUM_FIELD_COMPONENTS], minus_deriv_direction[NUM_FIELD_COMPONENTS];
  // step.cpp
  void phase_in_material(structure_chunk<T> *s);
  void phase_material(int phasein_time);
  bool step_db(field_type ft);
  void step_source(field_type ft, bool including_integrated);
  bool update_pols(field_type ft);
  void calc_sources(double time);

  // initialize.cpp
  void initialize_field(component, std::complex<double> f(const vec &));
  void initialize_with_nth_te(int n, double kz);
  void initialize_with_nth_tm(int n, double kz);
  // boundaries.cpp
  void alloc_extra_connections(field_type, connect_phase, in_or_out, size_t);
  // dft.cpp
  void update_dfts(double timeE, double timeH);

  void changing_structure();
};

enum boundary_condition { Periodic = 0, Metallic, Magnetic, None };
enum time_sink {
  Connecting,
  Stepping,
  Boundaries,
  MpiAllTime,
  MpiOneTime,
  FieldOutput,
  FourierTransforming,
  MPBTime,
  GetFarfieldsTime,
  Other
};

typedef void (*field_chunkloop)(fields_chunk<realnum> *fc, int ichunk, component cgrid, ivec is, ivec ie,
                                vec s0, vec s1, vec e0, vec e1, double dV0, double dV1, ivec shift,
                                std::complex<double> shift_phase, const symmetry &S, int sn,
                                void *chunkloop_data);
typedef std::complex<double> (*field_function)(const std::complex<double> *fields, const vec &loc,
                                               void *integrand_data_);
typedef double (*field_rfunction)(const std::complex<double> *fields, const vec &loc,
                                  void *integrand_data_);

field_rfunction derived_component_func(derived_component c, const grid_volume &gv, int &nfields,
                                       component cs[12]);

/* A utility class for loop_in_chunks, for fetching values of field
   components at grid points, accounting for the complications
   of symmetry and yee-grid averaging. */
template<class T>
class chunkloop_field_components {
private:
  fields_chunk<T> *fc;
  std::vector<component> parent_components;
  std::vector<std::complex<double> > phases;
  std::vector<ptrdiff_t> offsets;

public:
  chunkloop_field_components(fields_chunk<T> *fc, component cgrid, std::complex<double> shift_phase,
                             const symmetry &S, int sn, int num_fields,
                             const component *components);
#if __cplusplus >= 201103L // delegating constructors are a C++11 feature
  chunkloop_field_components(fields_chunk<T> *fc, component cgrid, std::complex<double> shift_phase,
                             const symmetry &S, int sn, std::vector<component> components)
      : chunkloop_field_components(fc, cgrid, shift_phase, S, sn, components.size(),
                                   components.data()) {}
#endif
  void update_values(ptrdiff_t idx);
  std::vector<std::complex<double> > values; // updated by update_values(idx)
};

class diffractedplanewave {

public:
  diffractedplanewave(int g[3], double axis[3], std::complex<double> s, std::complex<double> p);
  int *get_g() { return g; };
  double *get_axis() { return axis; }
  std::complex<double> get_s() const { return s; };
  std::complex<double> get_p() const { return p; };

private:
  int g[3];                // diffraction order
  double axis[3];          // axis vector
  std::complex<double> s;  // s polarization amplitude
  std::complex<double> p;  // p polarization ampiltude
};

class gaussianbeam {

public:
  gaussianbeam(const vec &x0, const vec &kdir, double w0, double freq,
               double eps, double mu, std::complex<double> E0[3]);
  void get_fields(std::complex<double> *EH, const vec &x) const;
  std::complex<double> get_E0(int n) const { return E0[n]; };

private:
  vec x0;           // beam center
  vec kdir;         // beam propagation direction
  double w0;        // beam waist radius
  double freq;      // beam frequency
  double eps, mu;   // permittivity/permeability of homogeneous medium
  std::complex<double> E0[3]; // polarization vector
};

/***************************************************************/
/* prototype for optional user-supplied function to provide an */
/* initial estimate of the wavevector of mode #mode at         */
/* frequency freq for eigenmode calculations                   */
/***************************************************************/
typedef vec (*kpoint_func)(double freq, int mode, void *user_data);

template<class T>
class fields {
public:
  int num_chunks;
  bool shared_chunks;
  fields_chunk<T> **chunks;
  src_time *sources;
  flux_vol<T> *fluxes;
  symmetry S;

  // The following is an array that is num_chunks by num_chunks.  Actually
  // it is two arrays, one for the imaginary and one for the real part.
  T **comm_blocks[NUM_FIELD_TYPES];
  // This is the same size as each comm_blocks array, and store the sizes
  // of the comm blocks themselves for each connection-phase type
  size_t *comm_sizes[NUM_FIELD_TYPES][CONNECT_COPY + 1];
  size_t comm_size_tot(int f, int pair) const {
    size_t sum = 0;
    for (int ip = 0; ip < 3; ++ip)
      sum += comm_sizes[f][ip][pair];
    return sum;
  }

  double a, dt; // The resolution a and timestep dt=Courant/a
  grid_volume gv, user_volume;
  volume v;
  double m;
  double beta;
  int t, phasein_time, is_real;
  std::complex<double> k[5], eikna[5];
  double coskna[5], sinkna[5];
  boundary_condition boundaries[2][5];
  char *outdir;
  bool components_allocated;

  // fields.cpp methods:
  fields(structure<T> *, double m = 0, double beta = 0, bool zero_fields_near_cylorigin = true);
  fields(const fields &);
  ~fields();
  bool equal_layout(const fields &f) const;
  void use_real_fields();
  void zero_fields();
  void remove_sources();
  void remove_susceptibilities();
  void remove_fluxes();
  void reset();

  // time.cpp
  std::vector<double> time_spent_on(time_sink);
  double mean_time_spent_on(time_sink);
  void print_times();
  // boundaries.cpp
  void set_boundary(boundary_side, direction, boundary_condition);
  void use_bloch(direction d, double k) { use_bloch(d, (std::complex<double>)k); }
  void use_bloch(direction, std::complex<double> kz);
  void use_bloch(const vec &k);
  vec lattice_vector(direction) const;
  // update_eh.cpp
  void update_eh(field_type ft, bool skip_w_components = false);

  volume total_volume(void) const;

  // h5fields.cpp:
  // low-level function:
  void output_hdf5(h5file *file, const char *dataname, int num_fields, const component *components,
                   field_function fun, void *fun_data_, int reim, const volume &where,
                   bool append_data = false, bool single_precision = false, double frequency = 0);
  // higher-level functions
  void output_hdf5(const char *dataname, // OUTPUT COMPLEX-VALUED FUNCTION
                   int num_fields, const component *components, field_function fun, void *fun_data_,
                   const volume &where, h5file *file = 0, bool append_data = false,
                   bool single_precision = false, const char *prefix = 0,
                   bool real_part_only = false, double frequency = 0);
  void output_hdf5(const char *dataname, // OUTPUT REAL-VALUED FUNCTION
                   int num_fields, const component *components, field_rfunction fun,
                   void *fun_data_, const volume &where, h5file *file = 0, bool append_data = false,
                   bool single_precision = false, const char *prefix = 0, double = 0);
  void output_hdf5(component c, // OUTPUT FIELD COMPONENT (or Dielectric)
                   const volume &where, h5file *file = 0, bool append_data = false,
                   bool single_precision = false, const char *prefix = 0, double frequency = 0);
  void output_hdf5(derived_component c, // OUTPUT DERIVED FIELD COMPONENT
                   const volume &where, h5file *file = 0, bool append_data = false,
                   bool single_precision = false, const char *prefix = 0, double frequency = 0);
  h5file *open_h5file(const char *name, h5file::access_mode mode = h5file::WRITE,
                      const char *prefix = NULL, bool timestamp = false);
  const char *h5file_name(const char *name, const char *prefix = NULL, bool timestamp = false);

  void output_times(const char *fname);

  // array_slice.cpp methods

  // given a subvolume, compute the dimensions of the array slice
  // needed to store field data for that subvolume.
  // if `where` has zero thickness in (say) the x dimension,
  // i.e. the volume lives entirely at a single x-coordinate x0,
  // then the array slice will nonetheless generally have length 2
  // in the x direction (corresponding to the two grid points
  // nearest x0, from which fields at x0 are interpolated).
  // if collapse_empty_dimensions==true, all such length-2
  // array dimensions are collapsed to length 1 by doing the
  // interpolation before returning the array.
  //
  // the `data` parameter is used internally in get_array_slice
  // and should be ignored by external callers.
  int get_array_slice_dimensions(const volume &where, size_t dims[3], direction dirs[3],
                                 bool collapse_empty_dimensions = false,
                                 bool snap_empty_dimensions = false, vec *min_max_loc = NULL,
                                 void *data = 0, component cgrid = Centered);

  // given a subvolume, return a column-major array containing
  // the given function of the field components in that subvolume
  // if slice is non-null, it must be a user-allocated buffer
  // of the correct size.
  // otherwise, a new buffer is allocated and returned; it
  // must eventually be caller-deallocated via delete[].
  double *get_array_slice(const volume &where, std::vector<component> components,
                          field_rfunction rfun, void *fun_data, double *slice = 0,
                          double frequency = 0,
                          bool snap = false);

  std::complex<double> *get_complex_array_slice(const volume &where,
                                                std::vector<component> components,
                                                field_function fun, void *fun_data,
                                                std::complex<double> *slice = 0,
                                                double frequency = 0,
                                                bool snap = false);

  // alternative entry points for when you have no field
  // function, i.e. you want just a single component or
  // derived component.)
  double *get_array_slice(const volume &where, component c, double *slice = 0,
                          double frequency = 0, bool snap = false);
  double *get_array_slice(const volume &where, derived_component c, double *slice = 0,
                          double frequency = 0, bool snap = false);
  std::complex<double> *get_complex_array_slice(const volume &where, component c,
                                                std::complex<double> *slice = 0,
                                                double frequency = 0,
                                                bool snap = false);

  // like get_array_slice, but for *sources* instead of fields
  std::complex<double> *get_source_slice(const volume &where, component source_slice_component,
                                         std::complex<double> *slice = 0);

  // master routine for all above entry points
  void *do_get_array_slice(const volume &where, std::vector<component> components,
                           field_function fun, field_rfunction rfun, void *fun_data, void *vslice,
                           double frequency = 0, bool snap = false);

  /* fetch and return coordinates and integration weights of grid points covered by an array slice, */
  /* packed into a vector with format [NX, xtics[:], NY, ytics[:], NZ, ztics[:], weights[:] ] */
  std::vector<double> get_array_metadata(const volume &where);

  // step.cpp methods:
  double last_step_output_wall_time;
  int last_step_output_t;
  void step();

  // when comparing times, e.g. for source cutoffs, it
  // is useful to round to float to avoid gratuitous sensitivity
  // to floating-point roundoff error
  inline double round_time() const { return float(t * dt); };
  inline double time() const { return t * dt; };

  // cw_fields.cpp:
  bool solve_cw(double tol, int maxiters, std::complex<double> frequency, int L = 2,
                std::complex<double> *eigfreq = NULL, double eigtol = 1e-8, int eigiters = 20);
  bool solve_cw(double tol = 1e-8, int maxiters = 10000, int L = 2,
                std::complex<double> *eigfreq = NULL, double eigtol = 1e-8, int eigiters = 20);

  // sources.cpp:
  double last_source_time();
  void add_point_source(component c, double freq, double width, double peaktime, double cutoff,
                        const vec &, std::complex<double> amp = 1.0, int is_continuous = 0);
  void add_point_source(component c, const src_time &src, const vec &,
                        std::complex<double> amp = 1.0);

  void add_volume_source(component c, const src_time &src, const volume &where_,
                         std::complex<double> *arr, size_t dim1, size_t dim2, size_t dim3,
                         std::complex<double> amp);
  void add_volume_source(component c, const src_time &src, const volume &where_,
                         const char *filename, const char *dataset, std::complex<double> amp);
  void add_volume_source(component c, const src_time &src, const volume &,
                         std::complex<double> A(const vec &), std::complex<double> amp = 1.0);
  void add_volume_source(component c, const src_time &src, const volume &,
                         std::complex<double> amp = 1.0);
  void add_volume_source(const src_time &src, const volume &, gaussianbeam beam);
  bool is_aniso2d();
  void require_source_components();
  void _require_component(component c, bool aniso2d);
  void require_component(component c) { _require_component(c, is_aniso2d()); }
  void add_srcdata(struct sourcedata cur_data, src_time *src, size_t n, std::complex<double>* amp_arr);

  // mpb.cpp

  // the return value of get_eigenmode is an opaque pointer
  // that can be passed to eigenmode_amplitude() to get
  // values of field components at arbitrary points in space.
  // call destroy_eigenmode_data() to deallocate it when finished.
  void *get_eigenmode(double frequency, direction d, const volume where, const volume eig_vol,
                      int band_num, const vec &kpoint, bool match_frequency, int parity,
                      double resolution, double eigensolver_tol, double *kdom = 0,
                      void **user_mdata = 0, diffractedplanewave *dp = 0);

  void add_eigenmode_source(component c, const src_time &src, direction d, const volume &where,
                            const volume &eig_vol, int band_num, const vec &kpoint,
                            bool match_frequency, int parity, double eig_resolution,
                            double eigensolver_tol, std::complex<double> amp,
                            std::complex<double> A(const vec &) = 0,
                            diffractedplanewave *dp = 0);

  void get_eigenmode_coefficients(dft_flux<T> flux, const volume &eig_vol, int *bands, int num_bands,
                                  int parity, double eig_resolution, double eigensolver_tol,
                                  std::complex<double> *coeffs, double *vgrp,
                                  kpoint_func user_kpoint_func, void *user_kpoint_data,
                                  vec *kpoints, vec *kdom, double *cscale, direction d,
                                  diffractedplanewave *dp = 0);
  void get_eigenmode_coefficients(dft_flux<T> flux, const volume &eig_vol, int *bands, int num_bands,
                                  int parity, double eig_resolution, double eigensolver_tol,
                                  std::complex<double> *coeffs, double *vgrp,
                                  kpoint_func user_kpoint_func = 0, void *user_kpoint_data = 0,
                                  vec *kpoints = 0, vec *kdom = 0, double *cscale = 0,
                                  diffractedplanewave *dp = 0);

  // initialize.cpp:
  void initialize_field(component, std::complex<double> f(const vec &));
  void initialize_with_nth_te(int n);
  void initialize_with_nth_tm(int n);
  void initialize_with_n_te(int ntot);
  void initialize_with_n_tm(int ntot);
  int phase_in_material(const structure<T> *s, double time);
  int is_phasing();

  // loop_in_chunks.cpp
  void loop_in_chunks(field_chunkloop chunkloop, void *chunkloop_data, const volume &where,
                      component cgrid = Centered, bool use_symmetry = true,
                      bool snap_unit_dims = false);

  // integrate.cpp
  std::complex<double> integrate(int num_fields, const component *components, field_function fun,
                                 void *fun_data_, const volume &where, double *maxabs = 0);
  double integrate(int num_fields, const component *components, field_rfunction fun,
                   void *fun_data_, const volume &where, double *maxabs = 0);
  std::complex<double> integrate2(const fields &fields2, int num_fields1,
                                  const component *components1, int num_fields2,
                                  const component *components2, field_function integrand,
                                  void *integrand_data_, const volume &where, double *maxabs = 0);
  double integrate2(const fields &fields2, int num_fields1, const component *components1,
                    int num_fields2, const component *components2, field_rfunction integrand,
                    void *integrand_data_, const volume &where, double *maxabs = 0);

  double max_abs(int num_fields, const component *components, field_function fun, void *fun_data_,
                 const volume &where);
  double max_abs(int num_fields, const component *components, field_rfunction fun, void *fun_data_,
                 const volume &where);

  double max_abs(int c, const volume &where);
  double max_abs(component c, const volume &where);
  double max_abs(derived_component c, const volume &where);

  // dft.cpp
  dft_chunk<T> *add_dft(component c, const volume &where, double freq_min, double freq_max, int Nfreq,
                        bool include_dV_and_interp_weights = true,
                        std::complex<double> stored_weight = 1.0, dft_chunk<T> *chunk_next = 0,
                        bool sqrt_dV_and_interp_weights = false,
                        std::complex<double> extra_weight = 1.0, bool use_centered_grid = true,
                        int vc = 0) {
    return add_dft(c, where, linspace(freq_min, freq_max, Nfreq), include_dV_and_interp_weights,
                   stored_weight, chunk_next, sqrt_dV_and_interp_weights, extra_weight,
                   use_centered_grid, vc);
  }
  dft_chunk<T> *add_dft(component c, const volume &where, const double *freq, size_t Nfreq,
                        bool include_dV_and_interp_weights = true,
                        std::complex<double> stored_weight = 1.0, dft_chunk<T> *chunk_next = 0,
                        bool sqrt_dV_and_interp_weights = false,
                        std::complex<double> extra_weight = 1.0, bool use_centered_grid = true,
                        int vc = 0);
  dft_chunk<T> *add_dft(component c, const volume &where, const std::vector<double> freq,
                        bool include_dV_and_interp_weights = true,
                        std::complex<double> stored_weight = 1.0, dft_chunk<T> *chunk_next = 0,
                        bool sqrt_dV_and_interp_weights = false,
                        std::complex<double> extra_weight = 1.0, bool use_centered_grid = true,
                        int vc = 0) {
    return add_dft(c, where, freq.data(), freq.size(), include_dV_and_interp_weights, stored_weight,
                   chunk_next, sqrt_dV_and_interp_weights, extra_weight, use_centered_grid, vc);
  }
  dft_chunk<T> *add_dft_pt(component c, const vec &where, double freq_min, double freq_max,
                           int Nfreq) {
    return add_dft(c, where, linspace(freq_min, freq_max, Nfreq), false);
  }
  dft_chunk<T> *add_dft_pt(component c, const vec &where, const std::vector<double> freq) {
    return add_dft(c, where, freq, false);
  }
  dft_chunk<T> *add_dft(const volume_list *where, double freq_min, double freq_max, int Nfreq,
                        bool include_dV = true) {
    return add_dft(where, linspace(freq_min, freq_max, Nfreq), include_dV);
  }
  dft_chunk<T> *add_dft(const volume_list *where, const std::vector<double> freq,
                        bool include_dV = true);
  void update_dfts();
  dft_flux<T> add_dft_flux(const volume_list *where, const double *freq, size_t Nfreq,
                           bool use_symmetry = true, bool centered_grid = true);
  dft_flux<T> add_dft_flux(const volume_list *where, const std::vector<double> freq,
                           bool use_symmetry = true, bool centered_grid = true) {
    return add_dft_flux(where, freq.data(), freq.size(), use_symmetry, centered_grid);
  }
  dft_flux<T> add_dft_flux(const volume_list *where, double freq_min, double freq_max, int Nfreq,
                           bool use_symmetry = true, bool centered_grid = true) {
    return add_dft_flux(where, linspace(freq_min, freq_max, Nfreq), use_symmetry, centered_grid);
  }
  dft_flux<T> add_dft_flux(direction d, const volume &where, double freq_min, double freq_max,
                           int Nfreq, bool use_symmetry = true, bool centered_grid = true) {
    return add_dft_flux(d, where, linspace(freq_min, freq_max, Nfreq), use_symmetry, centered_grid);
  }
  dft_flux<T> add_dft_flux(direction d, const volume &where, const std::vector<double> freq,
                           bool use_symmetry = true, bool centered_grid = true) {
    return add_dft_flux(d, where, freq.data(), freq.size(), use_symmetry, centered_grid);
  }
  dft_flux<T> add_dft_flux(direction d, const volume &where, const double *freq, size_t Nfreq,
                           bool use_symmetry = true, bool centered_grid = true);
  dft_flux<T> add_dft_flux_box(const volume &where, double freq_min, double freq_max, int Nfreq);
  dft_flux<T> add_dft_flux_box(const volume &where, const std::vector<double> freq);
  dft_flux<T> add_dft_flux_plane(const volume &where, double freq_min, double freq_max, int Nfreq);
  dft_flux<T> add_dft_flux_plane(const volume &where, const std::vector<double> freq);

  // a "mode monitor" is just a dft_flux with symmetry reduction turned off.
  dft_flux<T> add_mode_monitor(direction d, const volume &where, double freq_min, double freq_max,
                               int Nfreq, bool centered_grid = true) {
    return add_mode_monitor(d, where, linspace(freq_min, freq_max, Nfreq), centered_grid);
  }
  dft_flux<T> add_mode_monitor(direction d, const volume &where, const std::vector<double> freq, bool centered_grid = true) {
    return add_mode_monitor(d, where, freq.data(), freq.size(), centered_grid);
  }
  dft_flux<T> add_mode_monitor(direction d, const volume &where, const double *freq, size_t Nfreq, bool centered_grid = true);

  dft_fields<T> add_dft_fields(component *components, int num_components, const volume where,
                               double freq_min, double freq_max, int Nfreq,
                               bool use_centered_grid = true) {
    return add_dft_fields(components, num_components, where, linspace(freq_min, freq_max, Nfreq),
                          use_centered_grid);
  }
  dft_fields<T> add_dft_fields(component *components, int num_components, const volume where,
                               const std::vector<double> freq, bool use_centered_grid = true) {
    return add_dft_fields(components, num_components, where, freq.data(), freq.size(),
                          use_centered_grid);
  }
  dft_fields<T> add_dft_fields(component *components, int num_components, const volume where,
                               const double *freq, size_t Nfreq, bool use_centered_grid = true);

  /********************************************************/
  /* process_dft_component is an intermediate-level       */
  /* routine that serves as a common back end for several */
  /* operations involving DFT fields (specifically,       */
  /* writing DFT fields to HDF5 files, fetching arrays    */
  /* of DFT fields, and  evaluating overlap integrals     */
  /* flux and mode fields.)                               */
  /********************************************************/
  std::complex<double> process_dft_component(dft_chunk<T> **chunklists, int num_chunklists,
                                             int num_freq, component c, const char *HDF5FileName,
                                             std::complex<double> **field_array = 0, int *rank = 0,
                                             size_t *dims = 0, direction *dirs = 0,
                                             void *mode1_data = 0, void *mode2_data = 0,
                                             component c_conjugate = Ex, bool *first_component = 0,
                                             bool retain_interp_weights = true);

  // output DFT fields to HDF5 file
  void output_dft_components(dft_chunk<T> **chunklists, int num_chunklists, volume dft_volume,
                             const char *HDF5FileName);

  void output_dft(dft_flux<T> flux, const char *HDF5FileName);
  void output_dft(dft_force<T> force, const char *HDF5FileName);
  void output_dft(dft_near2far<T> n2f, const char *HDF5FileName);
  void output_dft(dft_fields<T> fdft, const char *HDF5FileName);

  // get array of DFT field values
  std::complex<double> *get_dft_array(dft_flux<T> flux, component c, int num_freq, int *rank,
                                      size_t dims[3]);
  std::complex<double> *get_dft_array(dft_fields<T> fdft, component c, int num_freq, int *rank,
                                      size_t dims[3]);
  std::complex<double> *get_dft_array(dft_force<T> force, component c, int num_freq, int *rank,
                                      size_t dims[3]);
  std::complex<double> *get_dft_array(dft_near2far<T> n2f, component c, int num_freq, int *rank,
                                      size_t dims[3]);

  // overlap integrals between eigenmode fields and DFT flux fields
  void get_overlap(void *mode1_data, void *mode2_data, dft_flux<T> flux, int num_freq,
                   std::complex<double> overlaps[2]);
  void get_mode_flux_overlap(void *mode_data, dft_flux<T> flux, int num_freq,
                             std::complex<double> overlaps[2]);
  void get_mode_mode_overlap(void *mode1_data, void *mode2_data, dft_flux<T> flux,
                             std::complex<double> overlaps[2]);

  dft_energy<T> add_dft_energy(const volume_list *where, double freq_min, double freq_max, int Nfreq) {
    return add_dft_energy(where, linspace(freq_min, freq_max, Nfreq));
  }
  dft_energy<T> add_dft_energy(const volume_list *where, const std::vector<double> freq) {
    return add_dft_energy(where, freq.data(), freq.size());
  }
  dft_energy<T> add_dft_energy(const volume_list *where, const double *freq, size_t Nfreq);

  // stress.cpp
  dft_force<T> add_dft_force(const volume_list *where, double freq_min, double freq_max, int Nfreq) {
    return add_dft_force(where, linspace(freq_min, freq_max, Nfreq));
  }
  dft_force<T> add_dft_force(const volume_list *where, const std::vector<double> freq) {
    return add_dft_force(where, freq.data(), freq.size());
  }
  dft_force<T> add_dft_force(const volume_list *where, const double *freq, size_t Nfreq);

  // near2far.cpp
  dft_near2far<T> add_dft_near2far(const volume_list *where, double freq_min, double freq_max,
                                   int Nfreq, int Nperiods = 1) {
    return add_dft_near2far(where, linspace(freq_min, freq_max, Nfreq), Nperiods);
  }
  dft_near2far<T> add_dft_near2far(const volume_list *where, const std::vector<double> freq,
                                   int Nperiods = 1) {
    return add_dft_near2far(where, freq.data(), freq.size(), Nperiods);
  }
  dft_near2far<T> add_dft_near2far(const volume_list *where, const double *freq, size_t Nfreq,
                                   int Nperiods = 1);
  // monitor.cpp
  std::complex<double> get_chi1inv(component, direction, const vec &loc, double frequency = 0,
                                   bool parallel = true) const;
  std::complex<double> get_inveps(component c, direction d, const vec &loc,
                                  double frequency = 0) const {
    return get_chi1inv(c, d, loc, frequency);
  }
  std::complex<double> get_eps(const vec &loc, double frequency = 0) const;
  std::complex<double> get_mu(const vec &loc, double frequency = 0) const;
  void get_point(monitor_point *p, const vec &) const;
  monitor_point *get_new_point(const vec &, monitor_point *p = NULL) const;

  std::complex<double> get_field(int c, const vec &loc, bool parallel = true) const;
  std::complex<double> get_field(component c, const vec &loc, bool parallel = true) const;
  double get_field(derived_component c, const vec &loc, bool parallel = true) const;

  // energy_and_flux.cpp
  void synchronize_magnetic_fields();
  void restore_magnetic_fields();
  double energy_in_box(const volume &);
  double electric_energy_in_box(const volume &);
  double magnetic_energy_in_box(const volume &);
  double thermo_energy_in_box(const volume &);
  double total_energy();
  double field_energy_in_box(const volume &);
  double field_energy_in_box(component c, const volume &);
  double field_energy();
  double flux_in_box_wrongH(direction d, const volume &);
  double flux_in_box(direction d, const volume &);
  flux_vol<T> *add_flux_vol(direction d, const volume &where);
  flux_vol<T> *add_flux_plane(const volume &where);
  flux_vol<T> *add_flux_plane(const vec &p1, const vec &p2);
  double electric_energy_max_in_box(const volume &where);
  double modal_volume_in_box(const volume &where);
  double electric_sqr_weighted_integral(double (*deps)(const vec &), const volume &where);
  double electric_energy_weighted_integral(double (*f)(const vec &), const volume &where);

  void set_output_directory(const char *name);
  double count_volume(component);
  // fields.cpp
  bool have_component(component);
  // material.cpp
  double max_eps() const;
  // step.cpp
  void step_boundaries(field_type);

  bool nosize_direction(direction d) const;
  direction normal_direction(const volume &where) const;

  // casimir.cpp
  std::complex<double> casimir_stress_dct_integral(direction dforce, direction dsource, double mx,
                                                   double my, double mz, field_type ft,
                                                   volume where, bool is_bloch = false);

  void set_solve_cw_omega(std::complex<double> omega);
  void unset_solve_cw_omega();

private:
  int synchronized_magnetic_fields; // count number of nested synchs
  double last_wall_time;
#define MEEP_TIMING_STACK_SZ 10
  time_sink working_on, was_working_on[MEEP_TIMING_STACK_SZ];
  double times_spent[Other + 1];
  // fields.cpp
  void figure_out_step_plan();
  // boundaries.cpp
  bool chunk_connections_valid;
  void find_metals();
  void disconnect_chunks();
  void connect_chunks();
  void connect_the_chunks(); // Intended to be ultra-private...
  bool on_metal_boundary(const ivec &);
  ivec ilattice_vector(direction) const;
  bool locate_point_in_user_volume(ivec *, std::complex<double> *phase) const;
  void locate_volume_source_in_user_volume(const vec p1, const vec p2, vec newp1[8], vec newp2[8],
                                           std::complex<double> kphase[8], int &ncopies) const;
  // mympi.cpp
  void boundary_communications(field_type);
  // step.cpp
  void phase_material();
  void step_db(field_type ft);
  void step_source(field_type ft, bool including_integrated = false);
  void update_pols(field_type ft);
  void calc_sources(double tim);
  // sources.cpp
  void add_volume_source_check(component c, const src_time &src, const volume &where,
                               std::complex<double> A(const vec &), std::complex<double> amp,
                               component c0, direction d, int has_tm, int has_te);

public:
  // monitor.cpp
  std::complex<double> get_field(component c, const ivec &iloc, bool parallel = true) const;
  std::complex<double> get_chi1inv(component, direction, const ivec &iloc, double frequency = 0,
                                   bool parallel = true) const;
  // boundaries.cpp
  bool locate_component_point(component *, ivec *, std::complex<double> *) const;
  // time.cpp
  void am_now_working_on(time_sink);
  void finished_working();
  void reset_timers();
};

template<class T>
class flux_vol {
public:
  flux_vol(fields<T> *f_, direction d_, const volume &where_) : where(where_) {
    f = f_;
    d = d_;
    cur_flux = cur_flux_half = 0;
    next = f->fluxes;
    f->fluxes = this;
  }
  ~flux_vol() { delete next; }

  void update_half() {
    cur_flux_half = flux_wrongE();
    if (next) next->update_half();
  }
  void update() {
    cur_flux = (flux_wrongE() + cur_flux_half) * 0.5;
    if (next) next->update();
  }

  double flux() { return cur_flux; }

  flux_vol *next;

private:
  double flux_wrongE() { return f->flux_in_box_wrongH(d, where); }
  fields<T> *f;
  direction d;
  volume where;
  double cur_flux, cur_flux_half;
};

// The following is a utility function to parse the executable name use it
// to come up with a directory name, avoiding overwriting any existing
// directory, unless the source file hasn't changed.

const char *make_output_directory(const char *exename, const char *jobname = NULL);
char *make_output_directory(); // make temporary directory
void trash_output_directory(const char *dirname);
void delete_directory(const char *path);
FILE *create_output_file(const char *dirname, const char *fname);

int do_harminv(std::complex<double> *data, int n, double dt, double fmin, double fmax, int maxbands,
               std::complex<double> *amps, double *freq_re, double *freq_im, double *errors = NULL,
               double spectral_density = 1.1, double Q_thresh = 50, double rel_err_thresh = 1e20,
               double err_thresh = 0.01, double rel_amp_thresh = -1, double amp_thresh = -1);

std::complex<double> *
make_casimir_gfunc(double T, double dt, double sigma, field_type ft,
                   std::complex<double> (*eps_func)(std::complex<double> omega) = 0,
                   double Tfft = 0);

std::complex<double> *make_casimir_gfunc_kz(double T, double dt, double sigma, field_type ft);

// random number generation: random.cpp
void set_random_seed(unsigned long seed);
void restore_random_seed();
double uniform_random(double a, double b);          // uniform random in [a,b]
double gaussian_random(double mean, double stddev); // normal random with given mean and stddev
int random_int(int a, int b);                       // uniform random in [a,b)

// Bessel function (in initialize.cpp)
double BesselJ(int m, double kr);

// analytical Green's functions (in near2far.cpp); upon return,
// EH[0..5] are set to the Ex,Ey,Ez,Hx,Hy,Hz field components at x
// from a c0 source of amplitude f0 at x0.
void green2d(std::complex<double> *EH, const vec &x, double freq, double eps, double mu,
             const vec &x0, component c0, std::complex<double> f0);
void green3d(std::complex<double> *EH, const vec &x, double freq, double eps, double mu,
             const vec &x0, component c0, std::complex<double> f0);

// non-class methods for working with mpb eigenmode data
//
void destroy_eigenmode_data(void *vedata, bool destroy_mdata = true);
std::complex<double> eigenmode_amplitude(void *vedata, const vec &p, component c);
double get_group_velocity(void *vedata);
vec get_k(void *vedata);

double linear_interpolate(double rx, double ry, double rz, double *data,
                          int nx, int ny, int nz, int stride);

// binary tree class for importing layout of chunk partition
class binary_partition {
public:
  binary_partition(int _id);
  binary_partition(direction _split_dir, double _split_pos);
  direction split_dir;
  double split_pos;
  int proc_id;
  binary_partition *left, *right;
};

} /* namespace meep */

#endif /* MEEP_H */
