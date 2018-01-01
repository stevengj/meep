/* Copyright (C) 2005-2015 Massachusetts Institute of Technology
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

/* given an arbitrary field configuration F, compute the coefficients
   in an expansion of F as a linear combination of normal modes
   of the geometry as computed by mpb.
*/


#include <stdio.h>
#include <string.h>
#include <math.h>

#include "meep_internals.hpp"

using namespace std;

typedef complex<double> cdouble;

namespace meep {

// prototype for optional user-supplied function to provide an
// initial estimate of the wavevector of band #band at frequency freq
typedef vec (*kpoint_func)(void *user_data, double freq, int band);

typedef struct {

  // information related to the volume covered by the
  // array slice (its size, etcetera) 
  // these fields are filled in by get_array_slice_dimensions
  // if the data parameter is non-null
  ivec min_corner, max_corner;
  int num_chunks;
  int rank;
  direction ds[3];
  int slice_size;

  // the function to output and related info (offsets for averaging, etc.)
  // note: either fun *or* rfun should be non-NULL (not both)
  field_function fun;
  field_rfunction rfun;
  void *fun_data;
  std::vector<component> components;

  void *vslice;

  // temporary internal storage buffers
  component *cS;
  cdouble *ph;
  cdouble *fields;
  int *offsets;

  int ninveps;
  component inveps_cs[3];
  direction inveps_ds[3];

  int ninvmu;
  component invmu_cs[3];
  direction invmu_ds[3];

} mode_projection_data;

#define UNUSED(x) (void) x // silence compiler warnings

/***************************************************************/
/* callback function passed to loop_in_chunks to evaluate the  */
/* projection of the user-specified fields onto one or more    */
/* normal modes as computed by mpb                             */
/***************************************************************/
static void mode_projection_chunkloop(fields_chunk *fc, 
                                      int ichnk, component cgrid,
				      ivec is, ivec ie,
				      vec s0, vec s1, vec e0, vec e1,
				      double dV0, double dV1,
				      ivec shift, complex<double> shift_phase,
				      const symmetry &S, int sn,
				      void *data_)
{
  UNUSED(ichnk);UNUSED(cgrid);UNUSED(s0);UNUSED(s1);UNUSED(e0);UNUSED(e1);
  UNUSED(dV0);UNUSED(dV1);UNUSED(shift_phase); UNUSED(fc);
  mode_projection_data *data = (mode_projection_data *) data_;

  ivec isS = S.transform(is, sn) + shift;
  ivec ieS = S.transform(ie, sn) + shift;
  data->min_corner = min(data->min_corner, min(isS, ieS));
  data->max_corner = max(data->max_corner, max(isS, ieS));
  data->num_chunks++;

}

/***************************************************************/
/* callback function passed to loop_in_chunks to fill array slice */
/***************************************************************/
static void get_array_slice_chunkloop(fields_chunk *fc, int ichnk, component cgrid,
	      			  ivec is, ivec ie, vec s0, vec s1, vec e0, vec e1,
				  double dV0, double dV1,
	  			  ivec shift, complex<double> shift_phase,
				  const symmetry &S, int sn, void *data_)
{
  UNUSED(ichnk);UNUSED(cgrid);UNUSED(s0);UNUSED(s1);UNUSED(e0);UNUSED(e1);
  UNUSED(dV0);UNUSED(dV1);
  array_slice_data *data = (array_slice_data *) data_;

  //-----------------------------------------------------------------------//
  // Find output chunk dimensions and strides, etc.

  int count[3]={1,1,1}, offset[3]={0,0,0}, stride[3]={1,1,1};

  ivec isS = S.transform(is, sn) + shift;
  ivec ieS = S.transform(ie, sn) + shift;
  
  // figure out what yucky_directions (in LOOP_OVER_IVECS)
  // correspond to what directions in the transformed vectors (in output).
  ivec permute(zero_ivec(fc->gv.dim));
  for (int i = 0; i < 3; ++i) 
    permute.set_direction(fc->gv.yucky_direction(i), i);
  permute = S.transform_unshifted(permute, sn);
  LOOP_OVER_DIRECTIONS(permute.dim, d)
    permute.set_direction(d, abs(permute.in_direction(d)));
  
  // compute the size of the chunk to output, and its strides etc.
  for (int i = 0; i < data->rank; ++i) {
    direction d = data->ds[i];
    int isd = isS.in_direction(d), ied = ieS.in_direction(d);
    count[i] = abs(ied - isd) / 2 + 1;
    if (ied < isd) offset[permute.in_direction(d)] = count[i] - 1;
  }
  for (int i = 0; i < data->rank; ++i) {
    direction d = data->ds[i];
    int j = permute.in_direction(d);
    for (int k = i + 1; k < data->rank; ++k) stride[j] *= count[k];
    offset[j] *= stride[j];
    if (offset[j]) stride[j] *= -1;
  }
  
  //-----------------------------------------------------------------------//
  // Compute the function to output, exactly as in fields::integrate.
  int *off = data->offsets;
  component *cS = data->cS;
  complex<double> *fields = data->fields, *ph = data->ph;
  const component *iecs = data->inveps_cs;
  const direction *ieds = data->inveps_ds;
  int ieos[6];
  const component *imcs = data->invmu_cs;
  const direction *imds = data->invmu_ds;
  int imos[6];
  int num_components=data->components.size();
  
  double *slice=0;
  cdouble *zslice=0;
  bool complex_data = (data->rfun==0);
  if (complex_data)
   zslice = (cdouble *)data->vslice;
  else
   slice = (double *)data->vslice;

  for (int i = 0; i < num_components; ++i) {
    cS[i] = S.transform(data->components[i], -sn);
    if (cS[i] == Dielectric || cS[i] == Permeability)
      ph[i] = 1.0;
    else {
      fc->gv.yee2cent_offsets(cS[i], off[2*i], off[2*i+1]);
      ph[i] = shift_phase * S.phase_shift(cS[i], sn);
    }
  }
  for (int k = 0; k < data->ninveps; ++k)
    fc->gv.yee2cent_offsets(iecs[k], ieos[2*k], ieos[2*k+1]);
  for (int k = 0; k < data->ninvmu; ++k)
    fc->gv.yee2cent_offsets(imcs[k], imos[2*k], imos[2*k+1]);

  vec rshift(shift * (0.5*fc->gv.inva));
  LOOP_OVER_IVECS(fc->gv, is, ie, idx) {
    IVEC_LOOP_LOC(fc->gv, loc);
    loc = S.transform(loc, sn) + rshift;

    for (int i = 0; i < num_components; ++i) {
      if (cS[i] == Dielectric) {
	double tr = 0.0;
	for (int k = 0; k < data->ninveps; ++k) {
	  const realnum *ie = fc->s->chi1inv[iecs[k]][ieds[k]];
	  if (ie) tr += (ie[idx] + ie[idx+ieos[2*k]] + ie[idx+ieos[1+2*k]]
			 + ie[idx+ieos[2*k]+ieos[1+2*k]]);
	  else tr += 4; // default inveps == 1
	}
	fields[i] = (4 * data->ninveps) / tr;
      }
      else if (cS[i] == Permeability) {
	double tr = 0.0;
	for (int k = 0; k < data->ninvmu; ++k) {
	  const realnum *im = fc->s->chi1inv[imcs[k]][imds[k]];
	  if (im) tr += (im[idx] + im[idx+imos[2*k]] + im[idx+imos[1+2*k]]
			 + im[idx+imos[2*k]+imos[1+2*k]]);
	  else tr += 4; // default invmu == 1
	}
	fields[i] = (4 * data->ninvmu) / tr;
      }
      else {
	double f[2];
	for (int k = 0; k < 2; ++k)
	  if (fc->f[cS[i]][k])
	    f[k] = 0.25 * (fc->f[cS[i]][k][idx]
			   + fc->f[cS[i]][k][idx+off[2*i]]
			   + fc->f[cS[i]][k][idx+off[2*i+1]]
			   + fc->f[cS[i]][k][idx+off[2*i]+off[2*i+1]]);
	  else
	    f[k] = 0;
	fields[i] = complex<double>(f[0], f[1]) * ph[i];
      }
    }
    int idx2 = ((((offset[0] + offset[1] + offset[2])
                   + loop_i1 * stride[0])
                   + loop_i2 * stride[1])
                   + loop_i3 * stride[2]);

    if (complex_data)
     zslice[idx2] = data->fun(fields, loc, data->fun_data);
    else
     slice[idx2]  = data->rfun(fields, loc, data->fun_data);

  };

}

/***************************************************************/
/***************************************************************/
/***************************************************************/
std::vec<cdouble> fields::get_mode_coefficients(dft_flux flux,
                                                direction d,
                                                const volume where,
                                                std::vec<int> bands,
                                                kpoint_func user_func,
                                                void *user_data)
{
  am_now_working_on(ModeExpansion);

  // create output array
  std::vec<cdouble> coefficients(bands.size());

  // some default inputs for add_eigenmode_source
  component DefaultComponent = Dielectric;
  vec kpoint(0,0,0);
  int parity = 0; /* NO_PARITY */
  bool match_frequency = true
  double eig_resolution = a;
  double eigensolver_tol = 1.0e-7;
  cdouble amp=1.0;

  // loop over all frequencies in the dft_flux
  for(int nfreq=0; nfreq<flux.NFreq; nfreq++)
   { 
     double freq = flux.freq_min + ((double)nfreq)*flux.dfreq;
     continuous_src_time src(freq);

     // loop over caller's list of requested bands
     for(int nband=0; nband<bands.size(); nband++)
      { 
        int band = bands[nband];

        // query user's function (if present) for initial k-point guess
        if (user_func)
         kpoint = user_func(user_data, freq, nband);
       
        // add source
        add_eigenmode_source(DefaultComponent, src, d, where, where,
                             band, kpoint, match_frequency, parity,
                             eig_resolution, eigensolver_tol, amp);
       
        // evaluate projection and normalization integrals
      };
   };

}

} // namespace meep
