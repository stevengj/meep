#include "meep_internals.hpp"
#include "config.h"

#ifdef HAVE_MPI
#ifdef NEED_UNDEF_SEEK_FOR_MPI
// undef'ing SEEK_* is needed for MPICH, possibly other MPI versions
#undef SEEK_SET
#undef SEEK_END
#undef SEEK_CUR
#endif
#include <mpi.h>
#endif

namespace meep {

// data structure for sending source information from one chunk to another
struct srcpt_info {
  std::complex<double> A; // amplitude
  ptrdiff_t index; // index in chunk's fields array
  size_t src_time_id;
  int chunk_idx;
  int c; // component
};

// comparison function for sorting srcpt_info lexicographically
// by (processor, src_time_id, chunk_idx, c)
struct srcpt_info_compare {
  fields_chunk **chunks;
  bool operator() (srcpt_info a, srcpt_info b) {
    int aproc = chunks[a.chunk_idx]->n_proc();
    int bproc = chunks[b.chunk_idx]->n_proc();
    return (aproc != bproc ? aproc < bproc :
            (a.src_time_id != b.src_time_id ? a.src_time_id < b.src_time_id :
             (a.chunk_idx != b.chunk_idx ? a.chunk_idx < b.chunk_idx :
              a.c < b.c)));
  }
};

void fields::fix_boundary_sources() {
  am_now_working_on(Connecting);

  std::vector<srcpt_info> boundarysources;

  // find all not-owned source points and figure out in which chunk
  // they are actually supposed to be located, storing info in boundarysources.
  for (int i = 0; i < num_chunks; i++) {
    FOR_FIELD_TYPES(ft) {
      for (src_vol *src = chunks[i]->sources[ft]; src; src = src->next)
        if (src->needs_boundary_fix) {
          for (size_t ipt = 0; ipt < src->npts; ++ipt) {
            component c = src->c;
            ivec here = chunks[i]->gv.iloc(c, src->index[ipt]);
            if (!chunks[i]->gv.owns(here) && src->A[ipt] != 0.0) {
              if (src->t->id == 0) abort("bug: fix_boundary_sources called for non-registered source");

              // find the chunk that owns this point, similar to logic in boundaries.cpp
              std::complex<double> thephase;
              if (locate_component_point(&c, &here, &thephase) && !on_metal_boundary(here)) {
                for (int j = 0; j < num_chunks; j++)
                  if (chunks[j]->gv.owns(here)) {
                    srcpt_info s = { src->A[ipt]*conj(thephase), chunks[j]->gv.index(c, here),  src->t->id, chunks[j]->chunk_idx, c };
                    boundarysources.push_back(s);
                    break;
                  }
              }
              src->A[ipt] = 0.0; // will no longer be needed
            }
          }
          src->needs_boundary_fix = false;
        }
    }
  }

  // we need each process's data to be contiguous
  srcpt_info_compare compare = {chunks};
  std::sort(boundarysources.begin(), boundarysources.end(), compare);

  // collect 2d (row-major) arrays offsets and numcomm,
  // where numcomm[i,j] is the number of srcpt_info items
  // to be set from process i to process j, and offsets[i,j]
  // is the corresponding offset in the boundarysources input.
  int p = my_rank();
  int P = count_processors();
  std::vector<size_t> offsets(P * P, size_t(0));
  std::vector<size_t> numcomm_(P * P, size_t(0));
  size_t idx0 = 0;
  int p0 = 0;
  for (size_t idx = 0; idx < boundarysources.size(); ++idx) {
    int pidx = chunks[boundarysources[idx].chunk_idx]->n_proc();
    if (pidx != p0) {
      offsets[p*P + p0] = idx0;
      numcomm_[p*P + p0] = idx - idx0;
      p0 = pidx;
      idx0 = idx;
    }
  }
  offsets[p*P + p0] = idx0;
  numcomm_[p*P + p0] = boundarysources.size() - idx0;

  // collect the numcomm data from all processes
  std::vector<size_t> numcomm(P * P, size_t(0));
  sum_to_all(numcomm_.data(), numcomm.data(), P*P);

#ifdef HAVE_MPI
  // declare an MPI datatype mirroring srcpt_info, so that we can send/receive srcpt_info arrays
  int srcpt_info_blocklengths[5] = {2,1,1,1,1};
  MPI_Datatype srcpt_info_types[5] = {MPI_DOUBLE, sizeof(ptrdiff_t) == 4 ? MPI_INT : MPI_LONG_LONG, sizeof(size_t) == 4 ? MPI_UNSIGNED : MPI_UNSIGNED_LONG_LONG, MPI_INT, MPI_INT};
  MPI_Aint srcpt_info_offsets[5] = { offsetof(srcpt_info,A), offsetof(srcpt_info,index), offsetof(srcpt_info,chunk_idx), offsetof(srcpt_info,src_time_id), offsetof(srcpt_info,c) };
  MPI_Datatype mpi_srcpt_info;
  MPI_Type_create_struct(5, srcpt_info_blocklengths, srcpt_info_offsets, srcpt_info_types, &mpi_srcpt_info);
  MPI_Type_commit(&mpi_srcpt_info);
#endif

for (int psrc = 0; psrc < P; ++psrc)
  for (int pdest = 0; pdest < P; ++pdest) {
    size_t N = numcomm[psrc*P + pdest];
    if (N == 0) continue;
    if (pdest == p) {
      srcpt_info *srcpts;
#ifdef HAVE_MPI
      if (psrc != p) {
        srcpts = new srcpt_info[N];
        MPI_Status status;
        MPI_Recv(srcpts, N, mpi_srcpt_info, psrc, psrc*P + pdest, mycomm, &status);
      }
      else
#endif
      srcpts = boundarysources.data() + offsets[psrc*P + pdest];
      int chunk_idx = srcpts[0].chunk_idx;
      size_t src_time_id = srcpts[0].src_time_id;
      int c = srcpts[0].c;
      size_t idx0 = 0;
      for (size_t idx = 0; idx <= N; ++idx) {
        if (idx == N || srcpts[idx].chunk_idx != chunk_idx || srcpts[idx].src_time_id != src_time_id || srcpts[idx].c != c) {
          std::vector<ptrdiff_t> idx_arr(idx - idx0);
          std::vector<std::complex<double> > amp_arr(idx - idx0);
          for (size_t i = idx0; i < idx; ++i) {
            idx_arr[i-idx0] = srcpts[i].index;
            amp_arr[i-idx0] = srcpts[i].A;
          }
          sourcedata srcdata = { (component) c, idx_arr, chunk_idx, amp_arr };
          src_time *srctime = lookup_src_time(src_time_id);
          if (srctime == NULL) abort("bug: unknown src_time_id (missing registration?)");
          add_srcdata(srcdata, srctime);
          if (idx < N) {
            chunk_idx = srcpts[idx].chunk_idx;
            src_time_id = srcpts[idx].src_time_id;
            c = srcpts[idx].c;
            idx0 = idx;
          }
        }
      }

      if (psrc != p) delete[] srcpts;
    }
#ifdef HAVE_MPI
    else if (psrc == p) {
        srcpt_info *srcpts = boundarysources.data() + offsets[psrc*P + pdest];
        MPI_Send(srcpts, N, mpi_srcpt_info, pdest, psrc*P + pdest, mycomm);
    }
#endif
  }

#ifdef HAVE_MPI
  MPI_Type_free(&mpi_srcpt_info);
#endif

  finished_working();
}

} // namespace meep
