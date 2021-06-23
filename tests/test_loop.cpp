/*
Steps:
    [x] Set up initial benchmark framework
    [ ] Set up initial iterator class
    [ ] Fill in iterator class
    [ ] Debug iterator class
    [ ] Parallelize iterator class
    [ ] Debug parallel version
    [ ] Integrate into main repo
    [ ] Finalize test
    [ ] Try different vectorizations and do basic timing
    [ ] Set up 3 actual test scripts
    [ ] Benchmark added benefit with new scripts
*/

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <omp.h>

#include <meep.hpp>
using namespace meep;
using std::complex;

#define PLOOP_OVER_IVECS(gv, is, ie, idx)			\
for(ptrdiff_t loop_is1 = (is).yucky_val(0), loop_is2 = (is).yucky_val(1),                          \
                 loop_is3 = (is).yucky_val(2), loop_n1 = ((ie).yucky_val(0) - loop_is1) / 2 + 1,   \
                 loop_n2 = ((ie).yucky_val(1) - loop_is2) / 2 + 1,                                 \
                 loop_n3 = ((ie).yucky_val(2) - loop_is3) / 2 + 1,                                 \
                 loop_d1 = (gv).yucky_direction(0), loop_d2 = (gv).yucky_direction(1),             \
                 loop_d3 = (gv).yucky_direction(2),                                                \
                 loop_s1 = (gv).stride((meep::direction)loop_d1),                                  \
                 loop_s2 = (gv).stride((meep::direction)loop_d2),                                  \
                 loop_s3 = (gv).stride((meep::direction)loop_d3),                                  \
                 idx0 = (is - (gv).little_corner()).yucky_val(0) / 2 * loop_s1 +                   \
                        (is - (gv).little_corner()).yucky_val(1) / 2 * loop_s2 +                   \
                        (is - (gv).little_corner()).yucky_val(2) / 2 * loop_s3,                    \
                  dummy_first=0;dummy_first<1;dummy_first++)                                       \
_Pragma("omp parallel for collapse(3)")				                                                     \
  for (ptrdiff_t loop_i1 = 0; loop_i1 < loop_n1; loop_i1++)                                        \
    for (ptrdiff_t loop_i2 = 0; loop_i2 < loop_n2; loop_i2++)                                      \
      for (ptrdiff_t loop_i3 = 0; loop_i3 < loop_n3; loop_i3++)                                    \
        for (ptrdiff_t idx = idx0 + loop_i1*loop_s1 + loop_i2*loop_s2 +                            \
           loop_i3*loop_s3, dummy_last=0;dummy_last<1;dummy_last++)

#define PLOOP_OVER_VOL(gv, c, idx)                                                                  \
  PLOOP_OVER_IVECS(gv, (gv).little_corner() + (gv).iyee_shift(c),                                   \
                  (gv).big_corner() + (gv).iyee_shift(c), idx)

double one(const vec &) { return 1.0; }

int test_normal_loop() {
  double a = 5.0;

  grid_volume gv = vol3d(3.0, 3.0, 1.0, a);
  structure s1(gv, one, pml(0.1), meep::identity(), 0);
  fields f(&s1);
  f.add_point_source(Ez, 0.8, 1.6, 0.0, 4.0, vec(1.299, 1.401, 0.0), 1.0);
  f.step();
  for (int i = 0; i < f.num_chunks; i++) {
      master_printf("\n---------------------------------\n");
      master_printf("%d",LOOPS_ARE_STRIDE1(f.chunks[i]->gv));
      master_printf("current chunk: %d\n",i);
      master_printf("\n---------------------------------\n");

      
      int my_thread=omp_get_thread_num();
        if (my_thread == 0)
          master_printf("%td ",idx);
      }
      
  initialize mpi(argc, argv);
  verbosity = 0;
  master_printf("Testing 2D...\n");
  
  if (!test_normal_loop()) abort("error in test_periodic_tm vacuum\n");

  master_printf("\n\n\n");
  return 0;
}