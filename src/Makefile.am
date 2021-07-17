lib_LTLIBRARIES = libmeep.la
include_HEADERS = meep.hpp
pkginclude_HEADERS = meep/mympi.hpp meep/vec.hpp meep/meep-config.h meepgeom.hpp material_data.hpp adjust_verbosity.hpp

AM_CPPFLAGS = -I$(top_srcdir)/src

BUILT_SOURCES = sphere-quad.h step_generic_stride1.cpp meep/meep-config.h

HDRS = meep.hpp meep_internals.hpp meep/mympi.hpp meep/vec.hpp	\
bicgstab.hpp meepgeom.hpp material_data.hpp adjust_verbosity.hpp

libmeep_la_SOURCES = array_slice.cpp anisotropic_averaging.cpp 		\
bands.cpp boundaries.cpp bicgstab.cpp casimir.cpp 	\
cw_fields.cpp dft.cpp dft_ldos.cpp energy_and_flux.cpp 	\
fields.cpp fields_dump.cpp loop_in_chunks.cpp h5fields.cpp h5file.cpp 	\
initialize.cpp integrate.cpp integrate2.cpp material_data.cpp monitor.cpp mympi.cpp 	\
multilevel-atom.cpp near2far.cpp output_directory.cpp random.cpp 	\
sources.cpp step.cpp step_db.cpp stress.cpp structure.cpp structure_dump.cpp		\
susceptibility.cpp time.cpp update_eh.cpp mpb.cpp update_pols.cpp 	\
vec.cpp step_generic.cpp meepgeom.cpp GDSIIgeom.cpp $(HDRS) $(BUILT_SOURCES)

SUBDIRS = support
libmeep_la_LIBADD = support/libsupport.la

libmeep_la_LDFLAGS = -version-info @SHARED_VERSION_INFO@

PRELUDE = "/* This file was automatically generated --- DO NOT EDIT */"

noinst_PROGRAMS = sphere_quad
sphere_quad_SOURCES = sphere-quad.cpp
SPHERE_QUAD = $(top_builddir)/src/sphere_quad
sphere-quad.h:
	make sphere_quad
	(echo $(PRELUDE); echo; $(SPHERE_QUAD)) > $@

step_generic_stride1.cpp: step_generic.cpp
	(echo $(PRELUDE); echo; sed 's/LOOP_OVER/S1LOOP_OVER/g' $(top_srcdir)/src/step_generic.cpp | sed 's/step_curl/step_curl_stride1/' | sed 's/step_update_EDHB/step_update_EDHB_stride1/' | sed 's/step_beta/step_beta_stride1/') > $@

MAINTAINERCLEANFILES = $(BUILT_SOURCES)
