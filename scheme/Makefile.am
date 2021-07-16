bin_PROGRAMS = meep
EXTRA_DIST = meep.scm.in meep-enums.scm casimir.scm materials.scm meep_wrap.patch examples

LIBMEEP = $(top_builddir)/src/libmeep.la

HDRS = meep-ctl.hpp meep-ctl-const.hpp meep-ctl-swig.hpp

LIBHDRS = $(top_srcdir)/src/meep.hpp $(top_srcdir)/src/meep/vec.hpp $(top_srcdir)/src/meep/mympi.hpp
CTLHDRS = $(LIBHDRS) $(srcdir)/meep-ctl.hpp $(top_builddir)/config.h $(srcdir)/meep-ctl-const.hpp ctl-io.h

meep_SOURCES = meep.cpp structure.cpp meep_wrap.cxx $(HDRS) meep.i meep_op_renames.i meep_renames.i meep_enum_renames.i meep_swig_bug_workaround.i
nodist_meep_SOURCES = main.cpp geom.cpp ctl-io.cpp ctl-io.h ctl-io.i
meep_LDADD = $(LIBMEEP) @LIBCTL_LIBS@
meep_CPPFLAGS = $(AM_CPPFLAGS) -Wno-unused-parameter -Wno-unused-variable -Wno-empty-body

BUILT_SOURCES = $(nodist_meep_SOURCES) meep_renames.i meep_enum_renames.i meep_swig_bug_workaround.i meep-enums.scm meep_wrap.cxx

# manual dependency hack to force the build order in certain cases
# ... for some reason automake's automatic dependencies are not working here
meep.o: meep.cpp meep-enums.scm $(CTLHDRS)
structure.o: structure.cpp $(CTLHDRS)
meep_wrap.o: meep_wrap.cxx $(srcdir)/meep-ctl-swig.hpp $(CTLHDRS)

if WITH_LIBCTL
if MAINTAINER_MODE
meep_wrap.cxx: meep.i meep_op_renames.i meep_enum_renames.i meep_renames.i ctl-io.i meep-ctl-swig.hpp meep_swig_bug_workaround.i $(LIBHDRS)
	$(SWIG) -I$(top_srcdir)/src -c++ -guile -o $@ $(srcdir)/meep.i
	patch -p0 $@ < $(srcdir)/meep_wrap.patch
	sed -i.bak '/(SCM_SMOB_PREDICATE(swig_tag, smob) || SCM_SMOB_PREDICATE(swig_collectable_tag, smob))/ s/swig_collectable_tag, smob)/swig_collectable_tag, smob) || SCM_SMOB_PREDICATE((swig_collectable_tag \& ~0xff00), (smob))/' $@;
else
meep_wrap.cxx:
	echo "#error need --with-maintainer-mode to generate this file" 1>&2
	exit 1
endif
else
meep_wrap.cxx:
	echo "#error need --with-libctl to generate this file" > $@
endif

# workaround missing namespace prefix in swig
meep_renames.i: $(LIBHDRS)
	(echo "// AUTOMATICALLY GENERATED -- DO NOT EDIT"; perl -pe 's/^ *class +([A-Za-z_0-9:]*)( *| *:[^{]*){.*$$/%rename(meep_\1) meep::\1;/' $(LIBHDRS) | grep "%rename" | sed '/meep_fields_chunk/d ; /meep_comms_manager/d ; /meep_comms_key_hash_fn/d ; /meep_timing_scope/d' | sort -u; echo; grep -hv typedef $(LIBHDRS) | perl -pe 's/(inline|const|extern|static) +//g' | perl -pe 's/^[A-Za-z_0-9:<>]+[* ]+([A-Za-z_0-9:]*) *\(.*$$/%rename(meep_\1) meep::\1;/' | grep "%rename" | sed '/choose_chunkdivision/d ; /meep_create_comms_manager/d' | sort -u; ) > $@

# work around bug in swig, where it doesn't prepend namespace to friend funcs
meep_swig_bug_workaround.i: $(LIBHDRS)
	(echo "// AUTOMATICALLY GENERATED -- DO NOT EDIT"; grep -h friend $(LIBHDRS) | perl -pe 's/^ *friend +[A-Za-z_0-9:<>]+[* ]+([A-Za-z_0-9:]*) *\(.*$$/%ignore \1;/' | grep "%ignore" | sort -u;) > $@

meep_enum_renames.i: $(LIBHDRS)
	(echo "// AUTOMATICALLY GENERATED -- DO NOT EDIT"; for f in $(LIBHDRS); do egrep "^enum" $$f | perl -pe 's/enum +([A-Za-z_0-9:]+).*$$/\1/g' | while read enum; do cat $$f | tr -d '\n' | perl -pe 's/.*enum +'$${enum}' *{([^}]*)}.*/\1/g' | perl -pe 's/= *[0-9]+//g' |tr -d ' \t' | tr ',' '\n' | perl -pe 's/^(.*)$$/'"%rename(meep_$${enum}_\1) meep::\1;/g"; echo; done; done;) > $@

meep-enums.scm: meep_enum_renames.i
	(perl -pe 's/%rename\(([A-Za-z0-9_]*)\) *([A-Za-z0-9:_]*);$$/(define \2 (\1))/' meep_enum_renames.i | perl -pe 's/[A-Za-z0-9:_]*:://g' | perl -pe 's/_/-/g' | perl -pe 's,//,;,'; echo "(define Centered Dielectric)") > $@

##############################################################################
#      stuff below is based on example Makefile.in included with libctl
##############################################################################

# what is printed out when invoking your program with --version:
VERSION_STRING = "Meep @VERSION@, Copyright (C) 2005-2021 Massachusetts Institute of Technology."

MY_DEFS = -DHAVE_CTL_HOOKS=1 -DHAVE_CTL_EXPORT_HOOK=1

PROGRAM_NAME = meep
SPECIFICATION_FILE = $(PROGRAM_NAME).scm
nodist_pkgdata_DATA = $(SPECIFICATION_FILE) meep-enums.scm casimir.scm materials.scm

##############################################################################

AM_CPPFLAGS = $(MY_DEFS) $(CTL_DEFS) $(CTL_H_CPPFLAG) @LIBCTL_CPPFLAGS@ -I$(top_srcdir)/src

# libctl install. dir., e.g. /usr/local/share/libctl
LIBCTL_DIR = @LIBCTL_DIR@

# gen-ctl-io program
GEN_CTL_IO = @GEN_CTL_IO@

##############################################################################

CTL_DEFS = -DCTL_SCM='"'$(LIBCTL_DIR)/base/ctl.scm'"' \
       -DINCLUDE_SCM='"'$(LIBCTL_DIR)/base/include.scm'"' \
          -DSPEC_SCM='"'$(pkgdatadir)/$(SPECIFICATION_FILE)'"' \
    -DVERSION_STRING='"'$(VERSION_STRING)'"'

##############################################################################

ctl-io.cpp: $(SPECIFICATION_FILE) $(LIBCTL_DIR)/utils/geom.scm ctl-io.h
	$(GEN_CTL_IO) --cxx --code -o $@ $(SPECIFICATION_FILE) $(LIBCTL_DIR)

ctl-io.h: $(SPECIFICATION_FILE) $(LIBCTL_DIR)/utils/geom.scm
	$(GEN_CTL_IO) --cxx --header -o $@ $(SPECIFICATION_FILE) $(LIBCTL_DIR)

ctl-io.i: $(SPECIFICATION_FILE) $(LIBCTL_DIR)/utils/geom.scm
	$(GEN_CTL_IO) --cxx --swig -o $@ $(SPECIFICATION_FILE) $(LIBCTL_DIR)

main.cpp: $(LIBCTL_DIR)/base/main.c ctl-io.h
	cp -f $(LIBCTL_DIR)/base/main.c $@

geom.cpp: $(LIBCTL_DIR)/utils/geom.c ctl-io.h
	cp -f $(LIBCTL_DIR)/utils/geom.c $@

clean-local:
	rm -f $(nodist_meep_SOURCES) meep-enums.scm ctl-io.* main.* geom.*

maintainer-clean-local:
	rm -f $(BUILT_SOURCES)
