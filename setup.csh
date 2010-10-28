# Andre Anjos <andre.anjos@idiap.ch>
# Wed 30 Jun 2010 16:22:58 CEST

# These are some variables you may want to configure
set externals=/idiap/group/torch5spro/nightlies/externals
set externals_version=last

# From this point onwards, I'll do the work
set args=($_)
set prog=${args[2]}
set dir=`dirname ${prog}`
${dir}/bin/setup.py ${prog} --base-dir=${dir} $* --check-options
if ( "$?" == "0" ) then
  if ( -e ${externals}/tools/setup.py ) then
    eval `${externals}/tools/setup.py --csh --version=${externals_version}`
  endif
  eval `${dir}/bin/setup.py $args[3-] --csh`
endif
