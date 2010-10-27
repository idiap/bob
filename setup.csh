# Andre Anjos <andre.anjos@idiap.ch>
# Wed 30 Jun 2010 16:22:58 CEST
set args=($_)
set prog=${args[2]}
set dir=`dirname ${prog}`
${dir}/bin/setup.py ${prog} --base-dir=${dir} $* --check-options
if ( "$?" == "0" ) then
  set externals=/idiap/group/torch5spro/nightlies/externals/tools/setup.py
  if ( -e ${externals} ) then
    eval `${externals} --csh`
  endif
  eval `${dir}/bin/setup.py $args[3-] --csh`
endif
