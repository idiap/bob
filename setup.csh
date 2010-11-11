# Andre Anjos <andre.anjos@idiap.ch>
# Wed 30 Jun 2010 16:22:58 CEST

# From this point onwards, I'll do the work
set args=($_)
set prog=${args[2]}
set dir=`dirname ${prog}`
${dir}/bin/setup.py ${prog} $* --check-options
if ( "$?" == "0" ) then
  eval `${dir}/bin/setup.py $args[3-] --csh`
endif
