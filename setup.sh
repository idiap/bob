# Andre Anjos <andre.anjos@idiap.ch>
# Wed 30 Jun 2010 16:22:58 CEST

# These are some variables you may want to configure
externals=/idiap/group/torch5spro/nightlies/externals
externals_version=v1

# From this point onwards, I'll do the work
prog=${BASH_SOURCE[0]}
dir=$(dirname ${prog})
${dir}/bin/setup.py ${prog} $* --check-options
if [ "$?" = "0" ]; then
  [ -e ${externals}/tools/setup.py ] && eval `${externals}/tools/setup.py --version=${externals_version}`
  eval `${dir}/bin/setup.py $*`
fi
