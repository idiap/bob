# Andre Anjos <andre.anjos@idiap.ch>
# Wed 30 Jun 2010 16:22:58 CEST
prog=${BASH_SOURCE[0]}
dir=$(dirname ${prog})
${dir}/bin/setup.py ${prog} --base-dir=${dir} $* --check-options
if [ "$?" = "0" ]; then
  externals=/idiap/group/torch5spro/nightlies/externals/tools/setup.py
  [ -e ${externals} ] && eval `${externals}`
  eval `${dir}/bin/setup.py --base-dir=${dir} $*`
fi
