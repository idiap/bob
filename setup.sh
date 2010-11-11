# Andre Anjos <andre.anjos@idiap.ch>
# Wed 30 Jun 2010 16:22:58 CEST

# From this point onwards, I'll do the work
prog=${BASH_SOURCE[0]}
dir=$(dirname ${prog})
${dir}/bin/setup.py ${prog} $* --check-options
if [ "$?" = "0" ]; then
  eval `${dir}/bin/setup.py $*`
fi
