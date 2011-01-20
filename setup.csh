# Andre Anjos <andre.anjos@idiap.ch>
# Thu Jan 20 15:03:22 CET 2011
set args=($_)
set prog=${args[2]}
set dir=`dirname ${prog}`
eval `${dir}/bin/setup.py ${args[3-]} --csh`
