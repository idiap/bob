# Andre Anjos <andre.anjos@idiap.ch>
# Wed 30 Jun 2010 16:22:58 CEST

set build_type=''
set args=($_)
set dir=`dirname ${args[2]}`
if ( ${#args} >= 4 ) then
  switch (${args[3]})
    case "-d":
    case "--debug":
      set build_type=' --debug'
      breaksw
    default:
      echo "usage: source ./setup.csh -[hd]"
      echo " -h|--help  Prints this help message"
      echo " -d|--debug Sets up in debug mode"
      exit
  endsw
endif

#echo `${dir}/bin/setup.py --csh${build_type}`
eval `${dir}/bin/setup.py --csh${build_type}`
