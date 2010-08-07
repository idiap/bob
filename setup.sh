# Andre Anjos <andre.anjos@idiap.ch>
# Wed 30 Jun 2010 16:22:58 CEST

build_type='';
if [ $# -ge 1 ]; then
  case $1 in
    (-d|--debug)
      build_type=" --debug";;
    (*)
      echo "usage: source ./setup.sh -[hd]"
      echo " -h|--help  Prints this help message"
      echo " -d|--debug Sets up in debug mode"
      return;;
  esac
fi

dir=`dirname ${BASH_SOURCE[0]}`
#echo "eval \`${dir}/bin/setup.py${build_type} --base-dir=${dir}\`"
eval `${dir}/bin/setup.py${build_type} --base-dir=${dir}`
