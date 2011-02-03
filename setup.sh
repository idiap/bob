# Andre Anjos <andre.anjos@idiap.ch>
# Thu Jan 20 15:03:22 CET 2011
dir=$(dirname ${BASH_SOURCE[0]})
echo "Warning: This script will be deprecated in favor of bin/shell.py";
echo "Warning: Please use it to create an environment like this:";
echo "         $ bin/shell.py #release build";
echo "         $ bin/shell.py -d # (or --debug) for a debug build setup";
echo "         $ bin/shell.py --help #will give you more usage options";
eval `${dir}/bin/setup.py $*`
