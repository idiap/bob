#!/bin/bash 
# Andre Anjos <andre.anjos@idiap.ch>
# Fri 23 Jul 2010 10:46:12 CEST

prefix=`dirname $0`
curdir=`pwd`

# Generates tags, if exuberant-ctags is found
count=`ctags --version | grep -c "5."`
if (( ${count} > 0 )); then
  echo -n "Generating vi(m) tags..."
  cd ${prefix};
  ctags -o ctags --recurse src python >& /dev/null
  if (( $? == 0 )); then echo "Ok";
  else echo "ERROR";
  fi
  cd ${curdir};
  echo -n "Generating emacs tags..."
  cd ${prefix};
  ctags -e -o etags --recurse src python >& /dev/null
  if (( $? == 0 )); then echo "Ok";
  else echo "ERROR";
  fi
  cd ${curdir};
else
  echo "Skipping generation of tags (exuberant-ctags not found)"
fi
exit 0;
