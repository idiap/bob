#!/bin/bash 
# Andre Anjos <andre.anjos@idiap.ch>
# Fri 06 Aug 2010 08:35:41 CEST

# Runs the full torch nightly build.
bindir=`dirname $0`;
checkout=`dirname ${bindir}`;
prefix=`pwd`;
nightly=`date +%d.%m.%Y`;

echo "[${nightly}] Cleaning up old nightlies..."
find . -depth 1 -type d -not -ctime -7 -print0 | xargs -0 rm -rf;

cd ${checkout}
echo "[${nightly}] Synchronizing repository..."
echo git pull /idiap/home/lelshafey/work/development/torchDep

cd ${prefix}
echo "[${nightly}] Building torch (debug)..."
${bindir}/build.py --log-output --build-type=debug --install-prefix=${nightly}/install --build-prefix=${nightly}/build
debug_status=$?
if [ ${debug_status} = 0 ]; then
  echo "[${nightly}] Debug build successful."
else
  echo "[${nightly}] Debug build failed."
fi

echo "[${nightly}] Building torch (release)..."
${bindir}/build.py --log-output --build-type=release --install-prefix=${nightly}/install --build-prefix=${nightly}/build
release_status=$?
if [ ${release_status} = 0 ]; then
  echo "[${nightly}] Release build successful."
else
  echo "[${nightly}] Release build failed."
fi

echo "[${nightly}] Installing setup files..."
for d in `ls -d ${nightly}/install`; do
  ln -s ${checkout}/setup.sh $d;
  ln -s ${checkout}/setup.csh $d;
done


echo "[${nightly}] Replacing nightly link: last -> ${nightly}"
rm -f last;
ln -s ${nightly} last;

echo "[${nightly}] Running build analysis..."
#${bindir}/analyze.py 
