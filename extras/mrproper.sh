#!/bin/bash 
# Andre Anjos <andre.dos.anjos@gmail.com>
# Mon 28 Jun 2010 15:47:43 CEST
find . -name '*~' -print0 | xargs -0 rm -vf
rm -rf linux*
rm -rf include
rm -rf build 
