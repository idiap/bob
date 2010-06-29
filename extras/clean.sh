#!/bin/bash 
# Andre Anjos <andre.dos.anjos@gmail.com>
# Mon 28 Jun 2010 15:43:17 CEST
find . -name '*~' -print0 | xargs -0 rm -vf
