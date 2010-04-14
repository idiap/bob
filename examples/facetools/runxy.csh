#!/bin/csh -f

set exe_ = "`uname -s`_`uname -m`"

#set prerun_ = "valgrind ./${exe_}"
set prerun_ = "./${exe_}"

#
echo "*** Default ***"
${prerun_}/tensor2dct 003_1_1.tensor -o 003_1_1.dct -sizeW 8 -sizeH 8 -overlapW 0 -overlapH 0 -dc 15 
${prerun_}/readtensor 003_1_1.dct.tensor -verbose

${prerun_}/tensor2dct 003_1_1.tensor -o 003_1_1.dctxy -sizeW 8 -sizeH 8 -overlapW 0 -overlapH 0 -dc 15 -xy
${prerun_}/readtensor 003_1_1.dctxy.tensor -verbose

#-blocks_to_tensor2d
echo "*** blocks_to_tensor2d ***"
${prerun_}/tensor2dct 003_1_1.tensor -o 003_1_1.dct -sizeW 8 -sizeH 8 -overlapW 0 -overlapH 0 -dc 15 -blocks_to_tensor2d
${prerun_}/readtensor 003_1_1.dct.tensor -verbose

${prerun_}/tensor2dct 003_1_1.tensor -o 003_1_1.dctxy -sizeW 8 -sizeH 8 -overlapW 0 -overlapH 0 -dc 15 -xy -blocks_to_tensor2d
${prerun_}/readtensor 003_1_1.dctxy.tensor -verbose

#-rowblocks_to_tensor3d
echo "*** rowblocks_to_tensor3d ***"
${prerun_}/tensor2dct 003_1_1.tensor -o 003_1_1.dct -sizeW 8 -sizeH 8 -overlapW 0 -overlapH 0 -dc 15 -rowblocks_to_tensor3d
${prerun_}/readtensor 003_1_1.dct.tensor -verbose

${prerun_}/tensor2dct 003_1_1.tensor -o 003_1_1.dctxy -sizeW 8 -sizeH 8 -overlapW 0 -overlapH 0 -dc 15 -xy -rowblocks_to_tensor3d
${prerun_}/readtensor 003_1_1.dctxy.tensor -verbose

#-colblocks_to_tensor3d
#
echo "*** colblocks_to_tensor3d ***"
${prerun_}/tensor2dct 003_1_1.tensor -o 003_1_1.dct -sizeW 8 -sizeH 8 -overlapW 0 -overlapH 0 -dc 15 -colblocks_to_tensor3d
${prerun_}/readtensor 003_1_1.dct.tensor -verbose

${prerun_}/tensor2dct 003_1_1.tensor -o 003_1_1.dctxy -sizeW 8 -sizeH 8 -overlapW 0 -overlapH 0 -dc 15 -xy -colblocks_to_tensor3d
${prerun_}/readtensor 003_1_1.dctxy.tensor -verbose

