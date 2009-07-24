#!/bin/csh -f

set exe_ = "`uname -s`_`uname -m`"

#set prerun_ = "valgrind ./${exe_}"
set prerun_ = "./${exe_}"

#
# Test filelist test program
#
${prerun_}/filelist ./lists/xm2vts.list ../data/images ../data/gt/eyecenter -image_ext pgm -gt_ext pos -gt_format 1 -one_gt_object -verbose

#
${prerun_}/filelist ./lists/xm2vts.list ../data/images ../data/gt/markup68pt -image_ext pgm -gt_ext pts -gt_format 7 -one_gt_object -verbose

#
${prerun_}/filelist ./lists/cmu.list ../data/cmu_faces ../data/gt/eyecenter -image_ext pgm -gt_ext pos -gt_format 1 -verbose

#
${prerun_}/filelist ./lists/banca.list ../data/images ../data/gt/banca -image_ext jpeg -gt_ext pos -gt_format 2 -one_gt_object -verbose

#
${prerun_}/filelist ./lists/eyenosechin_profile.list ../data/images ../data/gt/eyesnosechin -image_ext ppm -gt_ext pos -gt_format 6 -one_gt_object -verbose

#
${prerun_}/filelist ./lists/eyenosechin_halfprofile.list ../data/images ../data/gt/eyesnosechin -image_ext ppm -gt_ext pos -gt_format 5 -one_gt_object  -verbose

#
${prerun_}/filelist ./lists/eyenosechin_frontal.list ../data/images ../data/gt/eyesnosechin -image_ext ppm -gt_ext pos -gt_format 4 -one_gt_object -verbose


#
# Test GroundTruch image extraction program
${prerun_}/gtimageextract ./lists/xm2vts.list ../data/images ../data/gt/markup68pt ./cfg/geom.norm-64x80.cfg -image_ext pgm -gt_ext pts -gt_format 7 -one_gt_object
../dataset/${exe_}/readtensor 003_1_1.tensor
../dataset/${exe_}/readtensor 003_1_2.tensor
../dataset/${exe_}/tensor2image 003_1_1.tensor -xt jpg -base 003_1_1
../dataset/${exe_}/tensor2image 003_1_2.tensor -xt jpg -base 003_1_2

set images_dir = "/idiap/common_vision/visidiap/databases/xm2vts/normal/images_gray"
set gt_dir = "/idiap/common_vision/visidiap/databases/groundtruth/xm2vts/normal"
#
${prerun_}/gtimageextract ./lists/xm2vts-small.list ${images_dir} ${gt_dir}/eyecenter ./cfg/geom.norm-64x80.cfg -image_ext pgm -gt_ext pos -gt_format 1 -one_gt_object -onetensor
../dataset/${exe_}/readtensor onetensor.tensor
../dataset/${exe_}/tensor2image onetensor.tensor -xt jpg -base xm2vts
\rm *.jpg

#
#${prerun_}/gtimageextract ./lists/xm2vts-all.list  ${images_dir} ${gt_dir}/aam-cootes-markup68pt ./cfg/geom.norm-19x19.cfg -image_ext pgm -gt_ext pts -gt_format 7 -one_gt_object -onetensor
#../dataset/${exe_}/readtensor onetensor.tensor

#
# block decomposition
${prerun_}/tensor2blocks 003_1_1.tensor -verbose -sizeW 16 -sizeH 24 -overlapW 0 -overlapH 0 -o 003_1_1.blocks
../dataset/${exe_}/tensor2image 003_1_1.blocks.tensor -xt jpg -base 003_1_1.blocks
\rm *.jpg

#
# DCT feature extraction on a block-by-block basis, a rowblock-by-block basis or a colblock-by-block basis

mkdir -p tensor

mkdir -p tensor/faces

mv onetensor.tensor tensor/faces/xm2vts-wm-face64x80.tensor

foreach i (003 004)
  foreach j (train test)

	${prerun_}/gtimageextract ./lists/xm2vts-${i}-${j}.list ${images_dir} ${gt_dir}/eyecenter ./cfg/geom.norm-64x80.cfg -image_ext pgm -gt_ext pos -gt_format 1 -one_gt_object -onetensor
	mv onetensor.tensor tensor/faces/xm2vts-${i}-${j}-face64x80.tensor
  end
end

mkdir -p tensor/dct

${prerun_}/tensor2dct tensor/faces/xm2vts-wm-face64x80.tensor -o tensor/dct/xm2vts-wm-face64x80-dct -sizeW 8 -sizeH 8 -overlapW 4 -overlapH 4 -dc 15 
${prerun_}/tensor2dct tensor/faces/xm2vts-wm-face64x80.tensor -o tensor/dct/xm2vts-wm-face64x80-dct-blocks -sizeW 8 -sizeH 8 -overlapW 4 -overlapH 4 -dc 15 -blocks_to_tensor2d
${prerun_}/tensor2dct tensor/faces/xm2vts-wm-face64x80.tensor -o tensor/dct/xm2vts-wm-face64x80-dct-rowblocks -sizeW 8 -sizeH 8 -overlapW 4 -overlapH 4 -dc 15 -rowblocks_to_tensor3d
${prerun_}/tensor2dct tensor/faces/xm2vts-wm-face64x80.tensor -o tensor/dct/xm2vts-wm-face64x80-dct-colblocks -sizeW 8 -sizeH 8 -overlapW 4 -overlapH 4 -dc 15 -colblocks_to_tensor3d

foreach i (003 004)
  foreach j (train test)

	${prerun_}/tensor2dct tensor/faces/xm2vts-${i}-${j}-face64x80.tensor -o tensor/dct/xm2vts-${i}-${j}-face64x80-dct -sizeW 8 -sizeH 8 -overlapW 4 -overlapH 4 -dc 15 
	${prerun_}/tensor2dct tensor/faces/xm2vts-${i}-${j}-face64x80.tensor -o tensor/dct/xm2vts-${i}-${j}-face64x80-dct-blocks -sizeW 8 -sizeH 8 -overlapW 4 -overlapH 4 -dc 15 -blocks_to_tensor2d
	${prerun_}/tensor2dct tensor/faces/xm2vts-${i}-${j}-face64x80.tensor -o tensor/dct/xm2vts-${i}-${j}-face64x80-dct-rowblocks -sizeW 8 -sizeH 8 -overlapW 4 -overlapH 4 -dc 15 -rowblocks_to_tensor3d
	${prerun_}/tensor2dct tensor/faces/xm2vts-${i}-${j}-face64x80.tensor -o tensor/dct/xm2vts-${i}-${j}-face64x80-dct-colblocks -sizeW 8 -sizeH 8 -overlapW 4 -overlapH 4 -dc 15 -colblocks_to_tensor3d

  end
end

