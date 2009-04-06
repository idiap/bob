#
./make.csh

#
iimage

#
geomNorm ../data/images/003_1_1.pgm ../data/gt/markup68pt/003_1_1.pts geom.norm-64x80.cfg 003_1_1.norm.jpeg -gt_format 7

geomNorm ../data/images/003_1_1.pgm ../data/gt/eyecenter/003_1_1.pos geom.norm-19x19.cfg 003_1_1.norm.jpeg -gt_format 1
geomNorm ../data/images/003_1_1.pgm ../data/gt/eyecenter/003_1_1.pos geom.norm-19x19-inplane45.cfg 003_1_1.norm.jpeg -gt_format 1
geomNorm ../data/images/003_1_1.pgm ../data/gt/eyecenter/003_1_1.pos geom.norm-19x19-inplane-45.cfg 003_1_1.norm.jpeg -gt_format 1


