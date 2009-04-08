
# convert a bindata file to a tensor file
./Linux_i686/bindata2tensor /home/temp7a/marcel/visidiap-features/xm2vts/normal/face64x80-eyecenter/003_1_1.bindata 003_1_1.tensor

# read a tensor file
./Linux_i686/readtensor 003_1_1.tensor

# ImageScanDataSet test with just a small number of subwindows
./Linux_i686/makeWnds ../facetools/xm2vts.list ../data/images/ pgm 10 . 20 20 -verbose
./Linux_i686/readWnds 003_1_1.wnd
./Linux_i686/imagescandataset ../facetools/xm2vts.list ../facetools/xm2vts.list ../data/images pgm . 20 20 -verbose -save

# ImageScanDataSet test with just a large number of subwindows (~millions)
./Linux_i686/makeWnds ../facetools/xm2vts.list ../data/images/ pgm 10000000 . 20 20
time ./Linux_i686/imagescandataset ../facetools/xm2vts.list ../facetools/xm2vts.list ../data/images pgm . 20 20
