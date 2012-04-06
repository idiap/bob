# This include file only defines which dependencies we need, globally
set(bob_DEPENDENCIES lapack fftw3 imagemagick boost python blitz hdf)
set(bob_OPTIONALS googlePerfTools ffmpeg matio qt4 vlfeat libsvm cv v4l2 sphinx doxygen latex)

# *************************** READ THIS ***********************************
# IMPORTANT: When you update this file, think about updating both the 
# ubuntu/debian control file and our Portfile (OSX installation) so the
# package installations for those systems continue to work properly. 
# *************************** READ THIS ***********************************
