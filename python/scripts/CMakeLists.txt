# Installs all scripts in one shot
# Andre Anjos - 18.july.2010

project(pytorch_scripts)
cmake_minimum_required(VERSION 2.6)

# Installs all python scripts in a nice way
file(COPY . DESTINATION ${DIST_ROOT}/install/bin FILES_MATCHING PATTERN "*.py")
