project(visioner-python)
cmake_minimum_required(VERSION 2.6)

set(ccsrc
   "src/version.cc"
   "src/localize.cc"
   "src/main.cc"
   )

set(pysrc
   "lib/__init__.py"
   "lib/test/__init__.py"
   "lib/test/detection.py"
   "lib/test/localization.py"
   "lib/script/__init__.py"
   "lib/script/facebox.py"
   "lib/script/facepoints.py"
   "lib/script/transcode.py"
   )

set(data
  "lib/Face.MCT9.gz"
  "lib/Facial.MCT9.TMaxBoost.MultiPoint.gz"
  "lib/Facial.MCT9.TMaxBoost.EyeCenters.gz"
  )

bob_python_package_bindings(visioner "${ccsrc}" "${pysrc}" "core_array")
bob_python_data(visioner "${data}" "lib/")

# Visioner depends on Qt4
include (${QT_USE_FILE})
include_directories(SYSTEM ${QT_INCLUDES})

# Our test units
bob_python_add_unittest(visioner "lib/test/detection.py" main
  "${CMAKE_CURRENT_SOURCE_DIR}/../io/data")
bob_python_add_unittest(visioner "lib/test/localization.py" main
  "${CMAKE_CURRENT_SOURCE_DIR}/../io/data")

# No tests?
bob_python_script(visioner "visioner_transcode.py" "lib/script/transcode.py" "main")

bob_python_script(visioner "visioner_facebox.py" "lib/script/facebox.py" "main")
bob_python_add_test(visioner-1 visioner_facebox.py "${CMAKE_CURRENT_SOURCE_DIR}/../io/data/test.mov" --self-test=1)
bob_python_add_test(visioner-2 visioner_facebox.py "${CMAKE_CURRENT_SOURCE_DIR}/../ip/data/faceextract/test-faces.jpg" --self-test=2)
bob_python_script(visioner "visioner_facepoints.py" "lib/script/facepoints.py" "main")
bob_python_add_test(visioner-1 visioner_facepoints.py "${CMAKE_CURRENT_SOURCE_DIR}/../io/data/test.mov" --self-test=1)
bob_python_add_test(visioner-2 visioner_facepoints.py "${CMAKE_CURRENT_SOURCE_DIR}/../ip/data/faceextract/test-faces.jpg" --self-test=2)
