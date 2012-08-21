project(db-python)
cmake_minimum_required(VERSION 2.6)

bob_python_package_bindings(db "" "")

# Our test units
bob_python_add_unittest(measure lib/test/scripts.py)

# Our examples (run and install)
bob_python_example(db iris_lda.py lib/example/iris_lda.py)
bob_python_add_test(db iris_lda.py --self-test)

bob_python_example(db iris_rprop.py lib/example/iris_rprop.py)
bob_python_add_test(db iris_rprop.py --self-test)

bob_python_example(db iris_backprop.py lib/example/iris_backprop.py)
bob_python_add_test(db iris_backprop.py --self-test)
