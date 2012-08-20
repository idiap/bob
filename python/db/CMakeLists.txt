project(db-python)
cmake_minimum_required(VERSION 2.6)

bob_python_package_bindings(db "" "")

# Iris database driver and example tests
# TTT bob_python_add_test(db-iris-dump bob_dbmanage.py iris dump --self-test)
# TTT bob_python_add_test(db-iris-dump-2 bob_dbmanage.py iris dump --class=versicolor --self-test)

# Our examples (run and install)
# EEE bob_python_example(db iris_lda.py lib/example/iris_lda.py)
# TTT bob_python_add_test(db iris_lda.py --self-test)

# EEE bob_python_example(db iris_rprop.py lib/example/iris_rprop.py)
# TTT bob_python_add_test(db iris_rprop.py --self-test)

# EEE bob_python_example(db iris_backprop.py lib/example/iris_backprop.py)
# TTT bob_python_add_test(db iris_backprop.py --self-test)
