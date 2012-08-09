project(db-python)
cmake_minimum_required(VERSION 2.6)

bob_python_package_bindings(db "" "")

# Installs our script driver
bob_python_script(db "bob_dbmanage.py" "lib/script/dbmanage.py" "main")

# Iris database driver and example tests
bob_python_add_test(db-iris-dump bob_dbmanage.py iris dump --self-test)
bob_python_add_test(db-iris-dump-2 bob_dbmanage.py iris dump --class=versicolor --self-test)

# Our examples (run and install)
bob_python_example(db iris_lda.py lib/example/iris_lda.py)
bob_python_add_test(db iris_lda.py --self-test)

bob_python_example(db iris_rprop.py lib/example/iris_rprop.py)
bob_python_add_test(db iris_rprop.py --self-test)

bob_python_example(db iris_backprop.py lib/example/iris_backprop.py)
bob_python_add_test(db iris_backprop.py --self-test)
