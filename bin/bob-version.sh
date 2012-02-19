#!/bin/sh

# change directories to where I am
dir=`dirname $0`
test -n "${dir}" && cd ${dir}

# check for git short hash
version=$(cd "$1" && git describe --tags --match N 2> /dev/null)

# Shallow Git clones (--depth) do not have the N tag:
# use 'git-YYYY-MM-DD-hhhhhhh'.
test -n "${version}" || version=$(cd "$1" && git log -1 --pretty=format:"git-%cd-%h" --date=short 2> /dev/null)

# no version number found, just say "unknown"
test -n "${version}" || version="unknown"

echo "${version}"
