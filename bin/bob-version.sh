#!/bin/sh

git_bin=$(which git)
default=unknown

# if git is not installed, return "unknown"
if [ -z "${git_bin}" ]; then
  echo ${default};
  exit 0;
fi

# change directories to where I am
dir=`dirname $0`
test -n "${dir}" && cd ${dir}

# check for git short hash
version=$(cd "$1" && ${git_bin} describe --tags --match N 2> /dev/null)

# Shallow Git clones (--depth) do not have the N tag:
# use 'git-YYYY-MM-DD-hhhhhhh'.
test -n "${version}" || version=$(cd "$1" && ${git_bin} log -1 --pretty=format:"git-%cd-%h" --date=short 2> /dev/null)

# no version number found, just say ${default}
test -n "${version}" || version=${default}

echo "${version}"
