#!/bin/sh

subdirs=$1
hdir=$2
includeh=$3
prefix=$4
excludes=$5

# Eliminate ;s from the list of subdirectories
old_subdirs=$subdirs";"
while [ "$old_subdirs" != "$subdirs" ]
do
	old_subdirs=$subdirs
	subdirs=${subdirs/;/ }
done

# Copy all the headers in the given directory
mkdir -p $hdir
rm -rf $hdir
mkdir -p $hdir
for subdir in $subdirs
do
	for file in `ls -B $subdir/*.h`
	do
		cp $file $hdir"/"
	done
done

# Prepare the header
rm -f $includeh
echo "#ifndef _TORCH5SPRO_H_" >> $includeh
echo "#define _TORCH5SPRO_H_" >> $includeh
echo >> $includeh

# Include all the headers in just a single header
for file in `ls -B $hdir/*.h`
do
	file=`basename $file`

	excluded=0
	for exclude in $excludes
	do
		if [ "$exclude" == "$file" ]
		then
			let excluded=1
			break
		fi
	done
	if [ $excluded -eq 0 ]
	then
		echo "#include \"$prefix/$file\"" >> $includeh
	fi
done

echo >> $includeh
echo "#endif" >> $includeh
echo >> $includeh
