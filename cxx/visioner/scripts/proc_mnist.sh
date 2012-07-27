#!/bin/bash

prog=/idiap/user/catana/experiments/Visioner/build/projects/readmnist
db=/idiap/user/catana/databases/mnist/

echo "Preparing sub-directories ..."
for digit in `echo "0 1 2 3 4 5 6 7 8 9"`
do
	for part in `echo "train test"`
	do
		mkdir -p ${db}${part}${digit}
		rm -rf ${db}${part}${digit}/*
	done
done

echo "Processing the MNIST binary files ..."
cd ${db}
${prog} ${db}

echo "Building the list files ..."
for digit in `echo "0 1 2 3 4 5 6 7 8 9"`
do
	for part in `echo "train test"`
	do
		echo ">>> <"${part}"> for digit <"${digit}"> ..."
		cd ${db}/${part}${digit}/
		echo ${db}/${part}${digit}/ > ${db}/${part}${digit}.list
		for file in *.png
		do
			echo ${file}" # "${file/.png/.gt} >> ${db}/${part}${digit}.list;
		done
		cd ../
	done
done
