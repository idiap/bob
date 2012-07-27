#!/bin/bash

##################################################################################
# Plot some 2D points (X, Y) from different source files to the same graphic 
#	and save the plot as .eps using Gnuplot
##################################################################################

# Parameters
inputs=$1	# e.g. data file 1, data file 2, ...
names=$2	# e.g. name1, name2, ...
styles=$3	# e.g. lt:1:pt:1:lw:1, lt:2:pt:2:lw:2,...
title=$4	# Plot title
labels=$5	# e.g. FAR, DR
data=$6		# e.g. 1:2
limits=$7	# e.g. 0:200,0:1 or -
eps_file=$8	# Output file (.eps)

# lt chooses a particular line type: -1=black 1=red 2=grn 3=blue 4=purple 5=aqua 6=brn 7=orange 8=light-brn
# lt must be specified before pt for colored points
# for postscipt -1=normal, 1=grey, 2=dashed, 3=hashed, 4=dot, 5=dot-dash
# lw chooses a line width 1=normal, can use 0.8, 0.3, 1.5, 3, etc.

##################################################################################
# Build the Gnuplot script and run it
##################################################################################

# Decode parameters
inputs=${inputs//,/ }
inputs_=(${inputs})

names=${names//,/ }
names_=(${names})

styles=${styles//,/ }
styles_=(${styles})

labels=${labels//,/ }
labels_=(${labels})
xlabel=${labels_[0]}
ylabel=${labels_[1]}

plot_script="989dsadsadsjkjk00508.tmp"
rm -f ${plot_script}

# Set the plot attributes
echo "set title \"${title}\"" >> ${plot_script}
echo "set xlabel \"${xlabel}\"" >> ${plot_script}
echo "set ylabel \"${ylabel}\"" >> ${plot_script}
if [ "$limits" != "-" ]
then
	limits=${limits//,/ }
	limits_=(${limits})
	echo "set xrange [${limits_[0]}]" >> ${plot_script}
	echo "set yrange [${limits_[1]}]" >> ${plot_script}
fi
echo "set xtic auto" >> ${plot_script}
echo "set ytic auto" >> ${plot_script}   
echo "set grid xtics ytics" >> ${plot_script}
echo "set key right bottom" >> ${plot_script}
#echo "set autoscale;" >> ${plot_script}

# Display each 2D plot curve ...
echo -n "plot " >> ${plot_script}
for (( i=0; i<${#inputs_[*]}; i++ ))
do
	input=${inputs_[$i]}
	name=${names_[$i]}
	style=${styles_[$i]}
	
	echo -e -n "\t'${input}' using ${data} title '${name}' with linespoints ${style//:/ }" >> ${plot_script}
	
	let ii=${i}+1
	if [ $ii -eq ${#inputs_[*]} ] 
	then
		echo "" >> ${plot_script}
	else
		echo ",\\" >> ${plot_script}
	fi
done

echo "set terminal postscript eps color enhanced \"Helvetica\" 18" >> ${plot_script}        
echo "set output \"${eps_file}\"" >> ${plot_script}
echo "replot" >> ${plot_script}

gnuplot ${plot_script} 
rm -f ${plot_script}
