#!/bin/bash

#######################################################################
# Directories
#######################################################################

dir_exp=/home/cosmin/idiap/experiments/
mkdir -p ${dir_exp}

dir_bin=${dir_exp}/bin/
mkdir -p ${dir_bin}

dir_baselines=${dir_exp}/baselines/
mkdir -p ${dir_baselines}

dir_db=/home/cosmin/idiap/databases/

#######################################################################
# Boosting constants
#######################################################################

# Features
lbp="--model_feature=lbp,"
elbp="--model_feature=elbp,"
mct="--model_feature=mct,"

# Losses
diag_exp="--model_loss=diag_exp,"
diag_log="--model_loss=diag_log,"
diag_symlog="--model_loss=diag_symlog,"
jesorsky="--model_loss=jesorsky,"

# Optimizations
ept="--model_optimization=ept,"
var="--model_optimization=var,"

# Feature sharing
shared="--model_sharing=shared,"
indep="--model_sharing=indep,"

# Model resolutions
r20x24="--model_cols=20,--model_rows=24,--model_projections=0,"
r40x48="--model_cols=40,--model_rows=48,--model_projections=1,"
r80x96="--model_cols=80,--model_rows=96,--model_projections=2,"
r160x192="--model_cols=160,--model_rows=192,--model_projections=3,"

r40x48ex="--model_cols=40,--model_rows=48,--model_projections=0,"

# Bootstrapping
boot="--model_bootstraps=7,"
noboot="--model_bootstraps=0,"
			
# Trainers
avg="--model_trainer=avg,"
gboost="--model_trainer=gboost,"

#######################################################################
# Pad a string with spaces at the end and returns the
#	padded string in the <pad_str> variable.
#######################################################################

function pad_string
{
	__str=$1
	__size=$2	

	pad_str=${__str}
	while [ ${#pad_str} -lt ${__size} ]
	do
		pad_str=${pad_str}"_"
	done
}

#######################################################################
# Prints the filename if it does not contain the given string.
#######################################################################

function echo_if_not_found
{
	__file=$1
	__str=$2
        __log=log4echo_if_not_found
        
        grep "${__str}" ${__file} > ${__log}
        __cnt=`less ${__log} | wc -l`
	if [ "${__cnt}" -eq "0" ]
	then
		echo ${__file}
	fi
        rm -f ${__log}
}

#######################################################################
# Wait for some jobs (programs on the local machine) to finish
#######################################################################

function wait_jobs
{
	__prog=$1

	while [ `ps eax | grep ${__prog} | grep -v grep | wc -l` -gt 0 ]
	do
		still_left=`ps eax | grep ${__prog} | grep -v grep | wc -l`
		echo -n "["$still_left"]."
		sleep 5
	done
	
	echo ""
}

function wait_n_jobs
{
	__max_jobs=$1
	__prog=$2

	while [ `ps eax | grep ${__prog} | grep -v grep | grep -v ? | wc -l` -gt 0 ]
	do
		still_left=`ps eax | grep ${__prog} | grep -v grep | grep -v ? | wc -l`
		if [ $still_left -eq $__max_jobs ]
		then
			echo -n "["$still_left"]."
			sleep 5
		else
			echo ""
			break
		fi
	done
	
	echo ""
}

#######################################################################
# Wait for the submitted jobs (on SGE or MELODIX) to finish
#######################################################################

function sge_wait_jobs
{
        echo "Waiting ..."

        while [ `qstat | wc -l` -gt 0 ]
        do
                no_qstat=`qstat | wc -l`
                still_left=$(($no_qstat-2))
                echo -n "["$still_left"]."
                sleep 10
        done

        echo -e "\nDone!"
}

function sge_wait_n_jobs
{
        max_jobs=$1

        while [ `qstat | wc -l` -gt 0 ]
        do
                no_qstat=`qstat | wc -l`
                still_left=$(($no_qstat-2))
                if [ $still_left -eq $max_jobs ]
                then
                        echo -n "["$still_left"]."
                        sleep 10
                else
                        echo ""
                        break
                fi
        done
}

#######################################################################				
# Run some training scripts
#######################################################################

function check_train
{
	echo "<<< Checking the training scripts ..."	
        sleep 5
	for job in ${dir_task}/job_train*
	do
		log=${job/*job_train./}
                log=${log/%.sh/.train.log}
                echo_if_not_found ${dir_logs}/${log} "Program finished correctly"
		#echo ${job}-${log}
	done
	rm -f ${dir_task}/job_train*
}

function run_training_scripts_local
{
	echo "<<< Running the training scripts ..."
	max_n_jobs=`${dir_bin}/max_threads`
	for job_script in `ls ${dir_task}/job_train*.sh`
	do
                echo "--- Submitting <"${job_script}"> ..."
		time bash ${job_script}
	done
	
	check_train
}

function run_training_scripts_grid
{
	echo "<<< Running the training scripts ..."
	max_n_jobs=1024
	for job_script in `ls ${dir_task}/job_train*.sh`
	do
		echo "--- Submitting <"${job_script}"> ..."
		qsub -l q1w -e ${dir_log} -o ${dir_log} ${job_script}
		#qsub -l q8g -pe pe_exclusive"*" 1"-" -e ${dir_log} -o ${dir_log} ${job_script}
		#qsub -l q16g -pe pe_exclusive"*" 1"-" -e ${dir_log} -o ${dir_log} ${job_script}
		sge_wait_n_jobs ${max_n_jobs}
	done
	sge_wait_jobs
	
	check_train
}

#######################################################################				
# Run some testing scripts
#######################################################################

function check_test
{
	echo "<<< Checking the testing scripts ..."	
	sleep 5
	for job in ${dir_task}/job_test*
	do
		log=${job/*job_test./test.}
		log=${log/%.sh/.log}
		echo_if_not_found ${dir_logs}/${log} "Program finished correctly"
		#echo ${job}-${log}
	done
	rm -f ${dir_task}/job_test*
}

function run_testing_scripts_local
{
	echo "<<< Running the testing scripts ..."	
	max_n_jobs=`${dir_bin}/max_threads`
	for job_script in `ls ${dir_task}/job_test*.sh`
	do
		wait_n_jobs ${max_n_jobs} $1
	
		echo "--- Submitting <"${job_script}"> ..."
		bash ${job_script}&
		sleep 5
	done
	wait_jobs $1
	
	check_test
}

function run_testing_scripts_grid
{
	echo "<<< Running the testing scripts ..."	
	max_n_jobs=1024
	for job_script in `ls ${dir_task}/job_test*.sh`
	do
		echo "--- Submitting <"${job_script}"> ..."
		qsub -l q1d -l hostname=!melodix* -e ${dir_bin} -o ${dir_bin} ${job_script}
		sge_wait_n_jobs ${max_n_jobs}
	done
	sge_wait_jobs
	
	check_test
}
