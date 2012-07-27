#!/bin/bash

source common.sh

#######################################################################
# Directories
#######################################################################

dir_task=${dir_exp}/mnist_recognition/

dir_models=${dir_task}/models/
mkdir -p ${dir_models}

dir_results=${dir_task}/results/
mkdir -p ${dir_results}

dir_plots=${dir_task}/plots/
mkdir -p ${dir_plots}

dir_logs=${dir_task}/logs/
mkdir -p ${dir_logs}

#######################################################################
# Training and testing datasets
#######################################################################

data_train=""
data_train=${data_train}${dir_db}/mnist/train0.list:
data_train=${data_train}${dir_db}/mnist/train1.list:
data_train=${data_train}${dir_db}/mnist/train2.list:
data_train=${data_train}${dir_db}/mnist/train3.list:
data_train=${data_train}${dir_db}/mnist/train4.list:
data_train=${data_train}${dir_db}/mnist/train5.list:
data_train=${data_train}${dir_db}/mnist/train6.list:
data_train=${data_train}${dir_db}/mnist/train7.list:
data_train=${data_train}${dir_db}/mnist/train8.list:
data_train=${data_train}${dir_db}/mnist/train9.list:

data_valid=""
data_valid=${data_valid}${dir_db}/mnist/valid0.list:
data_valid=${data_valid}${dir_db}/mnist/valid1.list:
data_valid=${data_valid}${dir_db}/mnist/valid2.list:
data_valid=${data_valid}${dir_db}/mnist/valid3.list:
data_valid=${data_valid}${dir_db}/mnist/valid4.list:
data_valid=${data_valid}${dir_db}/mnist/valid5.list:
data_valid=${data_valid}${dir_db}/mnist/valid6.list:
data_valid=${data_valid}${dir_db}/mnist/valid7.list:
data_valid=${data_valid}${dir_db}/mnist/valid8.list:
data_valid=${data_valid}${dir_db}/mnist/valid9.list:

data_test=""
data_test=${data_test}${dir_db}/mnist/test0.list:
data_test=${data_test}${dir_db}/mnist/test1.list:
data_test=${data_test}${dir_db}/mnist/test2.list:
data_test=${data_test}${dir_db}/mnist/test3.list:
data_test=${data_test}${dir_db}/mnist/test4.list:
data_test=${data_test}${dir_db}/mnist/test5.list:
data_test=${data_test}${dir_db}/mnist/test6.list:
data_test=${data_test}${dir_db}/mnist/test7.list:
data_test=${data_test}${dir_db}/mnist/test8.list:
data_test=${data_test}${dir_db}/mnist/test9.list:

test_name="MNIST"

#######################################################################
# Model parameters
#######################################################################

labels=""
labels=${labels}digit-0:digit-1:digit-2:digit-3:digit-4:
labels=${labels}digit-5:digit-6:digit-7:digit-8:digit-9:

common_params=""
common_params=${common_params}"--model_labels=${labels},"	# Labels
common_params=${common_params}"--model_ds=2,"                   # Scanning resolution to collect training samples
common_params=${common_params}"--model_rows=28,"                # Model size
common_params=${common_params}"--model_cols=28,"
common_params=${common_params}"--model_min_gt_overlap=0.90,"	# Minimum overlapping with GT
common_params=${common_params}"--model_tagger=object_type,"	# Tagger: object types
common_params=${common_params}"--model_train_data=${data_train},"       # Training data
common_params=${common_params}"--model_valid_data=${data_valid},"       # Validation data
common_params=${common_params}"--model_valid_samples=10000,"           # #validation samples
common_params=${common_params}"--model_train_samples=50000,"          # #training samples
common_params=${common_params}"--model_rounds=1024,"		# #rounds
common_params=${common_params}"--model_loss=sum_log,"           # Loss
common_params=${common_params}"--model_loss_param=0.0,"         # Loss parameter

#######################################################################
# Models
#######################################################################

model_names=(	"Digits.MCT4.EPT"
                "Digits.MCT6.EPT"
                "Digits.MCT8.EPT"
                "Digits.MCT9.EPT"
                "Digits.MCT10.EPT"
                
                "Digits.MCT4.VAR"
                "Digits.MCT6.VAR"
                "Digits.MCT8.VAR"
                "Digits.MCT9.VAR"
                "Digits.MCT10.VAR")

model_params=(	"${sparse},${mct4},${gboost_indep_ept},"
                "${sparse},${mct6},${gboost_indep_ept},"
                "${sparse},${mct8},${gboost_indep_ept},"
                "${sparse},${mct9},${gboost_indep_ept},"
                "${sparse},${mct10},${gboost_indep_ept},"
                
                "${sparse},${mct4},${gboost_indep_var},"
                "${sparse},${mct6},${gboost_indep_var},"
                "${sparse},${mct8},${gboost_indep_var},"
                "${sparse},${mct9},${gboost_indep_var},"
                "${sparse},${mct10},${gboost_indep_var},")

#######################################################################
# Generate the training scripts
#######################################################################

echo "<<< Preparing the scripts to train the models ..."

rm -f ${dir_task}/job_train*

for (( i=0; i<${#model_names[*]}; i++ ))
do
        jname=${model_names[$i]}
        jparams=${common_params},${model_params[$i]}
        jparams=${jparams//,/ }

        log=${dir_logs}/${jname}.train.log
        job=${dir_task}/job_train.${jname}.sh
        rm -f ${job}

        jparams=${jparams}" --model ${dir_models}/${jname}"
        echo "#!/bin/bash" >> ${job}
        echo "${dir_bin}/trainer ${jparams} > ${log}" >> ${job}

        chmod +x ${job}
done

#######################################################################
# Generate the detection scripts
#######################################################################

echo "<<< Preparing the detection scripts ..."

rm -f ${dir_task}/job_test*

# Testing dataset ...
dir_result=${dir_results}/${test_name}
mkdir -p ${dir_result}

# Models
for (( i=0; i<${#model_names[*]}; i++ ))
do
        jname=${model_names[$i]}

        jparams=""
        jparams=${jparams}" --data ${data_test}"
        jparams=${jparams}" --roc ${dir_result}/${jname}"
        jparams=${jparams}" --detect_model ${dir_models}/${jname}"
        jparams=${jparams}" --detect_threshold 0.0"
        jparams=${jparams}" --detect_levels 0"
        jparams=${jparams}" --detect_ds 2"
        jparams=${jparams}" --detect_cluster 0.01"

        log=${dir_logs}/Test.${jname}.${test_name}.log
        job=${dir_task}/job_test.${jname}.${test_name}.sh

        echo "#!/bin/bash" > ${job}
        echo "${dir_bin}/detector_eval ${jparams} > ${log}" >> ${job}
        chmod +x ${job}
done

#######################################################################
# Train and test the models
#######################################################################

run_training_scripts_local
run_testing_scripts_local detector_eval

