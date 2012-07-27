#!/bin/bash

source common.sh

#######################################################################
# Directories
#######################################################################

dir_task=${dir_exp}/face_pose_estimation/

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
data_train=${data_train}${dir_db}/faces/multipie/train.01_0.list:
data_train=${data_train}${dir_db}/faces/multipie/train.04_1.list:
data_train=${data_train}${dir_db}/faces/multipie/train.05_0.list: 
data_train=${data_train}${dir_db}/faces/multipie/train.05_1.list:
data_train=${data_train}${dir_db}/faces/multipie/train.08_0.list:
data_train=${data_train}${dir_db}/faces/multipie/train.09_0.list: 
data_train=${data_train}${dir_db}/faces/multipie/train.11_0.list:
data_train=${data_train}${dir_db}/faces/multipie/train.12_0.list:  
data_train=${data_train}${dir_db}/faces/multipie/train.13_0.list:
data_train=${data_train}${dir_db}/faces/multipie/train.14_0.list:  
data_train=${data_train}${dir_db}/faces/multipie/train.19_0.list:  
data_train=${data_train}${dir_db}/faces/multipie/train.20_0.list:
data_train=${data_train}${dir_db}/faces/multipie/train.24_0.list:

data_valid=""
data_valid=${data_valid}${dir_db}/faces/multipie/valid.01_0.list:
data_valid=${data_valid}${dir_db}/faces/multipie/valid.04_1.list:
data_valid=${data_valid}${dir_db}/faces/multipie/valid.05_0.list: 
data_valid=${data_valid}${dir_db}/faces/multipie/valid.05_1.list:
data_valid=${data_valid}${dir_db}/faces/multipie/valid.08_0.list:
data_valid=${data_valid}${dir_db}/faces/multipie/valid.09_0.list: 
data_valid=${data_valid}${dir_db}/faces/multipie/valid.11_0.list:
data_valid=${data_valid}${dir_db}/faces/multipie/valid.12_0.list:  
data_valid=${data_valid}${dir_db}/faces/multipie/valid.13_0.list:
data_valid=${data_valid}${dir_db}/faces/multipie/valid.14_0.list:  
data_valid=${data_valid}${dir_db}/faces/multipie/valid.19_0.list:  
data_valid=${data_valid}${dir_db}/faces/multipie/valid.20_0.list:
data_valid=${data_valid}${dir_db}/faces/multipie/valid.24_0.list:

data_test=""
data_test=${data_test}${dir_db}/faces/multipie/test.01_0.list:
data_test=${data_test}${dir_db}/faces/multipie/test.04_1.list:
data_test=${data_test}${dir_db}/faces/multipie/test.05_0.list: 
data_test=${data_test}${dir_db}/faces/multipie/test.05_1.list:
data_test=${data_test}${dir_db}/faces/multipie/test.08_0.list:
data_test=${data_test}${dir_db}/faces/multipie/test.09_0.list: 
data_test=${data_test}${dir_db}/faces/multipie/test.11_0.list:
data_test=${data_test}${dir_db}/faces/multipie/test.12_0.list:  
data_test=${data_test}${dir_db}/faces/multipie/test.13_0.list:
data_test=${data_test}${dir_db}/faces/multipie/test.14_0.list:  
data_test=${data_test}${dir_db}/faces/multipie/test.19_0.list:  
data_test=${data_test}${dir_db}/faces/multipie/test.20_0.list:
data_test=${data_test}${dir_db}/faces/multipie/test.24_0.list:

test_name="MULTIPIE"

data_tests=(${data_test})
test_names=(${test_name})

#######################################################################
# Model parameters
#######################################################################

labels=""
labels=${labels}11_0:12_0:09_0:08_0:13_0:14_0:05_1:05_0:04_1:19_0:20_0:01_0:24_0:

common_params=""
common_params=${common_params}"--model_labels=${labels},"       # Pose labels
common_params=${common_params}"--model_seed=0,"                 # Random seed
common_params=${common_params}"--model_ds=2,"                   # Scanning resolution to collect training samples
common_params=${common_params}"--model_min_gt_overlap=0.80,"	# Minimum overlapping with GT
common_params=${common_params}"--model_tagger=object_pose,"	# Tagger: object types
common_params=${common_params}"--model_train_data=${data_train},"       # Training data
common_params=${common_params}"--model_valid_data=${data_valid},"       # Validation data
common_params=${common_params}"--model_valid_samples=80000,"           # #validation samples
common_params=${common_params}"--model_train_samples=80000,"           # #training samples
common_params=${common_params}"--model_rounds=4096,"		# #rounds
common_params=${common_params}"--model_loss=diag_log,"           # Loss
common_params=${common_params}"--model_loss_param=0.0,"         # Loss parameter

#######################################################################
# Models
#######################################################################

model_names=(	
                "pose40x48.mct.shared.ept.boot"
                "pose40x48.mct.shared.var.boot"
                "pose40x48.mct.shared.ept"                
                "pose40x48.elbp.shared.ept.boot"                
                
                "pose40x48.mct.indep.ept.boot"
                "pose40x48.mct.indep.var.boot"
                "pose40x48.mct.indep.ept" 
                "pose40x48.elbp.indep.ept.boot" 
                )

model_params=(	
                "${r40x48},${mct},${gboost},${ept},${shared},${boot},"                
                "${r40x48},${mct},${gboost},${var},${shared},${boot},"
                "${r40x48},${mct},${gboost},${ept},${shared},${noboot},"
                "${r40x48},${elbp},${gboost},${ept},${shared},${boot},"
                
                "${r40x48},${mct},${gboost},${ept},${indep},${boot},"
                "${r40x48},${mct},${gboost},${var},${indep},${boot},"
                "${r40x48},${mct},${gboost},${ept},${indep},${noboot},"
                "${r40x48},${elbp},${gboost},${ept},${indep},${boot},"
                )

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

        jparams=${jparams}" --model ${dir_models}/${jname}.model"
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
for (( i_data=0; i_data < ${#data_tests[*]}; i_data++ ))
do
        data_test=${data_tests[$i_data]}
        test_name=${test_names[$i_data]}

        dir_result=${dir_results}/${test_name}
        mkdir -p ${dir_result}

        # Models
        for (( i=0; i<${#model_names[*]}; i++ ))
        do
                jname=${model_names[$i]}
                
                jparams=""
                jparams=${jparams}" --data ${data_test}"
                jparams=${jparams}" --detect_model ${dir_exp}/face_detection/models/face20x24.mct.ept.boot.model"
                jparams=${jparams}" --detect_threshold 10.0"
                jparams=${jparams}" --detect_levels 8"
                jparams=${jparams}" --detect_ds 2"
                jparams=${jparams}" --detect_cluster 0.01"
                jparams=${jparams}" --detect_method groundtruth"
                jparams=${jparams}" --classify_model ${dir_models}/${jname}.model"

                log=${dir_logs}/test.${jname}.${test_name}.log
                job=${dir_task}/job_test.${jname}.${test_name}.sh

                echo "#!/bin/bash" > ${job}
                echo "${dir_bin}/classifier_eval ${jparams} > ${log}" >> ${job}
                chmod +x ${job}
        done
done

#######################################################################
# Train and test the models
#######################################################################

run_training_scripts_local
run_testing_scripts_local classifier_eval
