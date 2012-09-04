#!/bin/bash

source common.sh

#######################################################################
# Directories
#######################################################################

dir_task=${dir_exp}/face_detection/

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
#data_train=${data_train}${dir_db}/faces/multipie/train.04_1.list:
#data_train=${data_train}${dir_db}/faces/multipie/train.05_0.list:
#data_train=${data_train}${dir_db}/faces/multipie/train.05_1.list:
#data_train=${data_train}${dir_db}/faces/multipie/train.13_0.list:
#data_train=${data_train}${dir_db}/faces/multipie/train.14_0.list:
data_train=${data_train}${dir_db}/faces/xm2vts/all.list:
data_train=${data_train}${dir_db}/faces/cmu-pie/annotated_frontal.list:
data_train=${data_train}${dir_db}/faces/banca/english-subset.list:
data_train=${data_train}${dir_db}/faces/banca/spanish-subset.list:
data_train=${data_train}${dir_db}/faces/yale-b/all.list:
data_train=${data_train}${dir_db}/faces/cmutrain1/all.list:
#data_train=${data_train}${dir_db}/faces/mash-faces/train.list:
data_train=${data_train}${dir_db}/caltech-background/all.list:
data_train=${data_train}${dir_db}/fdhd-background/all-0.3.list:

data_valid=""
#data_valid=${data_valid}${dir_db}/faces/multipie/valid.04_1.list:
#data_valid=${data_valid}${dir_db}/faces/multipie/valid.05_0.list:
#data_valid=${data_valid}${dir_db}/faces/multipie/valid.05_1.list:
#data_valid=${data_valid}${dir_db}/faces/multipie/valid.13_0.list:
#data_valid=${data_valid}${dir_db}/faces/multipie/valid.14_0.list:
data_valid=${data_valid}${dir_db}/faces/banca/french-subset.list:
#data_valid=${data_valid}${dir_db}/faces/mash-faces/test.list:
data_valid=${data_valid}${dir_db}/caltech101-background/all.list:

multipie_frontal_test=""
multipie_frontal_test=${multipie_frontal_test}${dir_db}/faces/multipie/test.04_1.list:
multipie_frontal_test=${multipie_frontal_test}${dir_db}/faces/multipie/test.05_0.list:
multipie_frontal_test=${multipie_frontal_test}${dir_db}/faces/multipie/test.05_1.list:
multipie_frontal_test=${multipie_frontal_test}${dir_db}/faces/multipie/test.13_0.list:
multipie_frontal_test=${multipie_frontal_test}${dir_db}/faces/multipie/test.14_0.list:

data_tests=(	${dir_db}/faces/bioid/all.list
                ${dir_db}/faces/mit+cmu/all.list)
#                ${multipie_frontal_test})

test_names=(	BIOID
                MIT+CMU)
#                MULTIPIE-FRONTAL)

#######################################################################
# Model parameters
#######################################################################

common_params=""
common_params=${common_params}"--model_labels=face,"            # Face type
common_params=${common_params}"--model_seed=0,"                 # Random seed
common_params=${common_params}"--model_ds=2,"                   # Scanning resolution to collect training samples
common_params=${common_params}"--model_min_gt_overlap=0.80,"	# Minimum overlapping with GT
common_params=${common_params}"--model_tagger=object_type,"	# Tagger: object types
common_params=${common_params}"--model_train_data=${data_train},"       # Training data
common_params=${common_params}"--model_valid_data=${data_valid},"       # Validation data
common_params=${common_params}"--model_valid_samples=80000,"           # #validation samples
common_params=${common_params}"--model_train_samples=80000,"           # #training samples
common_params=${common_params}"--model_rounds=1024,"		# #rounds
common_params=${common_params}"--model_loss=diag_log,"          # Loss
common_params=${common_params}"--model_loss_param=0.0,"         # Loss parameter

#######################################################################
# Models
#######################################################################

model_names=(	
                "face20x24.elbp.ept.boot"
                "face20x24.lbp.ept.boot"
                
                "face20x24.mct.ept.boot"                
                "face20x24.mct.var.boot"
                "face20x24.mct.ept"                
                )
                
model_titles=(	
                "EMBLBP-EPT-BOOT"
                "MBLBP-EPT-BOOT"
                
                "MBMCT-EPT-BOOT"                
                "MBMCT-VAR-BOOT"
                "MBMCT-EPT"                
                )

model_params=(	
                "${r20x24},${elbp},${gboost},${shared},${ept},${boot},"
                "${r20x24},${lbp},${gboost},${shared},${ept},${boot},"
                
                "${r20x24},${mct},${gboost},${shared},${ept},${boot},"                
                "${r20x24},${mct},${gboost},${shared},${var},${boot},"
                "${r20x24},${mct},${gboost},${shared},${ept},${noboot},"                
                )
          
model_styles=(  
                "lt:1:lc:rgb:\"green\":pt:1:lw:3,"
                "lt:1:lc:rgb:\"black\":pt:1:lw:3,"
                
                "lt:1:lc:rgb:\"red\":pt:1:lw:3,"
                "lt:3:lc:rgb:\"blue\":pt:3:lw:3,"
                "lt:3:lc:rgb:\"blue\":pt:4:lw:3,"
                )

# Models to compare (the numbers are indices in <model_XXX> configs above)
#       --- "NAME: <indices of the configs to compare>"
plot_configs=(  "CONFIG-FEATURES:   0 1 2"
                "CONFIG-BOOT:  2 4"
                "CONFIG-OPT:   2 3")

# Detection speed
detect_levels=(4 6 8 10)
detect_styles=( "lt:1:lc:rgb:\"red\":pt:1:lw:3,"
                "lt:1:lc:rgb:\"green\":pt:1:lw:3,"
                "lt:1:lc:rgb:\"blue\":pt:1:lw:3,"
                "lt:1:lc:rgb:\"black\":pt:1:lw:3,")

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
                
                # Detection speed
                for (( j=0; j<${#detect_levels[*]}; j++ ))
                do
                        levels=${detect_levels[$j]}                

                        jparams=""
                        jparams=${jparams}" --data ${data_test}"
                        jparams=${jparams}" --roc ${dir_result}/${jname}.levels${levels}.roc"
                        jparams=${jparams}" --detect_model ${dir_models}/${jname}.model"
                        jparams=${jparams}" --detect_threshold 0.0"
                        jparams=${jparams}" --detect_levels ${levels}"
                        jparams=${jparams}" --detect_ds 2"
                        jparams=${jparams}" --detect_cluster 0.01"
                        jparams=${jparams}" --detect_method scanning"
        
                        log=${dir_logs}/test.${jname}.levels${levels}.${test_name}.log
                        job=${dir_task}/job_test.${jname}.levels${levels}.${test_name}.sh
        
                        echo "#!/bin/bash" > ${job}
                        echo "${dir_bin}/detector_eval ${jparams} > ${log}" >> ${job}
                        chmod +x ${job}
                done
        done
done

#######################################################################
# Train and test the models
#######################################################################

run_training_scripts_local
run_testing_scripts_local detector_eval

#######################################################################
# Baseline face detection results
#######################################################################

model_baseline_styles=()
model_baseline_names=()
model_baseline_inputs=()
model_baseline_datas=()

model_baseline_styles=("${model_baseline_styles[@]}"    "lt:1:lc:rgb:\"cyan\":pt:1:lw:3")
model_baseline_names=("${model_baseline_names[@]}"      "Froba")
model_baseline_inputs=("${model_baseline_inputs[@]}"    "baseline_detection_froba_mct_BIOID")
model_baseline_datas=("${model_baseline_datas[@]}"      "BIOID")

model_baseline_styles=("${model_baseline_styles[@]}"    "lt:2:lc:rgb:\"cyan\":pt:2:lw:3")
model_baseline_names=("${model_baseline_names[@]}"      "Viola")
model_baseline_inputs=("${model_baseline_inputs[@]}"    "baseline_detection_viola_rapid1_MIT+CMU") #rapid2
model_baseline_datas=("${model_baseline_datas[@]}"      "MIT+CMU")

#model_baseline_styles=("${model_baseline_styles[@]}"    "lt:3:lc:rgb:\"cyan\":pt:3:lw:3")
#model_baseline_names=("${model_baseline_names[@]}"      "Zhang")
#model_baseline_inputs=("${model_baseline_inputs[@]}"    "baseline_detection_zhang_mb_mct_MIT+CMU")
#model_baseline_datas=("${model_baseline_datas[@]}"      "MIT+CMU")

#model_baseline_styles=("${model_baseline_styles[@]}"    "lt:4:lc:rgb:\"cyan\":pt:4:lw:3")
#model_baseline_names=("${model_baseline_names[@]}"      "Froba")
#model_baseline_inputs=("${model_baseline_inputs[@]}"    "baseline_detection_froba_mct_MIT+CMU")
#model_baseline_datas=("${model_baseline_datas[@]}"      "MIT+CMU")

model_baseline_styles=("${model_baseline_styles[@]}"    "lt:5:lc:rgb:\"cyan\":pt:5:lw:3")
model_baseline_names=("${model_baseline_names[@]}"      "FCBoost")
model_baseline_inputs=("${model_baseline_inputs[@]}"    "baseline_detection_fcboost_MIT+CMU")
model_baseline_datas=("${model_baseline_datas[@]}"      "MIT+CMU")

#######################################################################
# Plot the ROC curves
#######################################################################

echo ">>> Plotting the ROC curves ..."

# Test dataset ...
for (( i_data=0; i_data < ${#data_tests[*]}; i_data++ ))
do
        test_name=${test_names[$i_data]}
        echo "<<< ${test_name} >>> ..."

        base_input=${dir_results}/${test_name}/

        base_dir=${dir_plots}/${test_name}
        mkdir -p ${base_dir}
        
        # Configurations to compare
        for (( i_config=0; i_config < ${#plot_configs[*]}; i_config++ ))
        do        
                config=${plot_configs[$i_config]}
                config_name=${config/:*/}
                config_indices=${config/*:/}
                
                # Plot for each detection speed
                for (( k=0; k<${#detect_levels[*]}; k++ ))
                do
                        levels=${detect_levels[$k]} 

                        plot_labels="FA,DR"
                        plot_data="2:1"
                        plot_limits="0:160,0.70:1.0"
                        plot_title=${test_name} #-${config_name}-levels${levels}
                        plot_eps=${base_dir}/detection.${test_name}.${config_name}.levels${levels}.eps
                
                        plot_inputs=""
                        plot_names=""
                        plot_styles=""
                        for i in ${config_indices}
                        do
                                plot_names=${plot_names},${model_titles[$i]}
                                plot_inputs=${plot_inputs}${base_input}/${model_names[$i]}.levels${levels}.roc,
                                plot_styles=${plot_styles}${model_styles[$i]},
                        done
                
                        # Add the baseline results for the SAME test dataset
                        for (( j=0; j<${#model_baseline_styles[*]}; j++ ))
                        do
                                if [ "${test_name}" == "${model_baseline_datas[$j]}" ]
                                then
                                        plot_styles=${plot_styles},${model_baseline_styles[$j]}
                                        plot_names=${plot_names},${model_baseline_names[$j]}
                                        plot_inputs=${plot_inputs},${dir_baselines}/${model_baseline_inputs[$j]}
                                fi
                        done
                
                        plot_params=${plot_inputs}" "${plot_names}" "${plot_styles}" "${plot_title}
                        plot_params=${plot_params}" "${plot_labels}" "${plot_data}" "${plot_limits}" "${plot_eps}
                        bash ${dir_bin}/plot.sh ${plot_params}
                done
        done
        
        # Models (vary the detection speed)
        for (( i_model=0; i_model<${#model_names[*]}; i_model++ ))
        do
                model_name=${model_names[$i_model]}  
                model_title=${model_titles[$i_model]}

                plot_labels="FA,DR"
                plot_data="2:1"
                plot_limits="0:160,0.70:1.0"
                plot_title=${test_name} #-${model_name}
                plot_eps=${base_dir}/detection.${test_name}.${model_name}.eps
        
                plot_inputs=""
                plot_names=""
                plot_styles=""
        
                # Detection speed
                for (( j=0; j<${#detect_levels[*]}; j++ ))
                do
                        levels=${detect_levels[$j]} 
        
                        plot_names=${plot_names},${model_title}-levels${levels}
                        plot_inputs=${plot_inputs}${base_input}/${model_name}.levels${levels}.roc,
                        plot_styles=${plot_styles}${detect_styles[$j]},
                done
                
                # Add the baseline results for the SAME test dataset
                for (( j=0; j<${#model_baseline_styles[*]}; j++ ))
                do
                        if [ "${test_name}" == "${model_baseline_datas[$j]}" ]
                        then
                                plot_styles=${plot_styles},${model_baseline_styles[$j]}
                                plot_names=${plot_names},${model_baseline_names[$j]}
                                plot_inputs=${plot_inputs},${dir_baselines}/${model_baseline_inputs[$j]}
                        fi
                done
        
                plot_params=${plot_inputs}" "${plot_names}" "${plot_styles}" "${plot_title}
                plot_params=${plot_params}" "${plot_labels}" "${plot_data}" "${plot_limits}" "${plot_eps}
                bash ${dir_bin}/plot.sh ${plot_params}
        done
done
