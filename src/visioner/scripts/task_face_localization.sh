#!/bin/bash

source common.sh

#######################################################################
# Facial feature point configurations
#######################################################################

all_ffs="leye:reye:leoc:leic:reic:reoc:ntip:lmc:rmc:tlip:blip:chin:leob:leib:reib:reob"

for ffs in `echo "leye:reye"` # leye:reye:ntip:lmc:rmc"` #leye:reye:ntip:chin leye:reye:lmc:rmc:ntip:chin leye:reye:leic:leoc:reic:reoc:lmc:rmc:ntip:tlip:blip:chin"`
do
        #######################################################################
        # Directories
        #######################################################################

        dir_task=${dir_exp}/face_localization_${ffs//:/_}/

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
        data_train=${data_train}${dir_db}/faces/multipie/train.04_1.list:
        data_train=${data_train}${dir_db}/faces/multipie/train.05_0.list:
        data_train=${data_train}${dir_db}/faces/multipie/train.05_1.list:
        data_train=${data_train}${dir_db}/faces/multipie/train.13_0.list:
        data_train=${data_train}${dir_db}/faces/multipie/train.14_0.list:
        
        data_valid=""
        data_valid=${data_valid}${dir_db}/faces/multipie/valid.04_1.list:
        data_valid=${data_valid}${dir_db}/faces/multipie/valid.05_0.list:
        data_valid=${data_valid}${dir_db}/faces/multipie/valid.05_1.list:
        data_valid=${data_valid}${dir_db}/faces/multipie/valid.13_0.list:
        data_valid=${data_valid}${dir_db}/faces/multipie/valid.14_0.list:
        
        data_test=""
        data_test=${data_test}${dir_db}/faces/multipie/test.04_1.list:
        data_test=${data_test}${dir_db}/faces/multipie/test.05_0.list:
        data_test=${data_test}${dir_db}/faces/multipie/test.05_1.list:
        data_test=${data_test}${dir_db}/faces/multipie/test.13_0.list:
        data_test=${data_test}${dir_db}/faces/multipie/test.14_0.list:

        data_tests=(	${data_test}
                        ${dir_db}/faces/bioid/all.list
                        ${dir_db}/faces/xm2vts/all.list)

        test_names=(	MULTIPIE
                        BIOID
                        XM2VTS)

        #######################################################################
        # Model parameters
        #######################################################################

        common_params=""
        common_params=${common_params}"--model_labels=${ffs},"          # Keypoint IDs
        common_params=${common_params}"--model_seed=0,"                 # Random seed
        common_params=${common_params}"--model_ds=2,"                   # Scanning resolution to collect training samples
        common_params=${common_params}"--model_min_gt_overlap=0.60,"	# Minimum overlapping with GT
        common_params=${common_params}"--model_tagger=keypoint,"	# Tagger: keypoints location
        common_params=${common_params}"--model_train_data=${data_train},"	# Training data
        common_params=${common_params}"--model_valid_data=${data_valid},"	# Validation data        
        common_params=${common_params}"--model_valid_samples=80000,"            # #validation samples
        common_params=${common_params}"--model_train_samples=80000,"            # #training samples
        common_params=${common_params}"--model_rounds=1024,"		# #rounds
        common_params=${common_params}"--model_loss=jesorsky,"          # Loss
        common_params=${common_params}"--model_loss_param=0.0,"         # Loss parameter
        
        # Keypoints to localize
        keypoint_ids=(${ffs//:/ })
        n_keypoints=${#keypoint_ids[*]}

        # Localization methods
        lmethods="1shot mshots+avg mshots+med"
        lmethods_styles=""
        lmethods_styles=${lmethods_styles}"lt:1:lc:rgb:\"red\":pt:1:lw:3,"
        lmethods_styles=${lmethods_styles}"lt:1:lc:rgb:\"green\":pt:1:lw:3,"
        lmethods_styles=${lmethods_styles}"lt:1:lc:rgb:\"blue\":pt:1:lw:3,"

        #######################################################################
        # Models
        ######################################################################

        model_names=(	
                        "facial40x48.mct.shared.ept"
                        "facial40x48ex.mct.shared.ept"  
                        
                        "facial40x48.mct.indep.ept"                             
                        "facial40x48ex.mct.indep.ept"
                        
                        "facial40x48.avg"
                        
                        "facial80x96.mct.shared.ept.boot"
                        "facial80x96.mct.shared.var.boot"
                        "facial80x96.mct.shared.ept"
                        
                        "facial80x96.mct.indep.ept.boot"
                        "facial80x96.mct.indep.var.boot"
                        "facial80x96.mct.indep.ept"
        
                        "facial80x96.avg")
                        
        model_titles=(	
                        "40x48PROJ-MBMCT-SHARED-EPT"
                        "40x48EXHA-MBMCT-SHARED-EPT"
                        
                        "40x48PROJ-MBMCT-INDEP-EPT"                        
                        "40x48EXHA-MBMCT-INDEP-EPT"
                        
                        "AVG"
        
                        "MBMCT-SHARED-EPT-BOOT"
                        "MBMCT-SHARED-VAR-BOOT"
                        "MBMCT-SHARED-EPT"
                        
                        "MBMCT-INDEP-EPT-BOOT"
                        "MBMCT-INDEP-VAR-BOOT"                                                
                        "MBMCT-INDEP-EPT"
                                
                        "AVG")
        
        model_params=(	
                        "${r40x48},${mct},${gboost},${ept},${shared},${noboot},"
                        "${r40x48ex},${mct},${gboost},${ept},${shared},${noboot},"
                        
                        "${r40x48},${mct},${gboost},${ept},${indep},${noboot},"
                        "${r40x48ex},${mct},${gboost},${ept},${indep},${noboot},"
                        
                        "${r40x48},${mct},${avg},${ept},${shared},${noboot},"
                        
                        "${r80x96},${mct},${gboost},${ept},${shared},${boot},"  
                        "${r80x96},${mct},${gboost},${var},${shared},${boot},"
                        "${r80x96},${mct},${gboost},${ept},${shared},${noboot},"
                        
                        "${r80x96},${mct},${gboost},${ept},${indep},${boot},"
                        "${r80x96},${mct},${gboost},${var},${indep},${boot},"                        
                        "${r80x96},${mct},${gboost},${ept},${indep},${noboot},"
                        
                        "${r80x96},${mct},${avg},${ept},${shared},${noboot},"
                        )

        model_styles=(  "lt:1:lc:rgb:\"red\":pt:1:lw:3,"
                        "lt:3:lc:rgb:\"red\":pt:2:lw:3,"
                        
                        "lt:1:lc:rgb:\"blue\":pt:1:lw:3,"
                        "lt:3:lc:rgb:\"blue\":pt:2:lw:3,"
                        
                        "lt:1:lc:rgb:\"cyan\":pt:1:lw:3,"
                        
                        "lt:1:lc:rgb:\"red\":pt:1:lw:3,"
                        "lt:3:lc:rgb:\"red\":pt:3:lw:3,"
                        "lt:3:lc:rgb:\"red\":pt:4:lw:3,"
                        
                        "lt:1:lc:rgb:\"blue\":pt:1:lw:3,"
                        "lt:3:lc:rgb:\"blue\":pt:3:lw:3,"
                        "lt:3:lc:rgb:\"blue\":pt:4:lw:3,"
                        
                        "lt:1:lc:rgb:\"cyan\":pt:1:lw:3,"
                        )
        
        # Models to compare (the numbers are indices in <model_XXX> configs above)
        #       --- "NAME: <indices of the configs to compare>"
        plot_configs=(  "CONFIG-40x48-SHARED: 0 1 4"
                        "CONFIG-40x48-INDEP: 2 3 4"
                                                
                        "CONFIG-80x96-SHARED: 5 6 7 11"
                        "CONFIG-80x96-INDEP: 8 9 10 11"
                        "CONFIG-80x96-EPT: 5 8 11"                         
                        "CONFIG-80x96-VAR: 6 9 11"                         
                        "CONFIG-80x96-BOOT: 7 10 11")

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
        # Generate the localization scripts
        #######################################################################

        echo "<<< Preparing the localization scripts ..."

        rm -f ${dir_task}/job_test*

        # Testing dataset ...
        for (( i_data=0; i_data < ${#data_tests[*]}; i_data++ ))
        do
                data_test=${data_tests[$i_data]}
                test_name=${test_names[$i_data]}

                dir_result=${dir_results}/${test_name}/
                mkdir -p ${dir_result}

                # Keypoint localization models
                for (( i=0; i<${#model_names[*]}; i++ ))
                do
                        jname=${model_names[$i]}

                        # Localization methods
                        for lmethod in ${lmethods}
                        do
                                jparams=""
                                jparams=${jparams}" --data ${data_test}"
                                jparams=${jparams}" --loc ${dir_result}/${jname}.${lmethod}"
                                jparams=${jparams}" --detect_model ${dir_exp}/face_detection/models/face20x24.mct.ept.boot.model"
                                jparams=${jparams}" --detect_threshold 10.0"
                                jparams=${jparams}" --detect_levels 10"
                                jparams=${jparams}" --detect_ds 2"
                                jparams=${jparams}" --detect_cluster 0.01"
                                jparams=${jparams}" --detect_method scanning"
                                jparams=${jparams}" --localize_model ${dir_models}/${jname}.model"
                                jparams=${jparams}" --localize_method ${lmethod}"
                                
                                log=${dir_logs}/test.${jname}.${test_name}.${lmethod}.log
                                job=${dir_task}/job_test.${jname}.${test_name}.${lmethod}.sh

                                echo "#!/bin/bash" > ${job}
                                echo "${dir_bin}/localizer_eval ${jparams} > ${log}" >> ${job}
                                chmod +x ${job}
                        done
                done
        done

        #######################################################################
        # Train and test the models
        #######################################################################               

        run_training_scripts_local
        run_testing_scripts_local localizer_eval

        #######################################################################
        # Plot the cumulative distance histograms for various keypoints localization methods.
        #######################################################################

        echo ">>> Plotting the keypoints localization cumulative error histograms ..."

        # Test dataset ...
        for (( i_data=0; i_data < ${#data_tests[*]}; i_data++ ))
        do
                test_name=${test_names[$i_data]}
                echo "<<< ${test_name} >>> ..."

                base_input=${dir_results}/${test_name}/

                base_dir=${dir_plots}/${test_name}/
                mkdir -p ${base_dir}
                
                # Generate plots for error and cumulated error histograms
                for base in `echo "cum.histo histo"`
                do
                        # Generate plots for the average and for every keypoint
                        histos=${base}
#                        for (( i=0; i<${n_keypoints}; i++ ))
#                        do
#                                histos=${histos}" ${base}."${keypoint_ids[$i]}
#                        done
    
                        for histo in ${histos}
                        do
                                hdir=${base_dir}/${histo}/   
                                mkdir -p ${hdir}    
                                
                                # Configurations to compare ...
                                for (( i_config=0; i_config < ${#plot_configs[*]}; i_config++ ))
                                do        
                                        config=${plot_configs[$i_config]}
                                        config_name=${config/:*/}
                                        config_indices=${config/*:/}

                                        # ... for each localization method sharedendently
                                        for lmethod in ${lmethods}
                                        do
                                                plot_labels="Distance,CumulatedError"
                                                plot_data="1:2"
                                                plot_limits="0.0:0.3,0.0:1.0"
                                                plot_title=${test_name} #-${config_name}-${lmethod}
                                                plot_eps=${hdir}/localize.${test_name}.${config_name}.${lmethod}.${histo}.eps
                                
                                                plot_inputs=""
                                                plot_names=""
                                                plot_styles=""
                                                for i in ${config_indices}
                                                do
                                                        plot_names=${plot_names},${model_titles[$i]}
                                                        plot_inputs=${plot_inputs}${base_input}/${model_names[$i]}.${lmethod}.${histo},
                                                        plot_styles=${plot_styles},${model_styles[$i]}
                                                done
                                
                                                plot_params=${plot_inputs}" "${plot_names}" "${plot_styles}" "${plot_title}
                                                plot_params=${plot_params}" "${plot_labels}" "${plot_data}" "${plot_limits}" "${plot_eps}
                                                bash ${dir_bin}/plot.sh ${plot_params}
                                        done
                                done
                                
                                # Fix the model, vary the localization method ...
                                for (( i=0; i<${#model_names[*]}; i++ ))
                                do
                                        title=${model_titles[$i]}
                                
                                        plot_labels="Distance,CumulatedError"
                                        plot_data="1:2"
                                        plot_limits="0.0:0.3,0.0:1.0"
                                        plot_styles=${lmethods_styles}
                                        plot_title=${test_name} #-${title}
                                        plot_eps=${hdir}/localize.${test_name}.${title}.${histo}.eps
                        
                                        plot_inputs=""
                                        plot_names=""
                                        for lmethod in ${lmethods}
                                        do
                                                plot_names=${plot_names},${lmethod}
                                                plot_inputs=${plot_inputs}${base_input}/${model_names[$i]}.${lmethod}.${histo},
                                        done
                        
                                        plot_params=${plot_inputs}" "${plot_names}" "${plot_styles}" "${plot_title}
                                        plot_params=${plot_params}" "${plot_labels}" "${plot_data}" "${plot_limits}" "${plot_eps}
                                        bash ${dir_bin}/plot.sh ${plot_params}
                                done
                        done
                done
        done
done

