===========================================================================

these are the main parts of my work:

- databases (/idiap/user/catana/databases):
	- the annotations are text mode (.gt extension) and generic
	- a more detailed description is provided in /idiap/user/catana/databases/readme
	
	- it SHOULD be backed-up!
	
- code and scripts (/idiap/user/catana/FINAL/Visioner):
	
  - ./detect.sh:
		- script to test a given detector by drawing the detected faces
		
	- ./classify.sh:
		- script to test a given classifier by drawing the ground truth objects with the predicted
			type, pose or id
			
	- ./localizer.sh:
		- script to test the facial feature localization model by drawing the 
			
	- ../visioner:
		- the source files (headers) of the Visioner library
		
		- ./visioner/util/:
			- utilities (util.*)
			- matrix manipulation (matrix.*)
			
		- ../visioner/vision/:
			- image loading and saving using Qt (image.*) 
			- integral image computation (integral.*)
			- utilities (vision.*)
			- generic annotated object (object.*)
				- the Visioner::Object maps the annotations from /idiap/user/catana/databases/
				- it contains: type, pose, id, bounding box, keypoints (facial features) locations (x, y, id)
			- MB-LBP and MB-MCT feature computation (mb_x*.*)
				- computed at a given (x, y) location or computed densely (at each pixel in the image)
				
		- ../visioner/model/:		
			- training and testing parameters (param.*)
			- mapping from C++ objects to string parameters (mdecoder.*)
				- if other custom objects (models, losses or taggers) are implemented
					then they must be registered to these managers!
			
			- check if a model is over-fitting (generalizer.*)
			
			- pyramid of scaled images (ipyramid.*)
			
			- LUT implementation (lut.*)
			- boosted LUT model (model.*)
			
			- training dataset (dataset.*) stores in memory the feature values, the targets and the costs
			- sampling training and validation datasets uniformly or using bootstrapping (sampler.*)
			- object to extract the targets from a sub-window to build the training and the validation dataset (tagger.*) 
			
			- generic loss (loss.*)
			
		- ../visioner/model/losses/:
			- specific regression and classification losses (implement the <Loss> class)
			
		- ../visioner/model/taggers/:
			- specific sub-window labeling with either face or background for face detection
				or the facial feature locations for facial feature localization
				(implement the <Tagger> class)
				
		- ../visioner/model/models/:
			- specific models, so far only MB-LBP-based models, but others can be added
				by implementing the <Model> class
				
		- ../visioner/model/trainers/:
			- boosting algorithms to train the <Model> objects
			
		- ../visioner/cv/:
			- draw the detected and classified objects and the predicted keypoints locations (cv_draw.*)
			- detect or output ground truth objects (cv_detector.*)
			- classify detected or ground truth objects (cv_classifier.*)
			- localize keypoints or facial features in the detected or ground truth objects (cv_localizer.*)
			
		- ../programs/vgui:
			- Qt applications to analyse the feature maps and the annotations
			- not used for training and testing models
		
		- ./programs/trainer.cpp:
			- trains a model 
			
		- the <program>*_eval.cpp programs are drawing the detection and localization result
			and they are the visual variants of the <program> programs
			- they are used by the scripts in the main directory to visually inspect results
				but are not used by the experimentation scripts from ./scripts/
		
		- other test programs should be added here
		
	- ./:
		- all the experimentation scripts are contained here
		- they are used for training, testing, plotting and logging results
		
		- ./scripts/common.sh:
			- this is the MAIN script and it is included in every other script
			- BEFORE running experiments make sure this script is set accordingly
			
			- variables to check (beginning of the script): 
				- dir_exp (the directory where to save ALL the related experimentation files
					like models, logs, results, plots)
					
				- dir_db (the path where is the database, in my case it was
					/idiap/user/catana/databases/)
					
				- the "boosting constants" section contains various parameters known by the framework
					or some other constants
					(e.g. loss names, feature names, traines, taggers, model sizes)
					
		- ./scripts/prepare_exp.sh:
			- this script will prepare the experimentation directory (the <dir_exp> from above)
				by compiling the code, copying the required programs to ${dir_exp}/bin,
				the experimentation scripts (e.g. task_*.sh) and the baseline results
				
		- ./scripts/plot.sh:
			- maps the received arguments to a gnuplot script that it is deleted after executing it
			- it should not be edited, unless bugs or enhancements
	 		
	 	- ./scripts/task_*.sh:
	 		- each experiment should have a distinct <task_*.sh> script
	 		- most task scripts are very similar in nature 
	 		
	 		- let's consider ./scripts/task_face_detection.sh:
	 			- the "Directories" section specifies the sub-directories (relative to the <dir_exp> in common.sh)
	 				where to store models, training and testing logs, results, plots 
	 				
	 			- the "Training and testing datasets" section sets the training, validation and test datasets
	 				it also provides a generic name for each test dataset (that is used for plotting)
	 				
	 			- the "Model parameters" section sets the common parameters to all models to train and evaluate for this task
	 				these parameters match the ./src/model/param.* structure and the command line of the programs in ./projects/
	 				
	 			- the "Models" sections specifies the models to train, test and compare with each other
	 				each model has beside some specific parameters (see ${model_params}),
	 				a title (${model_titles}) and a style (${model_styles}) used for plotting
	 				and a name (${model_names}) to identify the file where the model is saved to
	 				
	 			- the next sections generates automatically the training (for each model) and testing (for each model
	 				and for each test dataset) scripts
	 				
	 			- then the training and testing scripts are run locally (run_training_scripts_local and run_testing_scripts_local)
	 				or on the grid (run_training_scripts_grid and run_training_scripts_grid) as preferred 
	 				- if the scripts run successfully (most programs in ./projects output a fixed message at the end
		 				so to know that it run OK), then the automatically generated training and testing scripts are deleted
		 				
		 			- these scripts will generate the models, the logs and the results as instructed at the beginning of the script
		 			
		 		- optionally there is a plotting section where the models can be plotted against some baselines

	- ./ex_*:
		- detections, localization and classification results displayed over the test images
		 		
===========================================================================

how to run experiments:

	- make sure the project correctly builds	
	- if additional loss functions, taggers, models or trainers are required make
    sure they are registered with a string id to the library (for details see
    ../visioner/model/mdecoder.h)
		
	- make sure 'common.sh' points correctly to the database and the experimentation directory
		
  - create (or modify an existing) task script (e.g.
    task_<your_experiment_name>.sh) to set the models to evaluate, plotting
    etc. to run the jobs locally or on the grid ...
	
  - make sure ./scripts/prepare_exp.sh is copying the relevant test programs
    (from ./build/), task scripts and optionally baseline files into the
    experimentation directory
		
	- run ./scripts/prepare_exp.sh to setup the experimentation directory
	
	- if no error, then go to the experimentation directory and run:
		time bash task_<your_experiment_name>.sh

	- check experimentation directory for logs, models, results and plots
	
	- ATTENTION:

		- the models are trained one by one, but all the available CPUs are used
		- the test scripts are run in parallel, to keep busy all the CPUs
		
    - the number of threads that can be run on the current machine is
      determined AUTOMATICALLY using boost (checkout ./projects/max_threads
      program)
