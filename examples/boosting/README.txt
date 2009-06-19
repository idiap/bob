####################################################################
# 1 # Stump Haar
This particular version trains Haar features.

usage:
# Training a Single Stage
./Linux_i686/trainHaarStump -one_file ../data/tensors/ftrain_200.tensor -one_file ../data/tensors/nonftrain_200.tensor ../data/models/boosting/modelHaarSingleStage.wsm -wc 5


# Training a Cascade
./Linux_i686/trainHaarCascade ../data/list/ptrain.list ../data/list/pvalid.list ../data/parameters/cascadeHaarparameters.data ../data/list/nonface-small.list ../data/list/nonface-small.list ../data/images pgm ../data/wnd ../data/models/boosting/modelHaarCascade.wsm -verbose


# Testing
./Linux_i686/testBoostedClassifier ../data/list/ptrain.list ../data/models/boosting/modelHaarCascade.wsm -ii

# Parameters Haar
The Parameter file contains the dats for each stage of cascade.
first line: height and width
second Line: number of stages
for each stage mention
		number of classifiers
		detection rate


#################################################################

# 2 # Lut LBP/MCT
This particular version trains LBP features

This one trains MCT features 9 bits.
usage:

# Training a single stage

./Linux_i686/trainLBPStump -one_file ../data/tensors/ftrain_200.tensor -one_file ../data/tensors/nonftrain_200.tensor ../data/models/boosting/modelLBPSingleStage.wsm -wc 5 -nR 50



# Training a Cascade

./Linux_i686/trainLBPCascade ../data/list/ptrain.list ../data/list/pvalid.list ../data/parameters/cascadeLBPparameters.data ../data/list/nonface-small.list ../data/list/nonface-small.list ../data/images pgm ../data/wnd ../data/models/boosting/modelLBPCascade.wsm


# Testing
./Linux_i686/testBoostedClassifier ../data/list/ptrain.list ../data/models/boosting/modelLBPCascade.wsm


	////view the scores.out to view confidence score of last stage

# Parameters LBP/MCT
The Parameter file contains the dats for each stage of cascade.
first line: height and width
second Line: number of stages
for each stage mention
		number of classifiers
		number of rounds
		detection rate


####################################################################################

