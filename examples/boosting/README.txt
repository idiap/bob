# 1 # Stump Haar
The input to the machine/trainer is integral image.
This particular version trains Haar features.

usage:
to train
./Linux_i686/boostingintegral ../dataset/facetrain.tensor ../dataset/nonfacetrain.tensor -wc 5 -width 19 -height 19 -maxE 200

	//// after training model.wsm file is created

to test:
./Linux_i686/boottestintergral ../dataset/facetrain.tensor ../dataset/nonfacetrain.tensor -width 19 -height 19 -maxE 200

	///// the test program loads automatically model.wsm file.

the output is confidence score
should be > 0 or <0 for a particular class.


###############################################################

# 2 # Lut LBP/MCT
The input to the machine/trainer is just a image
This particular version trains LBP features

This one trains MCT features 9 bits.
usage:

./Linux_i686/boostingLBPRound -one_file /home/temp3/bvenka/data09/face/tensors/facetrain.tensor -one_file /home/temp3/bvenka/data09/nonface/tensor/nonfacetrain.tensor  -wc 5 -nR 20 

to test:
./Linux_i686/boosttest -one_file /home/temp3/bvenka/data09/face/tensors/facetrain.tensor


# testing boostingLBP
valgrind --tool=memcheck --leak-check=full ./Linux_i686/boostingLBPRound -one_file /home/temp3/bvenka/data09/face/tensors/ftrain_200.tensor -one_file /home/temp3/bvenka/data09/nonface/tensor/nonftrain_200.tensor -wc 5 -nR 20 


	////view the scores.out to view confidence score of last stage

#######################################################################
# 3 # LBP Cascade trainer

nohup nice ./Linux_i686/LBPtrainCascade /home/temp3/bvenka/data09/face/scripts/ptrain-deye-10.list /home/temp3/bvenka/data09/face/scripts/valid-deye-10.list parameter.data /home/temp3/bvenka/data09/nonface/scripts/nonface-small.list /home/temp3/bvenka/data09/nonface/scripts/nonface-small.list /home/temp3/bvenka/data09/nonface/pgm pgm /home/temp3/bvenka/data09/nonface/wnd > log.out &  


# testing LBPCascadeTrainer

valgrind --tool=memcheck --leak-check=full ./Linux_i686/LBPtrainCascade -one_file /home/temp3/bvenka/data09/face/tensors/ftrain_200.tensor -one_file /home/temp3/bvenka/data09/face/tensors/ftrain_200.tensor parameter.data /home/temp3/bvenka/data09/nonface/scripts/n-s.list /home/temp3/bvenka/data09/nonface/scripts/n-s.list /home/temp3/bvenka/data09/nonface/pgm pgm /home/temp3/bvenka/data09/nonface/wnd



The Parameter file contains the dats for each stage of cascade.
firt line: height and width
second Line: number of stages
for each stage mention 
		number of classifiers
		number of rounds
		detection rate



to test: the same as boosttest

