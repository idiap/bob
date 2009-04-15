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
./Linux_i686/boostingLBPRound ../dataset/facetrain.tensor ../dataset/nonfacetrain.tensor -wc 5 -nR 20 -width 19 -height 19 -maxE 1000

to test:
./Linux_i686/boosttest ../dataset/facetrain.tensor ../dataset/nonfacetrain.tensor -width 19 -height 19 -maxE 1000 > log.out



	////view the log.out to view the scores

#######################################################################

