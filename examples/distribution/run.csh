#!/bin/csh -f

set exe_ = "`uname -s`_`uname -m`"

#set prerun_ = "valgrind ./${exe_}"
set prerun_ = "./${exe_}"


#
${prerun_}/gmmforward

#
${prerun_}/gmmdev lists/train_faces.list lists/test_faces.list
${prerun_}/gmmdev lists/train_faces.list lists/test_nonfaces.list


#
# GMM training, adaptation and test

#set norm = "-norm -normfile models/xm2vts-wm.norm"
set norm = "-norm"
#set norm = ""

#set n_gaussians = "-n_gaussians 150"
set n_gaussians = "-n_gaussians 500"

\rm -r models
mkdir -p models
\rm -r scores
mkdir -p scores

# train WM
echo "Training ..."
${prerun_}/gmm lists/xm2vts-wm.list ${n_gaussians} ${norm} -seed 950305 -iterk 10 -iterg 20 -flooring 0.01 -e 1e-04 -prior 0.0 -save models/xm2vts-wm.gmm
#${prerun_}/gmm lists/xm2vts-train.list ${n_gaussians} ${norm} -seed 950305 -iterk 2 -iterg 2 -flooring 0.01 -e 1e-04 -save models/xm2vts-wm.gmm

${prerun_}/gmm --read models/xm2vts-wm.gmm

# adapt ID models
echo "Adapting ..."
foreach i (003 004)
	${prerun_}/gmm --adapt  models/xm2vts-wm.gmm lists/xm2vts-${i}-train.list -iterg 10 -flooring 0.01 -e 1e-04 -map 0.8  -prior 0.0 -save models/xm2vts-${i}.gmm

	${prerun_}/gmm --read models/xm2vts-${i}.gmm
end

# test
echo "Testing ..."
foreach i (003 004)
	foreach j (003 004)
		${prerun_}/gmm --test models/xm2vts-${j}.gmm lists/xm2vts-${i}-train.list scores/${i}-${j}
	end
	${prerun_}/gmm --test models/xm2vts-wm.gmm lists/xm2vts-${i}-train.list scores/${i}-wm
end

#
echo "Computing the scores ..."
foreach i (003 004)
	set llk_wm = `head -1 scores/${i}-wm | cut -d' ' -f2`
	foreach j (003 004)
		set llk_C = `head -1 scores/${i}-${j} | cut -d' ' -f2`
		set score = `echo "$llk_C - $llk_wm" | bc -l`
		echo "${i} ${j} $score" >> scores/scores_train
	end
end
cat scores/scores_train

# test
echo "Testing ..."
foreach i (003 004)
	foreach j (003 004)
		${prerun_}/gmm --test models/xm2vts-${j}.gmm lists/xm2vts-${i}-test.list scores/${i}-${j}
	end
	${prerun_}/gmm --test models/xm2vts-wm.gmm lists/xm2vts-${i}-test.list scores/${i}-wm 
end

#
echo "Computing the scores ..."
foreach i (003 004)
	set llk_wm = `head -1 scores/${i}-wm | cut -d' ' -f2`
	foreach j (003 004)
		set llk_C = `head -1 scores/${i}-${j} | cut -d' ' -f2`
		set score = `echo "$llk_C - $llk_wm" | bc -l`
		echo "${i} ${j} $score" >> scores/scores_test
	end
end
cat scores/scores_test


${prerun_}/gmm --test models/xm2vts-wm.gmm lists/xm2vts-003-test.list scores.txt
cat scores.txt

${prerun_}/gmm --test models/xm2vts-wm.gmm lists/blocks/xm2vts-003-test.list scores-blocks.txt
cat scores-blocks.txt

${prerun_}/gmm --test models/xm2vts-wm.gmm lists/rowblocks/xm2vts-003-test.list scores-rowblocks.txt
cat scores-rowblocks.txt

${prerun_}/gmm --test models/xm2vts-wm.gmm lists/colblocks/xm2vts-003-test.list scores-colblocks.txt
cat scores-colblocks.txt


