#include "MctLbpTrainer.h"

namespace Torch
{

    MctLbpTrainer::MctLbpTrainer()
    {
        //Initialize some default parameters
        n_features = 10;
        n_rounds = n_features*10;
        c_features=0;

        sizeof_LUT = (int)pow(2,8);
        // print("Size of LUT = %d\n",sizeof_LUT);
        // initialize ipLBP
        rad=1;
        LBPM = new LBPMachine();
    }
    void MctLbpTrainer::setData(FileBinDataSet *data_)
    {

        data = data_;
        //can find the width and height  and n_examples
        height = data->short_example->size(0);
        width = data->short_example->size(1);
        n_examples = data->short_example->size(2);

        //Shuffle data
        ShuffleData = new IntTensor(n_examples);


//        ip_lbp = new ipLBP8R(rad);
//        ip_lbp->setBOption("ToAverage", false);
//        ip_lbp->setBOption("AddAvgBit", false);
//        ip_lbp->setBOption("Uniform", false);
//        ip_lbp->setBOption("RotInvariant", false);


    }
    void  MctLbpTrainer::train()
    {

        print("Number of examples inside train() %d\n",n_examples);
        //initialize/Allocate all the LUTs
        LUT = new DoubleTensor(n_features,sizeof_LUT);
        b_pixellocation = new IntTensor(n_features,2); // stores only x and y location

        //for each iteration the weights and the LUTs
        it_weights = new DoubleTensor(n_rounds);
        it_LUT = new IntTensor(n_rounds,sizeof_LUT);
        it_b_pix = new IntTensor(n_rounds,2);
        weights = new DoubleTensor(n_examples);

        //allocate memory for one more LUT
        tempLUT = new IntTensor(sizeof_LUT);


        precompute_LBPfeatures();


        initializeWeights();

        for (int i=0;i<n_rounds;i++)
        {
            current_round=i;
            //shuffle the data based on weigth
            randw();
            findbestfeature();
            findweight();
        }


        //Get the final LUT
        finalLUT();
      //  Tprint(LUT);

        saveModel();


        //check if it is trained properly
        // get the scores
        int x,y;
        int lbpval;
        File file;
        file.open("score.out", "w");
        for (int i=0;i<n_examples;i++)
        {
            double score = 0;
            for (int j=0;j<n_features;j++)
            {
                x = (*b_pixellocation)(j,0);
                y = (*b_pixellocation)(j,1);
                lbpval = (*LBPvalue)(y,x,i);
                score += (*LUT)(j,lbpval);
            }
            file.printf("%g\n",score);


        }
        file.close();


//        file.open("LUT.out", "w");
//        for (int i=0;i<sizeof_LUT;i++)
//        {
//
//                ;
//
//            file.printf("%f\n",(*LUT)(8,i));
//
//
//        }
//        file.close();


    }
    void MctLbpTrainer::setnRounds(int n_rounds_,int n_features_)
    {
        n_rounds = n_rounds_;
        n_features = n_features_;
    }

    void MctLbpTrainer::setLbpParameters(LBPMachine::LBPType lbptype,int rad_)
    {
        bool t;
        print("LBPType %d\n",lbptype);
        ip_lbp = LBPM->makeIpLBP(lbptype);
        sizeof_LUT=ip_lbp->getMaxLabel();
        t=ip_lbp->setR(rad_);
        rad = rad_;
        lbp_type = lbptype;

    }

    void MctLbpTrainer::finalLUT()
    {
        IntTensor *track = new IntTensor(height,width);
        track->fill(0);
        int x,y;
        for (int i=0;i<n_features;i++)
        {
            x = (*b_pixellocation)(i,0);
            y = (*b_pixellocation)(i,1);

            (*track)(y,x) = i;
        }
        for (int i=0;i<n_rounds;i++)
        {
            x = (*it_b_pix)(i,0);
            y = (*it_b_pix)(i,1);
            int k=(*track)(y,x) ;
            for (int j=0;j<sizeof_LUT;j++)
                (*LUT)(k,j) += (*it_weights)(i) * (*it_LUT)(i,j);


        }
    }

    void MctLbpTrainer::saveModel()
    {
        bool t;
        double *lut = new double[sizeof_LUT];
        CascadeMachine* CM = new CascadeMachine();
        CM->resize(1);
        CM->resize(0,n_features);
        for(int j=0;j<n_features;j++)
        {
        LBPMachine *lbpmac = new LBPMachine();
               t= lbpmac->setLBPType(lbp_type);
        t=lbpmac->setLBPRadius(rad);
        for (int i=0;i<sizeof_LUT;i++)
            lut[i] = (*LUT)(j,i);

        t=lbpmac->setLUT(lut,sizeof_LUT);
        t=lbpmac->setXY((*b_pixellocation)(j,1),(*b_pixellocation)(j,0 )  );
        t= CM->setMachine(0,j,lbpmac);
        t= CM->setWeight(0,j,1.0);
    }
        t=CM->setThreshold(0,0.0);

        t=CM->setModelSize(width,height);
        File f1;
        t=f1.open("model.wsm","w");
        t=CM->saveFile(f1);
        f1.close();


        delete CM;

        delete lut;



    }
    void MctLbpTrainer::findweight()
    {
        double beta=0;
        ShortTensor *res=new ShortTensor(n_examples);
        int cx,cy;
        cx=(*it_b_pix)(current_round,0);
        cy=(*it_b_pix)(current_round,1);
        //find weighted error for all the examples

        for (int i=0;i<n_examples;i++)
        {
            if ( (*it_LUT)(current_round,(*LBPvalue)(cy,cx,i))*(*data->short_target)(i)<0)
            {
                beta += (*weights)(i);
                (*res)(i)=-1;

            }
            else
                (*res)(i)=1;


        }
        (*it_weights)(current_round) = -log(beta/(1.0-beta));


        double sum_=0;
        //update the weights
        for (int i=0;i<n_examples;i++)
        {
            //decrease the weight for those that are correctly classified
            if ((*res)(i)>0)
            {
                (*weights)(i) = (*weights)(i)* beta/(1.-beta);
            }
            sum_ += (*weights)(i);

        }

        //normalize the weights
        for (int i=0;i<n_examples;i++)
        {
            (*weights)(i) = (*weights)(i)/sum_;
        }

        delete res;

        print("Alpha %f \n", (*it_weights)(current_round));

    }

    void MctLbpTrainer::findbestfeature()
    {

        double minerror = 99999999999.0; // is there a better way of doign it


        int exmp_loc;

        int flag=0;
        int cx,cy;
        //loop for all features
        for (int x =rad;x<width-rad;x++)
            for (int y=rad;y<height-rad;y++)
            {

                //check here if the we have already selected n_features
                flag=0;
                if (c_features<n_features )
                {
                    //
                    cx=x;
                    cy=y;
                    flag=1;
                }
                if (c_features>=n_features )
                {
                    //need to compare the selected features and then set flag=1 only if it matches
                    for (int f=0;f<n_features;f++)
                    {
                        if ( (*b_pixellocation)(f,0)==x && (*b_pixellocation)(f,1)==y)
                        {
                            cx=x;
                            cy=y;
                            flag=1;
                            break;
                        }
                    }

                }





                //build LUT for each feature



                if (flag==1)
                {
                    tempLUT->fill(0);
                    for (int nexp = 0;nexp<n_examples;nexp++)
                    {
                        // 1 is considered +ve class and -1 or other value is considered as -ve value
                        exmp_loc = (*ShuffleData)(nexp);
                        if ((*data->short_target)(exmp_loc)==1)
                            (*tempLUT)((*LBPvalue)(cy,cx,exmp_loc)) += 1;
                        else
                            (*tempLUT)((*LBPvalue)(cy,cx,exmp_loc)) -= 1;
                    }// for nexp


                    for (int k=0;k<sizeof_LUT;k++)
                        if ((*tempLUT)(k)>=0)
                            (*tempLUT)(k)=1;
                        else
                            (*tempLUT)(k) = -1;
                    //Now calculate the error
                    double iterror=0;
                    for (int nexp = 0;nexp<n_examples;nexp++)
                    {
                        exmp_loc = (*ShuffleData)(nexp);

                        if ((*data->short_target)(exmp_loc)==1 &&  (*tempLUT)((*LBPvalue)(cy,cx,exmp_loc))<0)
                            iterror++;
                        if ((*data->short_target)(exmp_loc) !=1 &&  (*tempLUT)((*LBPvalue)(cy,cx,exmp_loc))>=0)
                            iterror++;
                    }//end of for nexp
                    if (iterror<minerror)
                    {
                        (*it_b_pix)(current_round,0) = cx;
                        (*it_b_pix)(current_round,1) = cy;

                        minerror=iterror;
                        for (int k=0;k<sizeof_LUT;k++)
                            (*it_LUT)(current_round,k) = (*tempLUT)(k);
                    }

                }//end of if flag==1

            }//end of for features

        //here try to track the best pixel locations
        if (c_features<n_features)
        {
            flag = 0;
            for (int f=0;f<c_features;f++)
            {
                if ((*it_b_pix)(current_round,0)== (*b_pixellocation)(f,0) && (*it_b_pix)(current_round,1)== (*b_pixellocation)(f,1))
                {
                    flag=1;
                    break;
                }


            }
            if (flag==0)
            {
                (*b_pixellocation)(c_features,0) = (*it_b_pix)(current_round,0);
                (*b_pixellocation)(c_features,1) = (*it_b_pix)(current_round,1);
                c_features++;
            }
        }//end of if c_feature
        print("Error min = %f, best location %d, %d\n", minerror,(*it_b_pix)(current_round,0),(*it_b_pix)(current_round,1));


    }


    void MctLbpTrainer::precompute_LBPfeatures()
    {
        /// temporary tensor for storing LBP values
        /// will be then copied to data
        /// so the data tensor will contain LBP code
        /// Note data has to be short tensor. *********
        ShortTensor *lbpimg = new ShortTensor();
        LBPvalue = new IntTensor(height,width,n_examples);
        LBPvalue->fill(0);



        print("N_examples %d\n",n_examples);
        assert(ip_lbp->setInputSize(height, width) == true);
        for (int nexmp=0;nexmp<n_examples;nexmp++)
        {
            lbpimg->narrow(data->short_example,2,nexmp,1);
            // now you have 2D image

            for (int x=rad;x<width-rad;x++)
                for (int y=rad;y<height-rad;y++)
                {
                    // get LBP code for each location and store in LBPvalue

                    assert(ip_lbp->setXY(y, x) == true);
                    assert(ip_lbp->process(*lbpimg) == true);
                    (*LBPvalue)(y,x,nexmp) = ip_lbp->getLBP();

                }//end of for int y
        }//end of for int nexmp
    }


    MctLbpTrainer::~MctLbpTrainer()
    {
    }

    void MctLbpTrainer::initializeWeights()
    {
        weights->fill(1.0/(double)(n_examples)); // can be changed here to fill Asymmetric weights
    }




    void MctLbpTrainer::randw()
    {
        THRandom_manualSeed( THRandom_seed());
        double random_ = THRandom_uniform(0, 1);
        for (int i=0;i<n_examples;i++)
            (*ShuffleData)(i) = i;

        DoubleTensor *repartition = new DoubleTensor(n_examples+1);
        repartition->fill(0);
        for (int i=0;i<n_examples;i++)
            (*repartition)(i+1) = (*repartition)(i)+(*weights)(i);

       // print("repartition %f\n",(*repartition)(n_examples));

        for (int i=0;i<n_examples;i++)
        {
            double z = THRandom_uniform(0, 1);
            int gauche = 0;
            int droite = n_examples;
            while (gauche+1 != droite)
            {
                int centre = (gauche+droite)/2;
                if ((*repartition)(centre) < z)
                    gauche = centre;
                else
                    droite = centre;
            }
            (*ShuffleData)(i) = gauche;

        }

        delete repartition;


    }






}
