#include "CmdLine.h"

#include "TensorFile.h"
#include "MemoryDataSet.h"
#include "TensorList.h"
#include "BoostingTrainer.h"
#include "FileListCmdOption.h"
//#include "StumpTrainer.h"
#include "StumpMachine.h"
#include "IntLutMachine.h"
#include "Image.h"

#include "spDeltaOne.h"
#include "CascadeMachine.h"
#include "Machines.h"
#include "spCores.h"
#include "ipLBP.h"

using namespace Torch;


int main(int argc, char* argv[])
{
    ///////////////////////////////////////////////////////////////////
    // Parse the command line
    ///////////////////////////////////////////////////////////////////

    // Set options

    char* modelfilename;
    int max_examples;
    int max_features;
    int n_classifiers;
    bool verbose;
    bool rot;
    bool scale;
    bool sLUT;
    bool Occlu;
    int new_h;
    int new_w;
    int width;
    int height;
    File f1,f2;
    bool t;

    FileListCmdOption* tensor_files = new FileListCmdOption("Patterns", "Pattern List");
    tensor_files->isArgument(true);

    // Build the command line object
    CmdLine cmd;
    cmd.setBOption("write log", false);

    cmd.info("Tests the performance of Cascade model");

    cmd.addText("\nArguments:");
    cmd.addCmdOption(tensor_files);
    cmd.addSCmdArg("model file name", &modelfilename, "Trained model file");


    cmd.addText("\nOptions:");
    cmd.addBCmdOption("-verbose", &verbose, false, "print Tensor values");
    cmd.addBCmdOption("-rot", &rot, false, "if you want to rotate");
    cmd.addBCmdOption("-scale", &scale, false, "if you want to scale");
     cmd.addBCmdOption("-sl", &sLUT, false, "if you want to save the lut");
    cmd.addBCmdOption("-Occlu", &Occlu, false, "if you want to take care of occlusion");
    cmd.addICmdOption("-nh", &new_h, 19, "new height");
    cmd.addICmdOption("-nw", &new_w, 19, "new width");

    // Parse the command line
    if (cmd.read(argc, argv) < 0)
    {
        return 0;
    }



    ShortTensor *target1 = new ShortTensor(1);
    target1->fill(1);
    DataSet *mdataset;
    TensorList *tensorList = new TensorList();
    if (tensorList->process(tensor_files,target1)==false)
    {
        print("Error in reading patterns - Tensor list\n");
        return 1;
    }
    mdataset = tensorList->getOutput();

    //




    // Tprint(mdataset.getExample(2));

    //
    print("Test Model ...\n");
    Tensor *st = mdataset->getExample(0);

    int w = st->size(1);
    int h = st->size(0);

    File ofile;
    ofile.open("scores.out","w");
    int n_examples = mdataset->getNoExamples();
    int count =0;

    print("Does some mapping of x,y Now fixed to 19x19 and save the model\n");
    print("\n\n\n\n\n...................Loading the model\n");
    CascadeMachine* cascade = (CascadeMachine*)Torch::loadMachineFromFile(modelfilename);
    print("Number of stages %d\n",cascade->getNoStages());
    int n_stages = cascade->getNoStages();

    /// out of plane rotation mapping



 //saving the lut in file
  double *lut_t1;
    if (sLUT)
    {
t=f1.open("Lut.dat","w");
t=f2.open("Thresh.dat","w");
int lsize = 0;
 for (int i=0;i<n_stages;i++)
        {
            int n_machines=cascade->getNoMachines(i);
            for (int j=0;j<n_machines;j++)
            {
                Machine *machine =cascade->getMachine(i,j);
                lsize = ((IntLutMachine *)machine)->getLUTSize();
                lut_t1 = ((IntLutMachine *)machine)->getLUT();
                for(int k=0;k<lsize;k++)
                    f1.printf("%g ",lut_t1[k]);

                    f1.printf("\n");
            }
            f2.printf("%g\n",cascade->getThreshold(i));
        }
        f1.close();
        f2.close();
    }

    ///scale points
    if (scale)
    {

        TensorSize *modelsize = new TensorSize(new_h,new_w);
        TensorRegion *tregion = new TensorRegion(0,0,new_h,new_w);
        cascade->setSize(*modelsize);

        double scale_h = (new_h+0.0)/(h+0.0);
        double scale_w = (new_w+0.0)/(w+0.0);
        print("Scale %f\n",scale_w);
        int c_h = h/2;
        int c_w = w/2;
        int nc_w = new_w/2;
        int nc_h = new_h/2;
        for (int i=0;i<n_stages;i++)
        {
            int n_machines=cascade->getNoMachines(i);
            for (int j=0;j<n_machines;j++)
            {
                Machine *machine =cascade->getMachine(i,j);
                int x,y;
                x = ((ipLBP*)(machine->getCore()))->getX();
                y = ((ipLBP*)(machine->getCore()))->getY();
                int x1,y1;
                x1=int((x-c_w)*scale_w+nc_w);
                y1=int((y-c_h)*scale_h+nc_h);
                t=((ipLBP*)(machine->getCore()))->setXY(x1,y1);
		machine->getCore()->setModelSize(*modelsize);
                machine->getCore()->setRegion(*tregion);
                print("x %d , y %d: %d,%d center %d %d\n",x,y,x1,y1,nc_w,c_w);
            }
        }
    }
    if (cascade == 0)
    {
        print("ERROR: loading model [%s]!\n", "model.wsm");
        return 1;
    }


    if (rot)
    {


         int map[h][w][2];


         if(Occlu==false)
         {
        t=f1.open("map.data","r");
        for (int i=0;i<361;i++)
        {
            int x1,y1,x2,y2;
            f1.scanf("%d",&x1);
            f1.scanf("%d",&y1);
            f1.scanf("%d",&x2);
            f1.scanf("%d",&y2);
            map[x1][y1][0] = x2;
            map[x1][y1][1] = y2;
        }

        f1.close();
        t=f1.open("mapped_coordinates.dat","w");
        for (int i=0;i<n_stages;i++)
        {
            int n_machines=cascade->getNoMachines(i);
            for (int j=0;j<n_machines;j++)
            {
                Machine *machine =cascade->getMachine(i,j);
                int x,y;
                x = ((ipLBP*)(machine->getCore()))->getX();
                y = ((ipLBP*)(machine->getCore()))->getY();
                int x1;
                x1=map[x][y][0];
                if (map[x][y][0]<1) x1=1;
                if (map[x][y][0]>w-2) x1=w-2;
                t=((ipLBP*)(machine->getCore()))->setXY(x1,map[x][y][1]);
                print("x %d , y %d: %d,%d\n",x,y,x1,map[x][y][1]);
                f1.printf("%d %d %d %d\n",x,y,x1,map[x][y][1]);
            }
        }
  f1.close();
         }

         if(Occlu==true)
         {
           t=f1.open("map.data","r");
        for (int i=0;i<361;i++)
        {
            int x1,y1,x2,y2;
            f1.scanf("%d",&x1);
            f1.scanf("%d",&y1);
            f1.scanf("%d",&x2);
            f1.scanf("%d",&y2);
            map[x1][y1][0] = x2;
            map[x1][y1][1] = y2;
        }

        f1.close();
        t=f1.open("mapped_coordinates.dat","w");
        for (int i=0;i<n_stages;i++)
        {
            int n_machines=cascade->getNoMachines(i);
            for (int j=0;j<n_machines;j++)
            {
                Machine *machine =cascade->getMachine(i,j);
                int x,y;
                x = ((ipLBP*)(machine->getCore()))->getX();
                y = ((ipLBP*)(machine->getCore()))->getY();
                int x1,y1;
                x1=map[x][y][0];
                y1 = map[x][y][1];
                if (map[x][y][0]<1) x1=1;
                if (map[x][y][0]>w-2) x1=w-2;

                if(map[x][y][1]==-999) y1=1;
                t=((ipLBP*)(machine->getCore()))->setXY(x1,y1);
                 if (map[x][y][1]==-999 && map[x][y][0]==-999) {y1 = 0; x1=0;}

                print("x %d , y %d: %d,%d\n",x,y,x1,y1);
                f1.printf("%d %d %d %d\n",x,y,x1,y1);
            }
        }
  f1.close();
         }












        TensorFile tfa;
        tfa.openWrite("mappedtensor.tensor",Tensor::Short, 2, h, w, 0, 0);
        ShortTensor *otensor = new ShortTensor(h,w);


        TensorRegion *tr = new TensorRegion(0,0,h,w);
        for (int i=0;i<n_examples;i++)
        {
            Tensor *st = mdataset->getExample(i);

            ShortTensor *ot = (ShortTensor *)st;
            otensor->fill(0);
            for (int m=0;m<h;m++)
                for (int n=0;n<w;n++)
                {
                    int x1;
                    x1=map[n][m][0];
                    if (map[n][m][0]<0) x1=0;
                    if (map[n][m][0]>w-1) x1=w-1;

                    (*otensor)(map[n][m][1],x1)= (*ot)(m,n);
                }
            tfa.save(*otensor);

            cascade->setRegion(*tr);
            if (cascade->forward(*st) == false)
            {

                delete cascade;
                return 1;
            }

            ofile.printf("%g\n",cascade->getConfidence());
            // print("CONFIDENCE = %f\n", cascade->getConfidence());
            count += cascade->isPattern() ? 1 : 0;
        }

        print("Number of Detection %d / %d \n",count,n_examples);
        print("Performance %f \n",(count+0.0)/(n_examples+1));
        ofile.close();

        tfa.close();

    }
    t=f1.open("model_c.wsm","w");
    t=cascade->saveFile(f1);
    f1.close();

    delete mdataset;
    // delete tensorList;
    delete cascade;


    // OK
    print("OK.\n");

    return 0;
}

