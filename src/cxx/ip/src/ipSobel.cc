/**
 * @file cxx/ip/src/ipSobel.cc
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "ip/ipSobel.h"

namespace bob
{

/////////////////////////////////////////////////////////////////////////
// Constructor

    ipSobel::ipSobel()
            :	ipCore()
    {
        createMask();
    }

/////////////////////////////////////////////////////////////////////////
// Destructor

    ipSobel::~ipSobel()
    {
        delete Sx;
        delete Sy;
    }

/////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
// Check if the input tensor has the right dimensions and type

    bool ipSobel::checkInput(const Tensor& input) const
    {
        // Accept only 3D tensors of bob::Image type
        if (	input.nDimension() != 3 ||
                input.getDatatype() != Tensor::Short)
        {
            return false;
        }


        // OK
        return true;
    }

/////////////////////////////////////////////////////////////////////////
// Allocate (if needed) the output tensors given the input tensor dimensions

    bool ipSobel::allocateOutput(const Tensor& input)
    {
        if (	m_output == 0 ||
                m_output[0]->nDimension() != 3 ||
                m_output[0]->size(0) != input.size(0) ||
                m_output[0]->size(1) != input.size(1)||
                m_output[0]->size(2) != input.size(2) ||
                m_output[1]->size(0) != input.size(0) ||
                m_output[1]->size(1) != input.size(1) ||
                m_output[1]->size(2) != input.size(2) ||
                m_output[2]->size(0) != input.size(0) ||
                m_output[2]->size(1) != input.size(1) ||
                m_output[2]->size(2) != input.size(2)
           )

        {
            cleanup();

            // Need allocation
            m_n_outputs = 3;
            m_output = new Tensor*[m_n_outputs];
            m_output[0] = new IntTensor(input.size(0), input.size(1), input.size(2));
            m_output[1] = new IntTensor(input.size(0), input.size(1), input.size(2));
            m_output[2] = new IntTensor(input.size(0), input.size(1), input.size(2));


            return true;
        }

        return true;
    }

/////////////////////////////////////////////////////////////////////////
// Process some input tensor (the input is checked, the outputs are allocated)

    bool ipSobel::processInput(const Tensor& input)
    {
        const ShortTensor* t_input = (ShortTensor*)&input;
        IntTensor* t_Gx_output = (IntTensor*)m_output[0];
        IntTensor* t_Gy_output = (IntTensor*)m_output[1];
        IntTensor* t_Mag_output = (IntTensor*)m_output[2];


        t_Gx_output->fill(0);
        t_Gy_output->fill(0);
        t_Mag_output->fill(0);
        const int width = input.size(1);
        const int height = input.size(0);

        //Create the mask
        //probably you can have different mask and compute the convolution
        //Assuming the mask sizes are all odd


        const int mask_h = Sx->size(0);
        const int mask_w = Sx->size(1);

        int mh = mask_h/2; // be carefull
        int mw = mask_w/2;

        //print("mh %d , mw %d\n",mh,mw);
        double rescale = 255*4;

        // An index for the 3D tensor is: [y * stride_h + x * stride_w + p * stride_p]

        const int n_planes = input.size(2);

        // compute the gradients in x and y direction
        // get the magnitude for each pixel.
        // since sobel in this example has 3x3 mask the border will be 0 in the output.
        for (int p = 0; p < n_planes; p ++)
        {
            for (int y = mh; y < height-mh; y ++)
            {
                //const int in_y = y + m_cropArea.y;
                for (int x = mw; x < width-mw; x ++)
                {
                    int sumx=0;
                    int sumy=0;
                    for (int mx = -mw;mx<=mw;mx++)
                        for (int my = -mh; my<=mh;my++)
                        {
                            int value  = t_input->get(y+my,x+mx,p);
                            sumx +=  value * Sx->get(my+mh,mx+mw);
                            sumy += value * Sy->get(my+mh,mx+mw);
                        }

                    // load the results in the output
                    sumx=round(double(sumx+rescale)/(2*rescale)*255);
                    sumy=round(double(sumy+rescale)/(2*rescale)*255);
                    t_Gx_output->set(y,x,p,sumx);
                    t_Gy_output->set(y,x,p,sumy);

                    //print("Gx %d, Gy %d\n",sumx,sumy);

                    int mag = sqrt(sumx*sumx+sumy*sumy)/360.6 *255;
                    t_Mag_output->set(y,x,p,mag);

                }
            }
        }


        // OK
        return true;
    }


    void ipSobel::createMask()
    {
        Sx = new IntTensor(3,3);
        Sy = new IntTensor(3,3);

        Sx->set(0,0,-1);
        Sx->set(0,1,-2);
        Sx->set(0,2,-1);
        Sx->set(1,0,0);
        Sx->set(1,1,0);
        Sx->set(1,2,0);
        Sx->set(2,0,1);
        Sx->set(2,1,2);
        Sx->set(2,2,1);

        Sy->set(0,0,-1);
        Sy->set(0,1,0);
        Sy->set(0,2,1);
        Sy->set(1,0,-2);
        Sy->set(1,1,0);
        Sy->set(1,2,2);
        Sy->set(2,0,-1);
        Sy->set(2,1,0);
        Sy->set(2,2,1);

    }
/////////////////////////////////////////////////////////////////////////

}
