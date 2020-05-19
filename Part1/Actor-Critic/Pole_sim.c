/*****************************************************************************/
/* File:           Pole_sim.c                                                */
/* Description:    Dynamic simulation of a pole-cart system                  */
/* Author:         Manfred Huber                                             */
/* Date:           04-23-93                                                  */
/* Modifications :                                                           */
/*****************************************************************************/

#include <stdio.h>
#include <math.h>
#include "Pole_sim.h"


/*------------------------------ Function -----------------------------------*/
/* Initializes the state of the system to 0, i.e. cart is in location 0 with */
/* velocity 0 and vertical pole without accelleration.                       */
/*---------------------------------------------------------------------------*/
void init_pole()
{
  int i;

  pole->dof = P_DOF;
  
  for (i=0; i<3; ++i)
    {
      pole->pos[i] = 0.0;
      pole->vel[i] = 0.0;
      pole->acc[i] = 0.0;

      if (i<2)
	{
	  pole->theta_dot[i] = 0.0;
	  pole->theta_ddot[i] = 0.0;
	}
    }
  pole->theta[0] = 0.5*M_PI;
  pole->theta[1] = 0.0;
  pole->time = 0.0;
}


/*------------------------------ Function -----------------------------------*/
/* Calculation of the dynamics and the new state of the pole and cart system */
/*---------------------------------------------------------------------------*/
void pole_dyn(f, dtime)
double f[3];
double dtime;
{
  int i;
  double s_x, c_x, s_y, c_y;

  s_x = sin(pole->theta[0]);
  c_x = cos(pole->theta[0]);
  s_y = sin(pole->theta[1]);
  c_y = cos(pole->theta[1]);

  if ((pole->dof)==3)
    {
      if (fabs(s_x) != 0.0)
	{
	  pole->theta_ddot[1] = 
	    (s_y*(f[2]-MU_Z*SIGN(pole->vel[2])) -
	     c_y*(f[0]-MU_Y*SIGN(pole->vel[0])) -
	     2.0*L_P*M_C*c_x*pole->theta_dot[0]*pole->theta_dot[1])
	      /(L_P*M_C*s_x);
	}
      pole->theta_ddot[0] = 
	(-M_C*L_P*s_x*c_x*SQR(pole->theta_dot[1]) -
	 c_x*c_y*(f[2]-MU_Z*SIGN(pole->vel[2])) -
	 s_x*(f[1]-MU_Y*SIGN(pole->vel[1])) +
	 c_x*s_y*(f[0]-MU_X*SIGN(pole->vel[0])))
	  /(L_P*M_C);
    }
  else
    {
      pole->theta_ddot[0] =
        (SQR(pole->theta_dot[0])*M_P*L_P*s_x*c_x*SQR(c_y) +
	 SQR(pole->theta_dot[1])*(M_P+M_C)*L_P*s_x*c_x -
	 (f[0]-MU_X*SIGN(pole->vel[0]))*((M_C+M_P)/M_C)*c_x*s_y -
	 (f[1]-MU_Y*SIGN(pole->vel[1]))*((M_P/M_C)*s_x*SQR(s_y)+s_x) -
	 G*(M_C+M_P)*c_x*c_y)
	  /(L_P*((M_C+M_P)-M_P*SQR(s_x)*SQR(c_y)));

      pole->theta_ddot[1] =
        (-SQR(pole->theta_dot[0])*M_P*L_P*s_x*s_y*c_y -
	 SQR(pole->theta_dot[1])*M_P*L_P*s_x*SQR(s_x)*s_y*c_y -
	 2.0*pole->theta_dot[0]*pole->theta_dot[1]*L_P*((M_C+M_P)*c_x-
	                                         M_P*c_x*SQR(s_x)*SQR(c_y)) -
	 (f[0]-MU_X*SIGN(pole->vel[0]))*((M_P/M_C)*SQR(c_x)*c_y+c_y) -
	 (f[1]-MU_Y*SIGN(pole->vel[1]))*s_x*c_x*s_y*c_y +
	 G*(M_C+M_P)*s_y)
	  /(L_P*((M_C+M_P)*s_x-M_P*SQR(s_x)*s_x*SQR(c_y)));
    }

  pole->acc[0] = 
    (f[0]-MU_X*SIGN(pole->vel[0])-
     M_P*L_P*(pole->theta_ddot[0]*c_x*s_y + pole->theta_ddot[1]*s_x*c_y -
	      (SQR(pole->theta_dot[0]) + SQR(pole->theta_dot[1]))*s_x*s_y +
	      2.0*pole->theta_dot[0]*pole->theta_dot[1]*c_x*c_y))/(M_C+M_P);
  if ((pole->dof) > 1)
    pole->acc[1] = 
      (f[1]-MU_Y*SIGN(pole->vel[1])-
       M_P*L_P*(pole->theta_ddot[0]*s_x + SQR(pole->theta_dot[0])*c_x))
       /(M_P+M_C);
  if ((pole->dof) > 2)
    pole->acc[2] =
      (f[2]-MU_Z*SIGN(pole->vel[2])-
       M_P*L_P*(pole->theta_ddot[0]*c_x*c_y - pole->theta_ddot[1]*s_x*s_y -
       (SQR(pole->theta_dot[0]) + SQR(pole->theta_dot[1]))*s_x*c_y -
       2.0*pole->theta_dot[0]*pole->theta_dot[1]*c_x*s_y))
       /(M_C+M_P) - G;

  for (i=0; i<pole->dof; ++i)
    {
      pole->pos[i] += 0.5*SQR(dtime)*pole->acc[i] + dtime*pole->vel[i];
      pole->vel[i] += dtime*pole->acc[i];

      if (i<2)
	{
	  pole->theta[1-i] += 0.5*SQR(dtime)*pole->theta_ddot[1-i] 
	                    + dtime*pole->theta_dot[1-i];
	  pole->theta_dot[1-i] += dtime*pole->theta_ddot[1-i];
	}
    }
  pole->time += dtime;
}

