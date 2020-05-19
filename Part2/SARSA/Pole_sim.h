/*****************************************************************************/
/* File:           Pole_sim.h                                                */
/* Description:    Parameters for pole-cart system                           */
/* Author:         Manfred Huber                                             */
/* Date:           04-23-93                                                  */
/* Modifications :                                                           */
/*****************************************************************************/

#include "stdio.h"
#include "math.h"

/*--------------------- Macro Definitions -----------------------------------*/
#ifndef SQR
#define SQR(x)   (((x)*(x)))            /* global macro definitions */
#endif
#ifndef SIGN
#define SIGN(x)  ((x) > 0.0 ? (1.0) : (-1.0))
#endif


#define G        9.81            /* Gravity constant                         */
#define P_DOF    1               /* Degrees of freedom of pole               */
#define L_P      0.5             /* Length of the pole                       */
#define M_P      0.1             /* Mass of the pole                         */
#define M_C      1.0             /* Mass of the 'cart'                       */
#define MU_X     0.0             /* Coefficient of friction in X direction   */
#define MU_Y     0.0             /* Coefficient of friction in Y direction   */
#define MU_Z     0.0             /* Coefficient of friction in Z direction   */
#define MU_T0    0.0
#define MU_T1    0.0
#define MU_T2    0.0
#define F_X      10.0            /* Force in X direction                     */
#define F_Y      10.0            /* Force in Y direction                     */
#define F_Z      10.0            /* Force in Z direction                     */

#define X_MIN    -1.5
#define X_MAX    1.5
#define Y_MIN    -1.5
#define Y_MAX    1.5
#define Z_MIN    -1.5
#define Z_MAX    1.5


#define DTIME    0.01            /* Interval for dynamics simulation (sec)   */
#define TRAINC   2               /* Simulation steps per training step       */
#define MAX_TIME 2000.0          /* Maximum time for one trial (sec)         */

/*------------- Data Structure for State of the Pole and Cart ---------------*/
typedef struct _polesys {
  int    dof;
  double theta[2],
         theta_dot[2],
         theta_ddot[2],
         pos[3],
         vel[3],
         acc[3],
         time;
} Polesys;

Polesys *pole;
