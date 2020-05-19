/*********************************************************************
 *  File:         Matrix_fncs.c
 *  Description:  some procedures for doing homogeneous transform math
 *  Author:       
 *  Date:         
 *********************************************************************/

#include <stdio.h>
#include <math.h>
#include "Matrix_fncs.h"


void T_mult(t1,t2,result)
double t1[4][4],t2[4][4],result[4][4];
{
  int i,j,k;
  for (i=0; i<4; ++i) {
    for (j=0; j<4; ++j) {
      result[i][j] = 0.0;
      for (k=0; k<4; ++k) {
        result[i][j] += t1[i][k] * t2[k][j];
      }
    }
  }
}

void P_mult(t1,pt,result)
double t1[4][4],pt[4],result[4];
{
  int i,j;
  for (i=0; i<4; ++i) {
    result[i] = 0.0;
    for (j=0; j<4; ++j) {
      result[i] += t1[i][j] * pt[j];
    }
  }
}

void P_add(p1, p2, result)
double p1[4],p2[4],result[4];
{
  int i;
  for (i=0; i<4; ++i)
    result[i] = p1[i] + p2[i];
}

void P_sub(p1, p2, result)
double p1[4],p2[4],result[4];
{
  int i;
  for (i=0; i<4; ++i)
    result[i] = p1[i] - p2[i];
}

void P_copy(p1,p2)
double p1[4], p2[4];
{
  int i;
  for (i=0; i<4; ++i)
    p2[i] = p1[i];
}

void T_copy(t1,t2)                  /* copy t1 into t2 */
double t1[4][4],t2[4][4];
{
  int i,j;
  for (i=0; i<4; ++i) {
    for (j=0; j<4; ++j) {
      t2[i][j] = t1[i][j];
    }
  }
}

void Identity(t)
double t[4][4];
{
  int i,j;
  for (i=0; i<4; ++i)
    for (j=0; j<4; ++j)
      {
	if (i==j)
	  t[i][j] = 1.0;
	else
	  t[i][j] = 0.0;
      }
}

void RotX(phi, t)
double phi;
double t[4][4];
{
  Identity(t);
  t[1][1] = cos(phi);
  t[1][2] = -sin(phi);
  t[2][1] = sin(phi);
  t[2][2] = cos(phi);
}

void RotY(phi, t)
double phi;
double t[4][4];
{
  Identity(t);
  t[0][0] = cos(phi);
  t[0][2] = -sin(phi);
  t[2][0] = sin(phi);
  t[2][2] = cos(phi);
}

void RotZ(phi, t)
double phi;
double t[4][4];
{
  Identity(t);
  t[0][0] = cos(phi);
  t[0][1] = -sin(phi);
  t[1][0] = sin(phi);
  t[1][1] = cos(phi);
}

void Bas_vect(i, sca, p)
int i;
double sca;
double p[4];
{
  int j;
  for (j=0; j<4; ++j)
    {
      if (j==i)
	p[j] = sca;
      else
	p[j] = 0.0;
    }
}
