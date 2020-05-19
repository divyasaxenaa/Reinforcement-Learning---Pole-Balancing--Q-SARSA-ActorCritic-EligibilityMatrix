/*********************************************************************
 *  File:         Matrix_fncs.h
 *  Description:  Function definitions for Martix_fncs.c
 *  Author:       
 *  Date:         
 *********************************************************************/

void T_mult(double t1[4][4], double t2[4][4], double result[4][4]);
void P_mult(double t1[4][4], double pt[4], double result[4]);
void P_add(double p1[4], double p2[4], double result[4]);
void P_sub(double p1[4], double p2[4], double result[4]);
void P_copy(double p1[4], double p2[4]);
void T_copy(double t1[4][4], double t2[4][4]);
void Identity(double t[4][4]);
void RotX(double phi, double t[4][4]);
void RotY(double phi, double t[4][4]);
void RotZ(double phi, double t[4][4]);
void Bas_vect(int i, double sca, double p[4]);
