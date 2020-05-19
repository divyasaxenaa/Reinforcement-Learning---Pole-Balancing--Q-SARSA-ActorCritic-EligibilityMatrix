
/*****************************************************************************/
/* File:        Reinf_learn.h                                                */
/* Description: Header file for Reinf_learn.c                                */
/* Author:                                                                   */
/* Date:                                                                     */
/* Modifications :                                                           */
/*****************************************************************************/

int Decoder3Dn(int axis, Polesys *pole, int *invert, int *fail);
void pole_learn(Polesys *pole, int reset, double force[3], int *fail, int *explore);
