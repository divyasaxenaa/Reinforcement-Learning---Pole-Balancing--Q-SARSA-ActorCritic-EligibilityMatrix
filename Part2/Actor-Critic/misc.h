#ifndef RAND_MAX
#define RAND_MAX   2147483647
#endif

/*--------------------- Macro Definitions -----------------------------------*/
#ifndef SQR
#define SQR(x)   (((x)*(x)))            /* global macro definitions */
#endif
#ifndef SIGN
#define SIGN(x)  (((x) > 0.0) ? (1.0) : (-1.0))
#endif
#ifndef MAX
#define MAX(x,y) (((x) > (y)) ? (x) : (y))
#endif