/*****************************************************************************/
/* File:           Xpole.c                                                   */
/* Description:    Graphic simulation of a pole-cart system                  */
/* Author:         Manfred Huber                                             */
/* Date:           04-24-93                                                  */
/* Modifications :                                                           */
/*****************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <X11/Xkw/Xkw.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "Pole_sim.h"
#include "Matrix_fncs.h"
#include "Reinf_learn.h"

#define WIDTH	  650
#define HEIGHT	  512
#define P_WIDTH	  450
#define H	   18
#define W	    ((int) WIDTH/5 - 5)
#define WHITE	    0
#define BLACK	    1
#define DRAWABLE    window
#define FAC	  200.0
#define UPDATE	    1
#define AXIS_SCALE  0.5

#define W2DR(num)	(((int)(FAC*1000.0*(num)+0.5)/1000))
#define W2DX(num)	(width/2+((int)(FAC*1000.0*(num)+0.5)/1000))
#define W2DY(num)	(height/2-((int)(FAC*1000.0*(num)+0.5)/1000))

extern FILE *learning;
extern int explore;
extern int fail;
extern int filecounter;

extern void init_pole();
extern void pole_dyn();

void            x_start_proc(), x_stop_proc(), x_quit_proc();
void            x_run_proc();
void            x_canvas_proc();
void            x_draw_proc();
void            x_quit();
void            x_pos_proc();
void            x_rot_proc();
void            x_ang_proc();
void            x_status_proc();
void            x_file_proc();
void            x_persp_proc();
void            x_view_proc();
void            x_done();
void			x_clear();
void            x_expose();
void			circle(int cu, int cv, int r, int fill);
void			simulate();
void            draw_frame(double xform[4][4]);
void            draw_pole(double perspective[4][4]);

Display         *display;
Window          window;
Pixmap          pixmap;
XtAppContext    app_con;
GC              gc;
int             screen;
Widget          toplevel;
Widget          canvas;
Widget          start, draw;
Widget          persp;
Widget          slider_cart[3];
Widget          slider_ang[2];
Widget          slider_persp[3];
int             width = WIDTH, height = HEIGHT;
int             Pwidth = P_WIDTH;
unsigned long   foreground, background;
XtIntervalId    timer = 0;
int             running = 0;
int             graphics = 0;
int             persp_on = 0;


#define DASH 2


String fallback_resources[]={  "Xpole.title:	XPole",
				"*Xpole*x:	100",
			        "*Xpole*y:	100",
				NULL,
			    };


static char label_pos[3][10];
static char label_rot[3][10];
static double persp_ang[3] = {0.0, 0.0, 0.0};
static double perspective[4][4];

int main(argc,argv)
int argc;
char **argv;
{
  Widget form, widget, vert, horiz;
  int i;
  double pos_ext[3][2];
        

  pole = (Polesys *) malloc(sizeof(Polesys));
  init_pole();
  toplevel =  XtAppInitialize(&app_con, "Xpole",
		NULL, ZERO,&argc, argv, fallback_resources, NULL, ZERO);
  form = XkwMakeForm(toplevel);
  vert = NULL; horiz = NULL;
  canvas = XkwMakeCanvas(form, vert, horiz, x_canvas_proc,width, height);
  vert = canvas; horiz = NULL;
  start = widget = XkwMakeCommand(form, vert, horiz, x_start_proc,
				  "Start", W, H);
  vert = canvas; horiz = widget;
  widget = XkwMakeCommand(form, vert, horiz, x_stop_proc, "Stop", W, H);
  vert = canvas; horiz = widget;
  widget = XkwMakeCommand(form, vert, horiz, x_status_proc, "Status", W, H);
  vert = canvas; horiz = widget;
  widget = draw = XkwMakeCommand(form, vert, horiz, x_draw_proc, "Draw", W, H);
  vert = canvas; horiz = widget;
  widget = XkwMakeCommand(form, vert, horiz, x_quit_proc, "Quit", W, H);

  vert = start; horiz = NULL;
  persp = XkwMakeCommand(form, vert, horiz, x_persp_proc, "Perspective",W, H);
  vert = start; horiz = persp;
  widget = XkwMakeCommand(form, vert, horiz, x_file_proc, "Dump", W, H);


  pos_ext[0][0] = X_MIN;
  pos_ext[0][1] = X_MAX;
  pos_ext[1][0] = Y_MIN;
  pos_ext[1][1] = Y_MAX;
  pos_ext[2][0] = Z_MIN;
  pos_ext[2][1] = Z_MAX;
  sprintf(label_pos[0], "X");
  sprintf(label_pos[1], "Y");
  sprintf(label_pos[2], "Z");
  for(i=0; i<3; i++) {
    vert = widget;  horiz = NULL;
    slider_cart[i] = widget = XkwMakeSlider(form, vert,
			 horiz, x_pos_proc,
			 label_pos[i], width, 0,
			 XtorientHorizontal,
			 pole->pos[i],
			 pos_ext[i][0], pos_ext[i][1], FALSE, Self);
  }

  vert = widget;  horiz = NULL;
  slider_ang[0] = widget = XkwMakeSlider(form, vert,
			 horiz, x_ang_proc,
			 "Theta X", width, 0,
			 XtorientHorizontal,
			 pole->theta[0]-0.5*M_PI,
			 -0.5*M_PI, 0.5*M_PI, TRUE, Self);
  vert = widget;  horiz = NULL;
  slider_ang[1] = widget = XkwMakeSlider(form, vert,
			 horiz, x_ang_proc,
			 "Theta Y", width, 0,
			 XtorientHorizontal,
			 pole->theta[1],
			 -0.5*M_PI, 0.5*M_PI, TRUE, Self);

  
  XtRealizeWidget(toplevel);

  display = XtDisplay(canvas);
  window = XtWindow(canvas);
  screen = DefaultScreen(display);
  
  foreground = BlackPixel(display, screen);
  background = WhitePixel(display, screen);

  gc = XCreateGC(display, window, 0, NULL);
  XSetFunction(display, gc, GXcopy);
  XSetForeground(display,gc,foreground);
  XSetBackground(display,gc,background);

  pixmap=XCreatePixmap(display,window,
		       XDisplayWidth(display, screen),
		       XDisplayHeight(display, screen),
		       DefaultDepth(display,screen));

  
  XkwWidgetHighlight(draw);
  x_rot_proc();
  x_clear();
  XtAppMainLoop(app_con);
  return(0);
}

void x_start_proc(w,client_data,call_data)
Widget w;
caddr_t client_data, call_data;
{     
  static int init = 1;

  if (!running)
    {
      if (init)
	{
	  init = 0;
	  learning = fopen("learning.dat", "w");
	}
      running = 1;
      simulate();
      timer = XtAppAddTimeOut(app_con, UPDATE, x_run_proc, client_data);
      XkwWidgetHighlight(start);
    }
}

void x_run_proc(w,client_data,call_data)
Widget w;
caddr_t client_data, call_data;
{
  simulate();
  timer = XtAppAddTimeOut(app_con, UPDATE, x_run_proc, client_data);
}

void x_stop_proc(w,client_data,call_data)
Widget w;
caddr_t client_data, call_data;
{
  running = 0;
  if(timer) {
    XtRemoveTimeOut(timer);
    XkwWidgetUnhighlight(start);
  }
}

void x_pos_proc(w,client_data,call_data)
Widget w;
caddr_t client_data, call_data;
{}


void x_view_proc(w, client_data, call_data)
Widget w;
XtPointer call_data, client_data;
{
  char name[10];
  int i;
  double rot[4][4];
  double p2[4][4];

  sscanf(XtName(w), "%s View", name);
  switch (name[0])
    {
    case 'F':
        persp_ang[0] = 0.0;
        persp_ang[1] = 0.0;
        persp_ang[2] = 0.0;
	break;
    case 'S':
	persp_ang[0] = 0.0;
	persp_ang[1] = 0.0;
	persp_ang[2] = -0.5*M_PI;
	break;
    case 'T':
	persp_ang[0] = 0.5*M_PI;
	persp_ang[1] = 0.0;
	persp_ang[2] = 0.0;
	break;
      }
  Identity(perspective);
  RotX(persp_ang[0],rot);
  T_mult(rot, perspective, p2);
  RotY(persp_ang[1],rot);
  T_mult(rot, p2, perspective);
  RotZ(persp_ang[2],rot);
  T_mult(rot, perspective, p2);
  T_copy(p2, perspective);
  x_clear();
  draw_frame(perspective);
  draw_pole(perspective);
  for (i=0; i<3; ++i)
    XkwSetSliderValue(slider_persp[i], persp_ang[i]);
}

void x_rot_proc(w,client_data,call_data)
Widget w;
caddr_t client_data, call_data;
{
  int i;
  double rot[4][4];
  double p2[4][4];

  if (persp_on)
    {
      for (i=0; i<3; ++i)
	persp_ang[i] = XkwGetSliderValue(slider_persp[i]);
    }
  Identity(perspective);
  RotX(persp_ang[0],rot);
  T_mult(rot, perspective, p2);
  RotY(persp_ang[1],rot);
  T_mult(rot, p2, perspective);
  RotZ(persp_ang[2],rot);
  T_mult(rot, perspective, p2);
  T_copy(p2, perspective);
  x_clear();
  draw_frame(perspective);
  draw_pole(perspective);
}

void x_ang_proc(w,client_data,call_data)
Widget w;
caddr_t client_data, call_data;
{}

void x_file_proc(w,client_data,call_data)
Widget w;
caddr_t client_data, call_data;
{
filecounter++;
}


void update_sliders()
{
  int i;
  double angle;

  for(i=0; i<3; i++)
    XkwSetSliderValue(slider_cart[i], pole->pos[i]);
  angle =  pole->theta[0]-0.5*M_PI;
  if (angle>M_PI)
    angle -= 2.0*M_PI;
  else
    if (angle<-M_PI)
      angle += 2.0*M_PI;
  XkwSetSliderValue(slider_ang[0], angle);
  angle =  pole->theta[1];
  if (angle>M_PI)
    angle -= 2.0*M_PI;
  else
    if (angle<-M_PI)
      angle += 2.0*M_PI;
  XkwSetSliderValue(slider_ang[1], angle);
}

void x_status_proc(w,client_data,call_data)
Widget w;
caddr_t client_data, call_data;
{ 
  printf("Time :                %f\n", pole->time);
  printf("Cart Position :       %f , %f , %f\n",pole->pos[0],pole->pos[1], 
	                                   pole->pos[2]);
  printf("Cart Velocity :       %f , %f , %f\n",pole->vel[0],pole->vel[1], 
	                                   pole->vel[2]);
  printf("Cart Accelleration :  %f , %f , %f\n\n",pole->acc[0],pole->acc[1], 
	                                   pole->acc[2]);
  printf("Pole Inclination :       %f , %f\n",pole->theta[0],pole->theta[1]);
  printf("Angular Pole Velocity :  %f , %f\n",pole->theta_dot[0],
                                              pole->theta_dot[1]);
  printf("Angular Accelleration :  %f , %f\n\n",pole->theta_ddot[0],
                                              pole->theta_ddot[1]);

}

void x_draw_proc(w,client_data,call_data)
Widget w;
caddr_t client_data, call_data;
{
	++graphics;
	graphics = graphics%3;
  if(graphics == 0) 
    {
      XkwSetWidgetLabel(w, "Draw");
      XkwWidgetHighlight(draw);
    }
  if(graphics == 1) 
    {
      XkwSetWidgetLabel(w, "Draw 1/sec");
      XkwWidgetUnhighlight(draw);
    }
  if(graphics == 2) 
    {
      XkwSetWidgetLabel(w, "Draw 1/20sec");
      XkwWidgetUnhighlight(draw);
    }
}

void x_quit_proc(w, client_data, call_data)
Widget w;
XtPointer call_data, client_data;
{
  if(timer) XtRemoveTimeOut(timer);
  XFreePixmap(display,pixmap);
  XFreeGC(display,gc);
  XtDestroyApplicationContext(app_con);
  fclose(learning);
  exit(0);
}


void x_canvas_proc(w, client_data, call_data)
Widget w;
XtPointer call_data, client_data;
{
  XEvent *event = ((XEvent *) call_data);
  char    text[10];
  KeySym  key;
  int     c;
  static Dimension nwidth = WIDTH, nheight = HEIGHT;
  int     w_x, w_y;
  
  switch(event->type) {
  case ConfigureNotify:
    nwidth = event->xconfigure.width;
    nheight = event->xconfigure.height;
    break;
  case VisibilityNotify:
    XCopyArea(display,window, pixmap,gc, 0,0,width,height,0,0);
    break;
  case Expose:
    if (nwidth == width && nheight == height)
      x_expose();
    else {
      width = nwidth;         height = nheight;
      x_clear();
      draw_frame(perspective);
      draw_pole(perspective);
    }
    break;
  case MotionNotify:
    w_x = event->xmotion.x;
    w_y = event->xmotion.y;
    if(event->xmotion.state == Button1MotionMask) { }
    break;
  case ButtonPress:
    w_x = event->xbutton.x;
    w_y = event->xbutton.y;
    if(event->xbutton.button == Button1) { }
    else if(event->xbutton.button == Button2) { }
    break;
  case ButtonRelease:
    break;
  case KeyPress:
    c = XLookupString((XKeyEvent*)event, text, 10, &key, 0);
    if (c == 1)
      switch (text[0]) {
      case 'c':
	x_clear();
	break;
      case 'q':
	x_quit_proc();
	break;
      }
  }
}

void x_expose() { XCopyArea(display,pixmap,window,gc, 0,0,width,height,0,0); }

void x_clear()
{
  XSetForeground(display,gc,background);
  XFillRectangle(display,pixmap,gc,0,0,
		 XDisplayWidth(display,screen),XDisplayHeight(display,screen));
  XCopyArea(display,pixmap,window,gc,0,0,width,height,0,0);
}

void x_persp_proc(w,client_data,call_data)
Widget w;
caddr_t client_data, call_data;
{ 
  int i;
  Widget  Wpopup, Wform, Wcommand, Wvert, Whoriz, Wwidget;

  if (!persp_on)
    {
      persp_on = 1;
      XkwWidgetHighlight(persp);
      
      Wpopup = XkwMakePopup(w,700,700,"Perspective");
      
      Wform = XkwMakeForm(Wpopup);

      
      sprintf(label_rot[0], "Rot X");
      sprintf(label_rot[1], "Rot Y");
      sprintf(label_rot[2], "Rot Z");

      Wvert = NULL;  Whoriz = NULL;
      slider_persp[0] = XkwMakeSlider(Wform, Wvert,
				      Whoriz, x_rot_proc,
				      label_rot[0], Pwidth, 0,
				      XtorientHorizontal,
				      persp_ang[0],
				      -M_PI, M_PI, FALSE, Self);
      for (i=1; i<3; ++i)
	{
	  Wvert = slider_persp[i-1];  Whoriz = NULL;
	  slider_persp[i] = XkwMakeSlider(Wform, Wvert,
					  Whoriz, x_rot_proc,
					  label_rot[i], Pwidth, 0,
					  XtorientHorizontal,
					  persp_ang[i],
					  -M_PI, M_PI, FALSE, Self);
	}

      Wvert = slider_persp[2];          Whoriz = NULL;
      Wcommand = Wwidget = XkwMakeCommand(Wform, Wvert, Whoriz,
					  x_view_proc, "Front View", 100, 18);
      Wvert = slider_persp[2];          Whoriz = Wwidget;
      Wcommand = Wwidget = XkwMakeCommand(Wform, Wvert, Whoriz,
					  x_view_proc, "Side View", 100, 18);
      Wvert = slider_persp[2];          Whoriz = Wwidget;
      Wcommand = Wwidget = XkwMakeCommand(Wform, Wvert, Whoriz,
					  x_view_proc, "Top View", 100, 18);
      Wvert = slider_persp[2];          Whoriz = Wwidget;
      Wcommand = Wwidget = XkwMakeCommand(Wform, Wvert, Whoriz,
					  x_done, "Done", 100, 18);

      XtPopup(Wpopup, XtGrabNone);
    }
}


void x_done(w, client_data, call_data)
Widget w;
XtPointer call_data, client_data;
{
        XtDestroyWidget(XtParent(XtParent(w)));
	persp_on = 0;
	XkwWidgetUnhighlight(persp);
}


void draw_frame(xform)
double xform[4][4];
{
  static double temp[4][4];


  XSetForeground(display,gc,background);
  XSetLineAttributes(display, gc, 3, LineSolid, CapButt, JoinRound);

  /* x-axis */
  XDrawLine(display, DRAWABLE, gc,
            W2DX(temp[0][3]), W2DY(temp[1][3]),
            W2DX(AXIS_SCALE*temp[0][0] + temp[0][3]),
            W2DY(AXIS_SCALE*temp[1][0] + temp[1][3]));
  XDrawString(display, DRAWABLE, gc,
              W2DX(AXIS_SCALE*(1.1*temp[0][0]) + temp[0][3]),
              W2DY(AXIS_SCALE*(1.1*temp[1][0]) + temp[1][3]), "x", 1);
  /* y-axis */
  XDrawLine(display, DRAWABLE, gc,
            W2DX(temp[0][3]), W2DY(temp[1][3]),
            W2DX(AXIS_SCALE*temp[0][1] + temp[0][3]),
            W2DY(AXIS_SCALE*temp[1][1] + temp[1][3]));
  XDrawString(display, DRAWABLE, gc,
              W2DX(AXIS_SCALE*(1.1*temp[0][1]) + temp[0][3]),
              W2DY(AXIS_SCALE*(1.1*temp[1][1]) + temp[1][3]), "y", 1);
  /* z-axis */
  XDrawLine(display, DRAWABLE, gc,
            W2DX(temp[0][3]), W2DY(temp[1][3]),
            W2DX(AXIS_SCALE*temp[0][2] + temp[0][3]),
            W2DY(AXIS_SCALE*temp[1][2] + temp[1][3]));
  XDrawString(display, DRAWABLE, gc,
              W2DX(AXIS_SCALE*(1.1*temp[0][2]) + temp[0][3]),
              W2DY(AXIS_SCALE*(1.1*temp[1][2]) + temp[1][3]), "z", 1);

  T_copy (xform, temp);

  XSetForeground(display,gc,foreground);
  XSetLineAttributes(display, gc, 1, LineSolid, CapButt, JoinRound);

  /* x-axis */
  XDrawLine(display, DRAWABLE, gc,
	    W2DX(xform[0][3]), W2DY(xform[1][3]), 
	    W2DX(AXIS_SCALE*(xform[0][0]) + xform[0][3]),
            W2DY(AXIS_SCALE*(xform[2][0]) + xform[2][3])); 
  XDrawString(display, DRAWABLE, gc,
	      W2DX(AXIS_SCALE*(1.1*xform[0][0]) + xform[0][3]), 
	      W2DY(AXIS_SCALE*(1.1*xform[2][0]) + xform[2][3]), "x", 1);
  /* y-axis */
  XDrawLine(display, DRAWABLE, gc,
	    W2DX(xform[0][3]), W2DY(xform[1][3]), 
	    W2DX(AXIS_SCALE*(xform[0][1]) + xform[0][3]),
            W2DY(AXIS_SCALE*(xform[2][1]) + xform[2][3])); 
  XDrawString(display, DRAWABLE, gc,
	      W2DX(AXIS_SCALE*(1.1*xform[0][1]) + xform[0][3]), 
	      W2DY(AXIS_SCALE*(1.1*xform[2][1]) + xform[2][3]), "y", 1);
  /* z-axis */
  XDrawLine(display, DRAWABLE, gc,
	    W2DX(xform[0][3]), W2DY(xform[1][3]), 
	    W2DX(AXIS_SCALE*(xform[0][2]) + xform[0][3]),
            W2DY(AXIS_SCALE*(xform[2][2]) + xform[2][3])); 
  XDrawString(display, DRAWABLE, gc,
	      W2DX(AXIS_SCALE*(1.1*xform[0][2]) + xform[0][3]), 
	      W2DY(AXIS_SCALE*(1.1*xform[2][2]) + xform[2][3]), "z", 1);
}

void draw_pole(perspective)
double perspective[4][4];
{
  int i;
  char ptime[8];
  double hpos1[4],hpos2[4];
  double cpos[8][4];
  static double ocpos[8][4];
  static double opos1[4], opos2[4];
  double s_x, c_x, s_y, c_y;

  s_x = sin(pole->theta[0]);
  c_x = cos(pole->theta[0]);
  s_y = sin(pole->theta[1]);
  c_y = cos(pole->theta[1]);

  hpos1[0] = pole->pos[0];
  hpos1[1] = pole->pos[1];
  hpos1[2] = pole->pos[2];
  hpos1[3] = 1.0;

  for (i=0; i<8; ++i)
    P_copy(hpos1, cpos[i]);

  Bas_vect(0,0.1,hpos2);
  P_add(cpos[0],hpos2,cpos[0]);
  P_add(cpos[1],hpos2,cpos[1]);
  P_add(cpos[4],hpos2,cpos[4]);
  P_add(cpos[5],hpos2,cpos[5]);
  P_sub(cpos[2],hpos2,cpos[2]);
  P_sub(cpos[3],hpos2,cpos[3]);
  P_sub(cpos[6],hpos2,cpos[6]);
  P_sub(cpos[7],hpos2,cpos[7]);
  Bas_vect(1,0.1,hpos2);
  P_add(cpos[1],hpos2,cpos[1]);
  P_add(cpos[2],hpos2,cpos[2]);
  P_add(cpos[5],hpos2,cpos[5]);
  P_add(cpos[6],hpos2,cpos[6]);
  P_sub(cpos[0],hpos2,cpos[0]);
  P_sub(cpos[3],hpos2,cpos[3]);
  P_sub(cpos[4],hpos2,cpos[4]);
  P_sub(cpos[7],hpos2,cpos[7]);
  Bas_vect(2,0.1,hpos2);
  P_sub(cpos[4],hpos2,cpos[4]);
  P_sub(cpos[5],hpos2,cpos[5]);
  P_sub(cpos[6],hpos2,cpos[6]);
  P_sub(cpos[7],hpos2,cpos[7]);

  hpos2[0] = hpos1[0] + L_P*s_x*s_y;
  hpos2[1] = hpos1[1] - L_P*c_x;
  hpos2[2] = hpos1[2] + L_P*s_x*c_y;

  XSetLineAttributes(display, gc, 3, LineSolid, CapButt, JoinRound);

  XSetForeground(display,gc,background);
  
  for (i=0; i<4; ++i)
    {
      XDrawLine(display, DRAWABLE, gc, W2DX(ocpos[i][0]), W2DY(ocpos[i][2]),
		W2DX(ocpos[(i+1)%4][0]), W2DY(ocpos[(i+1)%4][2]));
      XDrawLine(display, DRAWABLE, gc, W2DX(ocpos[i+4][0]),W2DY(ocpos[i+4][2]),
		W2DX(ocpos[(i+1)%4+4][0]), W2DY(ocpos[(i+1)%4+4][2]));
      XDrawLine(display, DRAWABLE, gc, W2DX(ocpos[i][0]), W2DY(ocpos[i][2]),
		W2DX(ocpos[i+4][0]), W2DY(ocpos[i+4][2]));
    }  

  XDrawLine(display, DRAWABLE, gc, W2DX(opos1[0]), W2DY(opos1[2]),
                                   W2DX(opos2[0]), W2DY(opos2[2]));


  XSetForeground(display,gc,foreground);

  P_mult(perspective, hpos1, opos1);
  P_mult(perspective, hpos2, opos2);

  XDrawLine(display, DRAWABLE, gc, W2DX(opos1[0]), W2DY(opos1[2]),
                                   W2DX(opos2[0]), W2DY(opos2[2]));

  XSetLineAttributes(display, gc, 1, LineSolid, CapButt, JoinRound);

  for (i=0; i<8; ++i)
    P_mult(perspective, cpos[i], ocpos[i]);

  for (i=0; i<4; ++i)
    {
      XDrawLine(display, DRAWABLE, gc, W2DX(ocpos[i][0]), W2DY(ocpos[i][2]),
		W2DX(ocpos[(i+1)%4][0]), W2DY(ocpos[(i+1)%4][2]));
      XDrawLine(display, DRAWABLE, gc, W2DX(ocpos[i+4][0]),W2DY(ocpos[i+4][2]),
		W2DX(ocpos[(i+1)%4+4][0]), W2DY(ocpos[(i+1)%4+4][2]));
      XDrawLine(display, DRAWABLE, gc, W2DX(ocpos[i][0]), W2DY(ocpos[i][2]),
		W2DX(ocpos[i+4][0]), W2DY(ocpos[i+4][2]));
    }  
  sprintf(ptime, "%4.2lf", pole->time);
  if (pole->time < 1000.0)
    {
      ptime[6] = ' ';
      if (pole->time < 100.0)
	{
	  ptime[5] = ' ';
	  if (pole->time < 10.0)
	    {
	      ptime[4] = ' ';
	    }
	}
    }
  XSetForeground(display,gc,background);
  XFillRectangle(display, DRAWABLE, gc, width-75, height-30, 50, 15);
  XSetForeground(display,gc,foreground);
  XDrawString(display, DRAWABLE, gc,width-75,height-20,ptime, 7); 
}



void circle(cu, cv, r, fill)
int cu, cv, r, fill;
{
  if(fill == WHITE)
    XDrawArc(display, DRAWABLE, gc, cu-r, cv-r, 2*r, 2*r, 0, 64*360);
  else
    XFillArc(display, DRAWABLE, gc, cu-r, cv-r, 2*r, 2*r, 0, 64*360);
}


void simulate()
{
  int i;
  int j = 0;
  static int trial = -1; 
  static double f[3];

  while (j<(20.0/(DTIME*(double)TRAINC)))
    {
      j += (int)pow(20.0, (double)(2-graphics));
      if ((fail) || (pole->time>=MAX_TIME))
	{
	  printf("* trial %d  time = %f\n", trial, pole->time);
	  if (trial >= 0)
	    fprintf(learning, "%d %f\n", trial, pole->time);
	  if(pole->time > 0.0)
	    printf("- exploration =  %f%%\n", 
			   (double)(100*explore*TRAINC)*DTIME/pole->time);
	  else
		printf("- exploration =  0.0%%\n"); 
					   
	  pole_learn(pole, 1, f, &fail, &explore);
	  explore = 0;
	  fail = 0;
	  init_pole();
	  trial++;
	  j=(int)(20.0/(DTIME*(double)TRAINC))+2;
	}
	pole_learn(pole, 0, f, &fail, &explore);
	for (i=0; i<TRAINC; ++i)
	  pole_dyn(f,DTIME);
    }      
  
  draw_frame(perspective);
  draw_pole(perspective);
  update_sliders();
}
