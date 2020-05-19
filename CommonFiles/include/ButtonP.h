/*
* $XConsortium: ButtonP.h,v 1.29 89/12/07 10:50:03 rws Exp $
*/


/***********************************************************
Copyright 1987, 1988 by Digital Equipment Corporation, Maynard, Massachusetts,
and the Massachusetts Institute of Technology, Cambridge, Massachusetts.

                        All Rights Reserved

Permission to use, copy, modify, and distribute this software and its 
documentation for any purpose and without fee is hereby granted, 
provided that the above copyright notice appear in all copies and that
both that copyright notice and this permission notice appear in 
supporting documentation, and that the names of Digital or MIT not be
used in advertising or publicity pertaining to distribution of the
software without specific, written prior permission.  

DIGITAL DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT SHALL
DIGITAL BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR
ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS
SOFTWARE.

******************************************************************/

/* 
 * ButtonP.h - Private definitions for Button widget
 * 
 */

#ifndef _XawButtonP_h
#define _XawButtonP_h

#include <X11/Xkw/Button.h>
#include <X11/Xaw/LabelP.h>

/***********************************************************************
 *
 * Button Widget Private Data
 *
 ***********************************************************************/

typedef enum {
  HighlightNone,		/* Do not highlight. */
  HighlightWhenUnset,		/* Highlight only when unset, this is
				   to preserve current button widget 
				   functionality. */
  HighlightAlways		/* Always highlight, lets the toggle widget
				   and other subclasses do the right thing. */
} XtButtonHighlight;

/************************************
 *
 *  Class structure
 *
 ***********************************/


   /* New fields for the Button widget class record */
typedef struct _ButtonClass 
  {
    int makes_compiler_happy;  /* not used */
  } ButtonClassPart;

   /* Full class record declaration */
typedef struct _ButtonClassRec {
    CoreClassPart	core_class;
    SimpleClassPart	simple_class;
    LabelClassPart	label_class;
    ButtonClassPart    button_class;
} ButtonClassRec;

extern ButtonClassRec buttonClassRec;

/***************************************
 *
 *  Instance (widget) structure 
 *
 **************************************/

    /* New fields for the Button widget record */
typedef struct {
    /* resources */
    Dimension   highlight_thickness;
    XtCallbackList callbacks;

    /* private state */
    Pixmap      	gray_pixmap;
    GC          	normal_GC;
    GC          	inverse_GC;
    Boolean     	set;
    XtButtonHighlight	highlighted;
#ifdef SHAPE
    /* more resources */
    int			shape_style;    
    Dimension		corner_round;
#endif
} ButtonPart;


/*    XtEventsPtr eventTable;*/


   /* Full widget declaration */
typedef struct _ButtonRec {
    CorePart         core;
    SimplePart	     simple;
    LabelPart	     label;
    ButtonPart      button;
} ButtonRec;

#endif /* _XawButtonP_h */


