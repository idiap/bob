#ifndef X_INC
#define X_INC

#include "general.h"

#ifdef HAVE_X11
// Xm stuff
#include <Xm/Xm.h>
#include <Xm/Text.h>
#include <Xm/RowColumn.h>
#include <Xm/Frame.h>
#include <Xm/Form.h>
#include <Xm/Label.h>
#include <Xm/LabelG.h>
#include <Xm/ToggleBG.h>
#include <Xm/PushB.h>
#include <Xm/PushBG.h>
#include <Xm/CascadeBG.h>
#include <Xm/ScrolledW.h>
#include <Xm/FileSB.h>
#include <Xm/RepType.h>
#include <Xm/DialogS.h>
#include <Xm/List.h>
#include <Xm/MainW.h>
#include <Xm/PushB.h>
#include <Xm/ScrollBar.h>
#include <Xm/SelectioB.h>
#include <Xm/ArrowBG.h>
#include <Xm/MessageB.h>
#include <Xm/TextF.h>
#include <Xm/SeparatoG.h>

// X11 stuff
#include <X11/Xlib.h>
#include <X11/Intrinsic.h>
#include <X11/Xatom.h>
#include <X11/Intrinsic.h>
#include <X11/Shell.h>
#include <X11/Xos.h>
#include <X11/keysym.h>
#include <X11/cursorfont.h>

// Xpm
#include <X11/xpm.h>

//
#define NO_XMSTRINGS

#endif // HAVE_X11

#endif 
