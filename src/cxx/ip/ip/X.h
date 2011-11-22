/**
 * @file cxx/ip/ip/X.h
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
#ifndef X_INC
#define X_INC

#include "core/general.h"

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
