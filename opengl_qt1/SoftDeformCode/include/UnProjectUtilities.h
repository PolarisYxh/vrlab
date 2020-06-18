/*****************************************************************************

Copyright (c) 2004 SensAble Technologies, Inc. All rights reserved.

OpenHaptics(TM) toolkit. The material embodied in this software and use of
this software is subject to the terms and conditions of the clickthrough
Development License Agreement.

For questions, comments or bug reports, go to forums at: 
    http://dsc.sensable.com

Module Name: 

  UnProjectUtilities.h

Description:

  Utility class for going between 2D screen space (mouse positions) and 3D.

*******************************************************************************/

#ifndef UnProjectUtilities_H_
#define UnProjectUtilities_H_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "DynamicsMath.h"
#include <QMatrix4x4>
#if defined(WIN32) || defined(linux)
# include <glad/glad.h>
# include <GL/glu.h>
#elif defined(__APPLE__)
# include <OpenGL/gl.h>
# include <OpenGL/glu.h>
#endif

class UnProjectUtilities  
{
public:
	UnProjectUtilities(QMatrix4x4 mvMa, QMatrix4x4 projMa)
	{
		//mvMa.copyDataTo((float *)mvMatrix);
		//projMa.copyDataTo((float*)projMatrix);
		for(int i=0;i<4;i++)
			for (int j = 0; j < 4; j++)
			{
				mvMatrix[i * 4 + j] = mvMa.data()[i * 4 + j];
				projMatrix[i * 4 + j] = projMa.data()[i * 4 + j];
			}
	};
    void GetMousePosition(int x, int y, VECTOR3D&mousePos);
    void GetMouseRay(int x, int y, 
                            VECTOR3D &mouseNear, VECTOR3D &mouseFar);
    static double GetDistanceFromLine(const VECTOR3D &point, 
                                      const VECTOR3D &x1, 
                                      const VECTOR3D &x2);
    static VECTOR3D GetLineIntersectPlaneZ(const VECTOR3D &x1, 
                                               const VECTOR3D &x2, 
                                               const double z);
	GLdouble mvMatrix[16];
	GLdouble projMatrix[16];
};

#endif // UnProjectUtilities_H_

/******************************************************************************/
