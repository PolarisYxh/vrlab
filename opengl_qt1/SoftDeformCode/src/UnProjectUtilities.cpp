/*****************************************************************************

Copyright (c) 2004 SensAble Technologies, Inc. All rights reserved.

OpenHaptics(TM) toolkit. The material embodied in this software and use of
this software is subject to the terms and conditions of the clickthrough
Development License Agreement.

For questions, comments or bug reports, go to forums at: 
    http://dsc.sensable.com

Module Name: 

  UnProjectUtilities.cpp

Description:

  Utility class for going between 2D screen space (mouse positions) and 3D.

*******************************************************************************/

#include "UnProjectUtilities.h"

#if defined(WIN32)
# include <windows.h>
#endif
void UnProjectUtilities::GetMouseRay(int x, int y, 
                                     VECTOR3D &mouseNear, VECTOR3D &mouseFar)
{
    GLint viewport[4];
    //GLdouble mvMatrix[16], projMatrix[16];
    GLint realY; // OpenGL y coordinate position
    GLdouble wx, wy, wz; // returned world x, y, z coords
	GLdouble mvMatr[16], projMatr[16];
    glGetIntegerv(GL_VIEWPORT, viewport);
    glGetDoublev(GL_MODELVIEW_MATRIX, mvMatr);
    glGetDoublev(GL_PROJECTION_MATRIX, projMatr);

    // note viewport[3] is height of window in pixels
    realY = viewport[3] - (GLint) y - 1;

    gluUnProject(x, realY, 0.0, mvMatrix, projMatrix, viewport, &wx, &wy, &wz);
    mouseNear[0] = static_cast<double>(wx);
    mouseNear[1] = static_cast<double>(wy);
    mouseNear[2] = static_cast<double>(wz);
    gluUnProject(x, realY, 1.0, mvMatrix, projMatrix, viewport, &wx, &wy, &wz);
    mouseFar[0] = static_cast<double>(wx);
    mouseFar[1] = static_cast<double>(wy);
    mouseFar[2] = static_cast<double>(wz);
}

void UnProjectUtilities::GetMousePosition(int x, int y, VECTOR3D &mousePos)
{
    GLfloat mouseZ;
    GLint viewport[4];
    //GLdouble mvMatrix[16], projMatrix[16];
    GLint realY; // OpenGL y coordinate position
    GLdouble wx, wy, wz; // returned world x, y, z coords

	GLdouble mvMatr[16], projMatr[16];
    glGetIntegerv(GL_VIEWPORT, viewport);
    glGetDoublev(GL_MODELVIEW_MATRIX, mvMatr);//固定管线中使用
    glGetDoublev(GL_PROJECTION_MATRIX, projMatr);//固定管线中使用

    // note viewport[3] is height of window in pixels
    realY = viewport[3] - (GLint) y - 1;

    glReadPixels(x, realY, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &mouseZ);//读取以x,realy像素点为矩形左下角点，宽度为1个像素，高度为1个像素的矩形包含的像素点的深度值
    gluUnProject(x, realY, mouseZ, mvMatrix, projMatrix, viewport, &wx, &wy, &wz);
    mousePos[0] = static_cast<double>(wx);
    mousePos[1] = static_cast<double>(wy);
    mousePos[2] = static_cast<double>(wz);
}

double UnProjectUtilities::GetDistanceFromLine(const VECTOR3D &point, const VECTOR3D &x1, const VECTOR3D &x2)
{
    return (x2 - x1).CrossProduct(x1 - point).GetLength() / (x2 - x1).GetLength();
}

VECTOR3D UnProjectUtilities::GetLineIntersectPlaneZ(const VECTOR3D &x1, const VECTOR3D &x2, const double z)
{
    double r = (z - x1[2]) / (x2[2] - x1[2]);
        
    VECTOR3D intersection;

    intersection[0] = x1[0] + r * (x2[0] - x1[0]);
    intersection[1] = x1[1] + r * (x2[1] - x1[1]);
    intersection[2] = z;

    return intersection;
}

/******************************************************************************/
