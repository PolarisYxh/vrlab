/*****************************************************************************

Copyright (c) 2004 SensAble Technologies, Inc. All rights reserved.

OpenHaptics(TM) toolkit. The material embodied in this software and use of
this software is subject to the terms and conditions of the clickthrough
Development License Agreement.

For questions, comments or bug reports, go to forums at: 
    http://dsc.sensable.com

Module Name: 

  MouseSpringConstraint.cpp

Description:

  Implementation of particle system constraint to let you use the mouse
  to pull on a particle.

*******************************************************************************/

#include "MouseSpringConstraint.h"

MouseSpringConstraint::MouseSpringConstraint() :
    mParticle(0),
    mPosition(0,0,0),
    mLength(0),             // rest length
    ks(DEFAULT_MOUSE_KS),
    kd(DEFAULT_MOUSE_KD),
    mState(false)
{ 
}

MouseSpringConstraint::MouseSpringConstraint(int inParticle, VECTOR3D inPosition) :
    mParticle(inParticle),
    mPosition(inPosition),
    mLength(0),             // rest length
    ks(DEFAULT_MOUSE_KS),
    kd(DEFAULT_MOUSE_KD),
    mState(false)
{ 
}

MouseSpringConstraint::MouseSpringConstraint(int inParticle) :
	mParticle(inParticle),
	mPosition(0, 0, 0),
	mLength(0),             // rest length
	mforce(VECTOR3D(0, 2, 0)),
    ks(DEFAULT_MOUSE_KS),
    kd(DEFAULT_MOUSE_KD),
    mState(false)
{ 
}

MouseSpringConstraint::~MouseSpringConstraint()
{
}

void MouseSpringConstraint::ApplyConstraint(ParticleListT &particles)
{
    if (!mState)    // don't apply if not active
        return;
        
    Particle *p1 = particles[mParticle];
        
	VECTOR3D qdiff = p1->x - mPosition;
	VECTOR3D vdiff = p1->v; // FIXME: shouldn't we subtract the mouse velocity?
        
    double dist = qdiff.GetLength();
        
	VECTOR3D force = - ((ks * (dist - mLength)) + (kd * qdiff.DotProduct(vdiff) / dist)) * (qdiff / dist);

    if (!p1->fixed)
        p1->f += mforce;
}

void MouseSpringConstraint::Draw(ParticleListT &particles)
{
    if (!mState)    // don't draw if not active
        return;

    float c[] = { .5, 0, 1 };
    glColor3fv(c);

    Particle *p1 = particles[mParticle];

    glBegin(GL_LINES);
        
    glVertex3dv(p1->x);
    glVertex3dv(mPosition);
        
    glEnd();
}

/******************************************************************************/
