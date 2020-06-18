/*****************************************************************************

Copyright (c) my copy
*******************************************************************************/

#include "MouseForceConstraint.h"

MouseForceConstraint::MouseForceConstraint() :
    mParticle(0),
    mPosition(0,0,0),
    mLength(0),             // rest length
    ks(DEFAULT_MOUSE_KS),
    kd(DEFAULT_MOUSE_KD),
    mState(false)
{ 
}
MouseForceConstraint::MouseForceConstraint(int inParticle,VECTOR3D force) :
	mParticle(inParticle),
	mPosition(0, 0, 0),
	mLength(0),             // rest length
	mforce(force),
    ks(DEFAULT_MOUSE_KS),
    kd(DEFAULT_MOUSE_KD),
    mState(false)
{ 
}
MouseForceConstraint::MouseForceConstraint(int inParticle) :
	mParticle(inParticle),
	mPosition(0, 0, 0),
	mLength(0),             // rest length
	mforce(VECTOR3D(0, 2, 0)),
	ks(DEFAULT_MOUSE_KS),
	kd(DEFAULT_MOUSE_KD),
	mState(false)
{
}
MouseForceConstraint::~MouseForceConstraint()
{
}

void MouseForceConstraint::ApplyConstraint(ParticleListT &particles)
{
    if (!mState)    // don't apply if not active
        return;
        
    Particle *p1 = particles[mParticle];
    if (!p1->fixed)
        p1->f += mforce;
}

void MouseForceConstraint::Draw(ParticleListT &particles)
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
