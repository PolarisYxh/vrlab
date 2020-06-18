/*****************************************************************************

Copyright (c) 2004 SensAble Technologies, Inc. All rights reserved.

OpenHaptics(TM) toolkit. The material embodied in this software and use of
this software is subject to the terms and conditions of the clickthrough
Development License Agreement.

For questions, comments or bug reports, go to forums at: 
    http://dsc.sensable.com

Module Name: 

  SpringConstraint.cpp

Description:

  Implementation of particle constraint representing spring between two
  particles.

*******************************************************************************/

#include "SpringConstraint.h"

static const double kDefaultKS = 10;//10
static const double kDefaultKD =  1.2;//1.2

double SpringConstraint::ks = kDefaultKS;
double SpringConstraint::kd = kDefaultKD;

SpringConstraint::SpringConstraint(int inParticle1, int inParticle2, double inLength) :
    mParticle1(inParticle1),
    mParticle2(inParticle2),
    mLength(inLength), // rest length
    mHasFixedLength(true)
{
}

SpringConstraint::SpringConstraint(int inParticle1, int inParticle2) :
    mParticle1(inParticle1),
    mParticle2(inParticle2),
    mHasFixedLength(false)
{   
}

SpringConstraint::~SpringConstraint()
{
}

void SpringConstraint::ApplyConstraint(ParticleListT &particles)
{
    Particle *p1 = particles[mParticle1];
    Particle *p2 = particles[mParticle2];
        
    // TODO: this function could be optimized
	VECTOR3D qdiff = p1->x - p2->x;
	VECTOR3D vdiff = p1->v - p2->v;
        
    double dist = qdiff.GetLength();
	double k1,k2;

	k1 = dist - mLength;//deta length; the displacement of spring length

	if(fabs(k1)<0.0001)
		k1 = 0.0;
   
	if(dist==0.0)
		k2 = 0.0;
	else
		k2 = qdiff.DotProduct(vdiff) /dist;//the projection of velocity to spring vector
        
    VECTOR3D force = - (ks * k1 + 
                            kd * k2)* (qdiff / dist);

    // FIXME: += and -= don't currently work here
    if (!p1->fixed)
        p1->f += force;

    if (!p2->fixed)
        p2->f -= force;
		
}


void SpringConstraint::FlexToSystem(ParticleListT &particles)
{
	if (mHasFixedLength) return;

    Particle *p1 = particles[mParticle1];
    Particle *p2 = particles[mParticle2];

    mLength = (p1->x - p2->x).GetLength();
}
void SpringConstraint::Draw(ParticleListT& particles)
{
	if (!ParticlesAreValid()) return;

	float c[] = { 0, 0, 1 };
	glColor3fv(c);

	Particle* p1 = particles[mParticle1];
	Particle* p2 = particles[mParticle2];

	glBegin(GL_LINES);

	glVertex3dv(p1->x);
	glVertex3dv(p2->x);

	glEnd();
}
/******************************************************************************/
