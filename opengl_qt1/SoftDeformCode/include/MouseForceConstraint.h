/*****************************************************************************

Copyright (c) my copy
*******************************************************************************/

//#if !defined(AFX_MOUSESPRINGCONSTRAINT_H__A6A078ED_7494_45BF_8CF7_CD62301E3429__INCLUDED_)
//#define AFX_MOUSESPRINGCONSTRAINT_H__A6A078ED_7494_45BF_8CF7_CD62301E3429__INCLUDED_
#ifndef MouseForceConstraint_H_
#define MouseForceConstraint_H_
#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "Constraint.h"
class MouseForceConstraint : public Constraint
{
public:
	#define DEFAULT_MOUSE_KS 10
	#define DEFAULT_MOUSE_KD 1

	MouseForceConstraint();
	MouseForceConstraint(int inParticle,VECTOR3D force);
	MouseForceConstraint(int inParticle);
	virtual ~MouseForceConstraint();

	virtual void ApplyConstraint(ParticleListT &particles);
	virtual void Draw(ParticleListT &particles);

	virtual void FlexToSystem(ParticleListT &particles) { }

	void SetState(bool inState) { mState = inState; }
	void SetPosition(VECTOR3D inPosition) { mPosition = inPosition; }
	void SetParticle(int inParticle) { mParticle = inParticle; }
	void SetForce(VECTOR3D force) { mforce = force; }
	int GetParticle(void) { return mParticle; }

private:
	int mParticle;
	VECTOR3D mPosition;
	float mLength;	// usually 0
	VECTOR3D mforce;
	float ks;		// spring stiffness
	float kd;		// damping constant
	
	bool mState; // on or off
};

#endif // !defined(AFX_MOUSESPRINGCONSTRAINT_H__A6A078ED_7494_45BF_8CF7_CD62301E3429__INCLUDED_)
