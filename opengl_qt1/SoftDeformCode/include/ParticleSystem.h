/*****************************************************************************

Copyright (c) 2004 SensAble Technologies, Inc. All rights reserved.

OpenHaptics(TM) toolkit. The material embodied in this software and use of
this software is subject to the terms and conditions of the clickthrough
Development License Agreement.

For questions, comments or bug reports, go to forums at: 
    http://dsc.sensable.com

Module Name: 

  ParticleSystem.h

Description:

  Set of particles with springs and other constraints.

*******************************************************************************/

#ifndef ParticleSystem_H_
#define ParticleSystem_H_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include <list>
#include <vector>
#include <iostream>
#include "DynamicsMath.h"
#include "Particle.h"
#include "OdeSolver.h"
//当你在头文件声明成员变量或成员函数时，如果只需要用到某个类的指针而不需要用到类的对象，那么就可以直接只是声明一下这个类，不用include，这样可以避免编译时include编译这个类。
//但是cpp实现文件里是需要include类的。
class Constraint;
class NailConstraint;
class SpringConstraint;
class HapticDeviceConstraint;
class MouseSpringConstraint;
class MouseForceConstraint;
typedef std::vector<Particle*> ParticleListT;
typedef std::list<Constraint*> ConstraintListT;

const int kDim = 3; // dimensions of the system

class ParticleSystem
{
public:
    ParticleSystem();
    ~ParticleSystem();

    Particle& AddParticle(double x = 0, double y = 0, double z = 0, 
                          double inMass = 0);
    void AddParticle(Particle* p);

	void SetParticle(double x, double y, double z, int i);

    void AddConstraint(Constraint* c);
    NailConstraint* AddNailConstraint(int p);
    SpringConstraint* AddSpringConstraint(int p1, int p2);
    SpringConstraint* AddSpringConstraint(int p1, int p2, float length);
	void AddMouseSpringConstraint(int inParticle);//, VECTOR3D inPosition);
	void AddMouseForceConstraint(int inParticle, VECTOR3D force);
    void SetSpringConstraint(int p1, int p2, int t1, int t2, float length);
	Particle& GetParticle(int j);//return particle on j
	void DeleteSpring(int p,int num);
    void DeleteNail(int p);

    void DeleteConstraint(Constraint* c);
	void deleteLastConstraint(int num);

    int GetClosestParticle(const VECTOR3D& pos);
    void GetClosestCutting(const VECTOR3D& pos, int pnum);
    void AddHapticDeviceConstraint(void);
	
   ////////////////////////////////phantom 1/////////////// 

    void ActivateHapticDeviceConstraint0(void);
    void DeactivateHapticDeviceConstraint0(void);
    void DeactivateForce0(int i);
    void DeactivateFix0();

    int HapticDeviceMove0(const VECTOR3D& pos,
                          const VECTOR3D& force);//top point

    void HapticMove0(int j,int i,const VECTOR3D& force);//neighbour circles, local deforamtion

    void HapticcrossMove0(int j, int i, int m, const VECTOR3D& force); //circles, for global deformation

    void HapticForce0(int i,const VECTOR3D& force);//collision detection, add constrain to new points with force 

    void HapticSetForce0(int i,const VECTOR3D& force);//collision detection, change force value to a points already with force 

    void HapticFix0(int i,const VECTOR3D& force);//adding tools left side, copy constraint

   ////////////////////////////////phantom 2/////////////// 

    void ActivateHapticDeviceConstraint1(void);
    void DeactivateHapticDeviceConstraint1(void);
    void DeactivateForce1(int i);
    void DeactivateFix1();

    int HapticDeviceMove1(const VECTOR3D& pos,
                          const VECTOR3D& force);

    void HapticMove1(int j,int i,const VECTOR3D& force);

    void HapticcrossMove1(int j, int i, int m, const VECTOR3D& force);

    void HapticForce1(int i,const VECTOR3D& force);

    void HapticSetForce1(int i,const VECTOR3D& force);

    void HapticFix1(int i,const VECTOR3D& force);

    /* For switching modes (design/simulation). */
    void StartConstructingSystem(void);
    void FinishConstructingSystem(void);
    void ClearSystem(void);
        
    void AdvanceSimulation(double tPrev, double tCurr);

    void SetSpringConstant(double inKS);
    void SetSpringDampingConstant(double inKD);
    double GetSpringConstant(void);
    double GetSpringDampingConstant(void);
	void Draw();
    ParticleListT particles;
    ConstraintListT constraints;
        
    float t;                // simulation clock
        
    int kind[80000][2];//0,nail; 1 spring; 2 haptics; 3 mouse;why 20000? because for this model, there are totally less then 20000 edges. if size of model increase, it also should increase
    long kindn;

	int index[3];//the nearst three vertices for contact point.

    float restitution;
    float drag;
    float gravity;

	int ncross;
        
    nvectord x0; // particles.size() * kStateSize;store position and velocity
    nvectord xFinal; // particles.size() * kStateSize);store position and velocity

    IOdeSolver *odeSolver;

         ///device 1
    HapticDeviceConstraint* hapticDeviceConstraint0;//top point of proxy,
    HapticDeviceConstraint* hapticFixConstraint0[76];// when adding tool, the top and other collision points
    HapticDeviceConstraint* hapticConstraint0[20];//first and second neigbour cirle of tip point. local deformation
    HapticDeviceConstraint* hapticforceConstraint0[75];//for collision detection points
    HapticDeviceConstraint* hapticcrossConstraint0[3][80];//for global deformation, circles 


         //device 2
    HapticDeviceConstraint* hapticDeviceConstraint1;
    HapticDeviceConstraint* hapticFixConstraint1[76];
    HapticDeviceConstraint* hapticConstraint1[20];
    HapticDeviceConstraint* hapticforceConstraint1[75];
    HapticDeviceConstraint* hapticcrossConstraint1[3][80];

	MouseSpringConstraint* mouseSpring;
	MouseForceConstraint* mouseForce;
    bool design;    // are we in design mode?
    bool useDepthForMouseZ;

private:
	VECTOR3D GetLineIntersectPlaneZ(const VECTOR3D& x1,
                                        const VECTOR3D& x2,
                                        const double z);
    void ParticlesStateToArray(double *dst);
    void ParticlesArrayToState(double *dst);
    static bool DxDt(double t, nvectord &x, nvectord &xdot, void *userData);
    void LimitStateChanges(void);
    void DdtParticlesStateToArray(double *xdot);
    void ClearForces(void);
    void ApplyRegularForces(void);
    void ApplyDragForces(void);
    void ApplyConstraintForces(void);
};

#endif // ParticleSystem_H_

/*****************************************************************************/



