/*****************************************************************************

Copyright (c) 2004 SensAble Technologies, Inc. All rights reserved.

OpenHaptics(TM) toolkit. The material embodied in this software and use of
this software is subject to the terms and conditions of the clickthrough
Development License Agreement.

For questions, comments or bug reports, go to forums at: 
    http://dsc.sensable.com

Module Name: 

  ParticleSystem.cpp

Description:

  Set of particles with springs and other constraints.
*******************************************************************************/

#if defined(WIN32)
# include "windows.h"
#endif

#include <assert.h>
#include "SpringConstraint.h"
#include "DynamicsMath.h"
#include "NailConstraint.h"
#include "HapticDeviceConstraint.h"
#include "MouseSpringConstraint.h"
#include "MouseForceConstraint.h"
#include "ParticleSystem.h"

#if defined(WIN32) || defined(linux)
# include <GL/gl.h>
#elif defined(__APPLE__)
# include <OpenGL/gl.h>
#endif

#include "UnProjectUtilities.h" // wrapper function for gluUnProject()
#include <qvector3d.h>
// the number of values (double) needed to store the state for one particle
static const int kStateSize = 6;//x[3], position: x,y,z and v[3], vx,vy,vz, totally 6 parameters

static const double kGravityDef = -9.8f;

static const double kMaxVelocity = 10;
static const double kMaxAcceleration = 100;

ParticleSystem::ParticleSystem()
{
    restitution = .5;//摩擦力
    drag = 4.5f;//0.03,空气阻力系数
    gravity = 0.0f; //9.8;
    t = 0;
        
    kindn = 0;
    hapticDeviceConstraint0 = NULL;
    hapticDeviceConstraint1 = NULL;   
	mouseSpring = NULL;
	mouseForce = NULL;
    design = true;
    ncross = 80;

    //odeSolver = new OdeSolverEuler();
    odeSolver = new OdeSolverRungeKutta4();//
}

ParticleSystem::~ParticleSystem()
{
    ClearSystem();
    delete odeSolver;
}

// Adds a new particle with position (x, y, z) and returns a reference to the
// particle.
// If there is no space left for a new particle, a reference to the last
// particle is returned, but no new particle is created.
Particle& ParticleSystem::AddParticle(double x, double y, double z, double inMass)
{
    assert(design);
        
    Particle* part;
    if (inMass == 0)
        part = new Particle();
    else
        part = new Particle(inMass);
        
    part->x[0] = x;
    part->x[1] = y;
    part->x[2] = z;
        
    particles.push_back(part);
        
    return *part;
}

void ParticleSystem::SetParticle(double x, double y, double z, int i)
{      
	particles[i]->x[0] = x;
	particles[i]->x[1] = y;
    particles[i]->x[2] = z; 
}


void ParticleSystem::AddParticle(Particle* p)
{
    assert(design);
    particles.push_back(p);
}

void ParticleSystem::AddConstraint(Constraint* c)
{
    assert(design);
        
    constraints.push_back(c);
}

NailConstraint* ParticleSystem::AddNailConstraint(int p)
{
    assert(design);
        
    // rely on FlexToSystem to fix the constraint
    NailConstraint* nc = new NailConstraint(p);
        
    AddConstraint(nc);

	kind[kindn][0] = p;
	kind[kindn][1] = -1;
	kindn++;
        
    return nc;
}

void ParticleSystem::DeleteNail(int p)
{
	int i = 0,j;
	int q1, q2;

    typedef ConstraintListT::const_iterator LI; // constant because not modifying list
    LI ci;
/*
	ci = constraints.end();
	ci--;
    constraints.erase(ci);
    kindn--;
*/

    for (ci = constraints.begin(); ci != constraints.end(); ++ci,i++)
    {  
		 q1 = kind[i][0];
		 q2 = kind[i][1];

		if(q1==p&&q2==-1)
		{ 	          
         constraints.erase(ci);//remove(ci);
         delete *ci;

         for(j=i+1;j<kindn;j++)
		 {
			kind[j-1][0] = kind[j][0];
			kind[j-1][1] = kind[j][1];
		 }
         kindn--;
		 break;
		}//ifq1, q2
    }//for
	
}

SpringConstraint* ParticleSystem::AddSpringConstraint(int p1, int p2)
{
    assert(design);
	if (p1 == p2)return NULL;
    // rely on FlexToSystem to fix the constraint
    SpringConstraint* sc = new SpringConstraint(p1, p2);
        
    AddConstraint(sc);

	kind[kindn][0] = p1;
	kind[kindn][1] = p2;
	kindn++;
  
    return sc;
}

SpringConstraint* ParticleSystem::AddSpringConstraint(int p1, int p2, float length)
{
    assert(design);
        
    SpringConstraint* sc = new SpringConstraint(p1, p2, length);
        
    AddConstraint(sc);
        
	kind[kindn][0] = p1;
	kind[kindn][1] = p2;
	kindn++;

    return sc;
}

void ParticleSystem::DeleteSpring(int p, int num)//p is the particle index to delete, num is a fix value for a given model, total number of edges for model
{
	int i = 0,j;
	int q1, q2;

    typedef ConstraintListT::const_iterator LI; // constant because not modifying list
    LI ci;

    if(constraints.size()<=num)
		return;

    for (ci = constraints.begin(); ci != constraints.end(); ++ci,i++)
    {  
		if(i>=num)
	  {
		 q1 = kind[i][0];
		 q2 = kind[i][1];

		if(q1==p)
		{ 	          
         constraints.erase(ci);//remove(ci);
         delete *ci;

         for(j=i+1;j<kindn;j++)
		 {
			kind[j-1][0] = kind[j][0];
			kind[j-1][1] = kind[j][1];
		 }
         kindn--;
		 break;
		}//ifq1, q2
	  }//if
    }//for 
}

void ParticleSystem::DeleteConstraint(Constraint* c)
{
    assert(c != NULL);
        
    constraints.remove(c);
	kindn--;
        
    delete c;
}
void ParticleSystem::AddMouseSpringConstraint(int inParticle)//inparticle:the vertex mouse pushed,inposition:the position mouse released
{
	assert(design);

	mouseSpring = new MouseSpringConstraint(inParticle);

	mouseSpring->SetState(true);

	AddConstraint(mouseSpring);
	//// rely on FlexToSystem to fix the constraint
	//SpringConstraint* sc = new SpringConstraint(p1, p2);

	//AddConstraint(sc);

	//kind[kindn][0] = p1;
	//kind[kindn][1] = p2;
	//kindn++;

	//return sc;
}
void ParticleSystem::AddMouseForceConstraint(int inParticle, VECTOR3D force)//inparticle:the vertex mouse pushed,inposition:the position mouse released
{
	//assert(design);

	mouseForce = new MouseForceConstraint(inParticle, force);

	mouseForce->SetState(true);

	AddConstraint(mouseForce);
	//// rely on FlexToSystem to fix the constraint
	//SpringConstraint* sc = new SpringConstraint(p1, p2);

	//AddConstraint(sc);

	//kind[kindn][0] = p1;
	//kind[kindn][1] = p2;
	//kindn++;

	//return sc;
}
void ParticleSystem::deleteLastConstraint(int num)
{
	assert(design);
	constraints.pop_back();
	constraints.pop_back();
}
void ParticleSystem::AddHapticDeviceConstraint(void)
{
	int i,j;
    assert(design);
        
    hapticDeviceConstraint0 = new HapticDeviceConstraint;
        
    hapticDeviceConstraint0->SetState(false);
        
    AddConstraint(hapticDeviceConstraint0);

	kind[kindn][0] = -1;
	kind[kindn][1] = -1;
	kindn++;
///////////////////////////////
	for(i=0;i<75;i++)
	{
    hapticforceConstraint0[i] = new HapticDeviceConstraint;
        
    hapticforceConstraint0[i]->SetState(false);
       
    AddConstraint(hapticforceConstraint0[i]);

	kind[kindn][0] = -1;
	kind[kindn][1] = -1;
	kindn++;
	}
///////////////////////////////////////

	for(i=0;i<76;i++)
	{
    hapticFixConstraint0[i] = new HapticDeviceConstraint;
        
    hapticFixConstraint0[i]->SetState(false);
        
    AddConstraint(hapticFixConstraint0[i]);

	kind[kindn][0] = -1;
	kind[kindn][1] = -1;
	kindn++;
	}
//////////////////////////////////////////
    hapticDeviceConstraint1 = new HapticDeviceConstraint;
        
    hapticDeviceConstraint1->SetState(false);
        
    AddConstraint(hapticDeviceConstraint1);

	kind[kindn][0] = -1;
	kind[kindn][1] = -1;
	kindn++;

///////////////////////////////

	for(i=0;i<75;i++)
	{
    hapticforceConstraint1[i] = new HapticDeviceConstraint;
        
    hapticforceConstraint1[i]->SetState(false);
        
    AddConstraint(hapticforceConstraint1[i]);

	kind[kindn][0] = -1;
	kind[kindn][1] = -1;
	kindn++;
	}

///////////////////////////////////////

	for(i=0;i<76;i++)
	{
    hapticFixConstraint1[i] = new HapticDeviceConstraint;
        
    hapticFixConstraint1[i]->SetState(false);
        
    AddConstraint(hapticFixConstraint1[i]);

	kind[kindn][0] = -1;
	kind[kindn][1] = -1;
	kindn++;
	}
//////////////////////////////////////////////
    for(i=0;i<20;i++)
	{
    hapticConstraint0[i] = new HapticDeviceConstraint;     
    hapticConstraint0[i]->SetState(false);     
    AddConstraint(hapticConstraint0[i]);

	kind[kindn][0] = -1;
	kind[kindn][1] = -1;
	kindn++;
	}

    for(i=0;i<20;i++)
	{
    hapticConstraint1[i] = new HapticDeviceConstraint;     
    hapticConstraint1[i]->SetState(false);     
    AddConstraint(hapticConstraint1[i]);

	kind[kindn][0] = -1;
	kind[kindn][1] = -1;
	kindn++;
	}
	/////////////////////cross section///////////////
//////////////////////////////////////////////
     for(j=0;j<3;j++)
    for(i=0;i<ncross;i++)
	{
    hapticcrossConstraint0[j][i] = new HapticDeviceConstraint;     
    hapticcrossConstraint0[j][i]->SetState(false);     
    AddConstraint(hapticcrossConstraint0[j][i]);

	kind[kindn][0] = -1;
	kind[kindn][1] = -1;
	kindn++;
	}

     for(j=0;j<3;j++)
    for(i=0;i<ncross;i++)
	{
    hapticcrossConstraint1[j][i] = new HapticDeviceConstraint;     
    hapticcrossConstraint1[j][i]->SetState(false);     
    AddConstraint(hapticcrossConstraint1[j][i]);

	kind[kindn][0] = -1;
	kind[kindn][1] = -1;
	kindn++;
	}
	i = i;
}

// Find the particle nearest to 3D position
int ParticleSystem::GetClosestParticle(const VECTOR3D& pos)
{
    int closestPart = -1;
    double closestDist, currDist;   
        
    typedef ParticleListT::const_iterator LI; // constant because not modifying list
    LI pi;
    int i;
        
    for (pi = particles.begin(), i = 0; pi != particles.end(); ++pi, i++)
    {
        Particle& p = **pi;
        assert(&p != NULL);
        currDist = pos.distanceToPointSqr(p.x);                
        if (closestPart == -1 || (currDist < closestDist))
        {
            closestDist = currDist;
            closestPart = i;
        }
    }
        
    return closestPart;
}

void ParticleSystem::GetClosestCutting(const VECTOR3D& pos, int pnum)//searching the nearst three vertices for the contact point, used for cutting and subdivision cutting
{
    double closestDist0 = 1.0; 
	double closestDist1 = 1.0;
	double closestDist2 = 1.0;
	double currDist;   
        
    typedef ParticleListT::const_iterator LI; // constant because not modifying list
    LI pi;
    int i;
	int mini;

	index[0] = 0;
	index[1] = 0;
	index[2] = 0;

    for (pi = particles.begin(), i = 0; i<pnum; ++pi, i++)
    {
        Particle&p = **pi;
        assert(&p != NULL);
                
        currDist = pos.distanceToPointSqr(p.x);                
        if(closestDist0>currDist)
        {
        index[2] = index[1];
        closestDist2 = closestDist1;

        index[1] = index[0];
        closestDist1 = closestDist0;

        index[0] = i;
        closestDist0 = currDist;

		mini = i;
        }
		else if(closestDist1>currDist)
        {
        index[2] = index[1];
        closestDist2 = closestDist1;

        index[1] = i;
        closestDist1 = currDist;
        }
		else if(closestDist2>currDist)
        {
        index[2] = i;
        closestDist2 = currDist;
        }
    }

}


///////////////////////////////////////////////////////////////////haptic////////////////////////////
void ParticleSystem::ActivateHapticDeviceConstraint0(void)
{
    assert(hapticDeviceConstraint0 != NULL);
    hapticDeviceConstraint0->SetForce(VECTOR3D(0,0,0));
    hapticDeviceConstraint0->SetState(true);

	int i,j;
    for(i=0;i<20;i++)
	{
    hapticConstraint0[i]->SetForce(VECTOR3D(0,0,0));   
    hapticConstraint0[i]->SetState(true);     
	}

    for(j=0;j<3;j++)
    for(i=0;i<ncross;i++)
	{
    hapticcrossConstraint0[j][i]->SetForce(VECTOR3D(0,0,0));   
    hapticcrossConstraint0[j][i]->SetState(true);     
	}
	
}


void ParticleSystem::ActivateHapticDeviceConstraint1(void)
{
    assert(hapticDeviceConstraint1 != NULL);
    hapticDeviceConstraint1->SetForce(VECTOR3D(0,0,0));
    hapticDeviceConstraint1->SetState(true);

	int i,j;
    for(i=0;i<20;i++)
	{
    hapticConstraint1[i]->SetForce(VECTOR3D(0,0,0));   
    hapticConstraint1[i]->SetState(true);     
	}

    for(j=0;j<3;j++)
    for(i=0;i<ncross;i++)
	{
    hapticcrossConstraint1[j][i]->SetForce(VECTOR3D(0,0,0));   
    hapticcrossConstraint1[j][i]->SetState(true);     
	}
	
}

void ParticleSystem::DeactivateHapticDeviceConstraint0(void)
{
    assert(hapticDeviceConstraint0 != NULL);
    hapticDeviceConstraint0->SetState(false);

	int i,j;
    for(i=0;i<20;i++) 
    hapticConstraint0[i]->SetState(false);     

    for(j=0;j<3;j++)
    for(i=0;i<ncross;i++) 
    hapticcrossConstraint0[j][i]->SetState(false);   
}

void ParticleSystem::DeactivateForce0(int i)
{
    int j;

    for(j=0;j<75;j++)
	{
	 if(hapticforceConstraint0[j]->mState&&hapticforceConstraint0[j]->mParticle==i)
	 {
      hapticforceConstraint0[j]->SetState(false);
	  break;
	 }
	}

}

void ParticleSystem::DeactivateHapticDeviceConstraint1(void)
{
    assert(hapticDeviceConstraint1 != NULL);
    hapticDeviceConstraint1->SetState(false);

	int i,j;
    for(i=0;i<20;i++) 
    hapticConstraint1[i]->SetState(false); 

    for(j=0;j<3;j++)
    for(i=0;i<ncross;i++) 
    hapticcrossConstraint1[j][i]->SetState(false); 
}

void ParticleSystem::DeactivateForce1(int i)
{
    int j;

    for(j=0;j<75;j++)
	{
	 if(hapticforceConstraint1[j]->mState&&hapticforceConstraint1[j]->mParticle==i)
	 {
      hapticforceConstraint1[j]->SetState(false);
	  break;
	 }
	}
}
void ParticleSystem::DeactivateFix0()
{
    assert(hapticFixConstraint0!= NULL);
	int j;

     for(j=0;j<76;j++)
	 hapticFixConstraint0[j]->SetState(false);  
}

void ParticleSystem::DeactivateFix1()
{
    assert(hapticFixConstraint1!= NULL);
	int j;

     for(j=0;j<76;j++)
	 hapticFixConstraint1[j]->SetState(false);  
}

int  ParticleSystem::HapticDeviceMove0(const VECTOR3D& pos,
                                      const VECTOR3D& force)
{
    assert(hapticDeviceConstraint0 != NULL);

    int i,closestParticle = GetClosestParticle(pos);

    hapticDeviceConstraint0->SetParticle(closestParticle);
    hapticDeviceConstraint0->SetForce(force);

	return closestParticle;
}

int  ParticleSystem::HapticDeviceMove1(const VECTOR3D& pos,
                                      const VECTOR3D& force)
{
    assert(hapticDeviceConstraint1 != NULL);

    int i,closestParticle = GetClosestParticle(pos);

    hapticDeviceConstraint1->SetParticle(closestParticle);
    hapticDeviceConstraint1->SetForce(force);

	return closestParticle;
}

void  ParticleSystem::HapticMove0(int j, int i,const VECTOR3D& force)                                  
{   
    hapticConstraint0[j]->SetParticle(i);
    hapticConstraint0[j]->SetForce(force);
}

void  ParticleSystem::HapticMove1(int j, int i,const VECTOR3D& force)                                  
{      
    hapticConstraint1[j]->SetParticle(i);
    hapticConstraint1[j]->SetForce(force);
}

void  ParticleSystem::HapticcrossMove0(int j, int i,int m, const VECTOR3D& force)                                  
{   
    hapticcrossConstraint0[m][j]->SetParticle(i);
    hapticcrossConstraint0[m][j]->SetForce(force);
}

void  ParticleSystem::HapticcrossMove1(int j, int i,int m, const VECTOR3D& force)                                  
{      
    hapticcrossConstraint1[m][j]->SetParticle(i);
    hapticcrossConstraint1[m][j]->SetForce(force);
}

void  ParticleSystem::HapticForce0(int i,const VECTOR3D& force)                                  
{
    int j;

    for(j=0;j<75;j++)
	{
		 if(!hapticforceConstraint0[j]->mState)
		 {
			 hapticforceConstraint0[j]->SetForce(force);
			 hapticforceConstraint0[j]->SetParticle(i);
			 hapticforceConstraint0[j]->SetState(true);
			 break;
		 }
	}
 
}

void  ParticleSystem::HapticSetForce0(int i,const VECTOR3D& force)                                  
{
    int j;

    for(j=0;j<75;j++)
	{
	 if(!hapticforceConstraint0[j]->mParticle==i)
	 {
     hapticforceConstraint0[j]->SetForce(force);
	 hapticforceConstraint0[j]->SetParticle(i);
     hapticforceConstraint0[j]->SetState(true);
	 break;
	 }
	}
}

void  ParticleSystem::HapticForce1(int i,const VECTOR3D& force)                                  
{   
    int j;

    for(j=0;j<75;j++)
	{
	 if(!hapticforceConstraint1[j]->mState)
	 {
	 hapticforceConstraint1[j]->SetParticle(i);
     hapticforceConstraint1[j]->SetForce(force);
	 hapticforceConstraint1[j]->SetState(true);
	 break;
	 }
	} 
}

void  ParticleSystem::HapticSetForce1(int i,const VECTOR3D& force)                                  
{
    int j;

    for(j=0;j<75;j++)
	{
	 if(!hapticforceConstraint1[j]->mParticle==i)
	 {   
	 hapticforceConstraint1[j]->SetParticle(i);
     hapticforceConstraint1[j]->SetForce(force);
	 hapticforceConstraint1[j]->SetState(true);
	 break;
	 }
	}
}

void ParticleSystem::HapticFix0(int i,const VECTOR3D& force)
{   
	assert(hapticFixConstraint0 != NULL);
    hapticFixConstraint0[75]->SetForce(force);
	hapticFixConstraint0[75]->SetState(true);
    hapticFixConstraint0[75]->SetParticle(i);
    

	 int j,k;
	 bool b;
     VECTOR3D c;

     for(j=0;j<75;j++)
	{
     b = hapticforceConstraint0[j]->mState;
	 c = hapticforceConstraint0[j]->mForce;
     k = hapticforceConstraint0[j]->mParticle;

     hapticFixConstraint0[j]->SetForce(c);
	 hapticFixConstraint0[j]->SetState(b);
	 hapticFixConstraint0[j]->SetParticle(k);   
	}
}

void ParticleSystem::HapticFix1(int i,const VECTOR3D& force)
{   
    assert(hapticFixConstraint0 != NULL);
    hapticFixConstraint1[75]->SetForce(force);
	hapticFixConstraint1[75]->SetState(true);
    hapticFixConstraint1[75]->SetParticle(i);
    

	 int j,k;
	 bool b;
     VECTOR3D c;

     for(j=0;j<75;j++)
	{
     b = hapticforceConstraint1[j]->mState;
	 c = hapticforceConstraint1[j]->mForce;
     k = hapticforceConstraint1[j]->mParticle;

     hapticFixConstraint1[j]->SetForce(c);
	 hapticFixConstraint1[j]->SetState(b);
	 hapticFixConstraint1[j]->SetParticle(k);   
	}
}

void ParticleSystem::StartConstructingSystem(void)
{
      
    // delete the hapticDeviceConstraint if there is one
    if (hapticDeviceConstraint0 != NULL)
    {
        constraints.remove(hapticDeviceConstraint0);
        delete hapticDeviceConstraint0;
        hapticDeviceConstraint0 = NULL;
    }
    
    if (hapticDeviceConstraint1 != NULL)
    {
        constraints.remove(hapticDeviceConstraint1);
        delete hapticDeviceConstraint1;
        hapticDeviceConstraint1 = NULL;
    }
	if (mouseSpring != NULL)
	{
		constraints.remove(mouseSpring);
		delete mouseSpring;
		mouseSpring = NULL;
	}
	if (mouseForce!= NULL)
	{
		//constraints.remove(mouseForce);
		constraints.pop_back();
		constraints.pop_back();
		constraints.pop_back();
		delete mouseForce;
		mouseForce = NULL;
	}
    design = true;
}

void ParticleSystem::FinishConstructingSystem(void)
{
    unsigned int i;
    //AddHapticDeviceConstraint();
           
    x0.resize(particles.size() * kStateSize);
    xFinal.resize(particles.size() * kStateSize);
        
        // clear any velocities
    for (i = 0; i < particles.size(); i++)
    {
        particles[i]->v = VECTOR3D(0,0,0);
        particles[i]->f = VECTOR3D(0,0,0);
    }

    ParticlesStateToArray(&xFinal[0]);

    odeSolver->setSize(particles.size() * kStateSize);

    typedef ConstraintListT::const_iterator LI; // constant because not modifying list
    LI ci;
	int m = 0;
    for (ci = constraints.begin(); ci != constraints.end(); ++ci)
    {
        Constraint& c = **ci;
        assert(&c != NULL);
        c.FlexToSystem(particles);
    }
        
    design = false;
	kindn = kindn;
}
void ParticleSystem::Draw(void)
{
	typedef ConstraintListT::const_iterator LI; // constant because not modifying list

	LI ci;

	for (ci = constraints.begin(); ci != constraints.end(); ++ci)
	{
		Constraint& c = **ci;
		assert(&c != NULL);

		c.Draw(particles);
	}

	Particle::BeginDraw();

	for (unsigned int i = 0; i < particles.size(); i++)
	{
		particles[i]->Draw();
	}

	Particle::EndDraw();

}

void ParticleSystem::SetSpringConstraint(int p1, int p2, int t1, int t2, float length)
{
	int i = 0,j;
	int q1, q2;

    typedef ConstraintListT::const_iterator LI; // constant because not modifying list
    LI ci;


    for (ci = constraints.begin(); ci != constraints.end(); ++ci,i++)
    {  
		q1 = kind[i][0];
		q2 = kind[i][1];

		if(q1==p1&&q2==p2||q1==p2&&q2==p1)
		{             
         constraints.erase(ci);
		 delete *ci; 

         SpringConstraint* sc = new SpringConstraint(t1, t2, length); 
         AddConstraint(sc);

         for(j=i+1;j<kindn;j++)
		 {
			kind[j-1][0] = kind[j][0];
			kind[j-1][1] = kind[j][1];
		 }

		kind[kindn-1][0] = t1;
		kind[kindn-1][1] = t2;
		 break;
		}//if
    }//for 
}
Particle& ParticleSystem::GetParticle(int j)
{
	return *particles.at(j);
}
void ParticleSystem::ClearSystem(void)
{
    typedef ConstraintListT::iterator CLI; // constant because not modifying list
        
    CLI ci;
        
    while (constraints.size() > 0)
    {
        ci = constraints.begin();
                
        Constraint& c = **ci;
        assert(&c != NULL);
                
        delete *ci;
                
        constraints.erase(ci);
    }
        
    hapticDeviceConstraint0 = NULL;
    hapticDeviceConstraint1 = NULL;

    typedef ParticleListT::iterator PLI; // constant because not modifying list
        
    PLI pi;

    while (particles.size() > 0)
    {
        pi = particles.begin();
                
        Particle& p = **pi;
        assert(&p != NULL);
                
        delete *pi;

        particles.erase(pi);
    }
        
    assert(particles.size() == 0);
    assert(constraints.size() == 0);
}


void ParticleSystem::AdvanceSimulation(double tPrev, double tCurr)
{
    // copy xFinal back to x0
    for(unsigned int i=0; i<kStateSize * particles.size(); i++)
      x0[i] = xFinal[i];

    //ode(x0, xFinal, kStateSize * mBodies.size(), tPrev, tCurr, dxdt);
    odeSolver->solve(x0, xFinal, tPrev, tCurr, DxDt, this);//x0存储当前位置和速度，将计算结果送入xFinal
        
    // copy d/dt X(tNext) into state variables
    ParticlesArrayToState(&xFinal[0]);
}

bool ParticleSystem::DxDt(double t, nvectord &x, nvectord &xdot, void *userData)//加约束算力，并且把速度和加速度送入xdot数组
{
    ParticleSystem *pThis = static_cast<ParticleSystem *>(userData);
    assert(pThis);
        
    // Put data in x[] into particles
    pThis->ParticlesArrayToState(&x[0]);
        
    pThis->ClearForces();

    // evaluate all forces
	pThis->ApplyRegularForces();//add gravity
    pThis->ApplyDragForces();
    pThis->ApplyConstraintForces();

    pThis->LimitStateChanges();//if exceed max velocity,set max velocity

    pThis->DdtParticlesStateToArray(&xdot[0]);//把速度和加速度送入xdot数组

    return false;
}

void ParticleSystem::ParticlesStateToArray(double *dst)
{
    typedef ParticleListT::const_iterator LI; // constant because not modifying list
        
    LI pi;

    for (pi = particles.begin(); pi != particles.end(); ++pi)
    {
        Particle& p = **pi;
        assert(&p != NULL);

        *(dst++) = p.x[0];
        *(dst++) = p.x[1];
        *(dst++) = p.x[2];
        *(dst++) = p.v[0];
        *(dst++) = p.v[1];
        *(dst++) = p.v[2];
    }
}

void ParticleSystem::ParticlesArrayToState(double *dst)
{
    ParticleListT::const_iterator pi; // constant because not modifying list

    for (pi = particles.begin(); pi != particles.end(); ++pi)
    {
        Particle& p = **pi;
        assert(&p != NULL);
                
        p.x[0] = *(dst++);
        p.x[1] = *(dst++);
        p.x[2] = *(dst++);
        p.v[0] = *(dst++);
        p.v[1] = *(dst++);
        p.v[2] = *(dst++);
    }
}

void ParticleSystem::ClearForces(void)
{
    for (unsigned int i = 0; i < particles.size(); i++)
    {
        particles[i]->f = VECTOR3D(0,0,0);
    }
}

void ParticleSystem::ApplyRegularForces(void)
{
    Particle *p;

    for (unsigned int i = 0; i < particles.size(); i++)
    {
        p = particles[i];
        p->f[1] -= gravity * p->mass;
    }
}

void ParticleSystem::ApplyDragForces(void)
{
    Particle *p;

    for (unsigned int i = 0; i < particles.size(); i++)
    {
        p = particles[i];
        p->f -= p->v * drag;
    }
}

void ParticleSystem::ApplyConstraintForces(void)
{
    typedef ConstraintListT::const_iterator LI; // constant because not modifying list
        
    LI ci;

    unsigned int i = constraints.size();

    for (ci = constraints.begin(); ci != constraints.end(); ++ci)
    {
        Constraint& c = **ci;
        assert(&c != NULL); 
        c.ApplyConstraint(particles);
    }
}

#if LIMIT_CHANGES_PER_PARTICLE // an experimental alternative
// For each particle, make sure the max v and f aren't exceeded
void ParticleSystem::LimitStateChanges(void)
{
    ParticleListT::const_iterator pi; // constant because not modifying list
    double vMag;
    double fMag;
        
    for (pi = particles.begin(); pi != particles.end(); ++pi)
    {
        Particle& p = **pi;
        assert(&p != NULL);

        vMag = norm(p.v);
        if (vMag > kMaxVelocity)
            p.v *= kMaxVelocity / vMag;

        fMag = norm(p.f);
        if (fMag > kMaxAcceleration * p.mass)
            p.f *= kMaxAcceleration * p.mass / fMag;
    }
}
#endif

// Scale back the v and f of all particles to ensure
// the max values aren't exceeded, effectively 
// reducing the size of the time step taken
void ParticleSystem::LimitStateChanges(void)
{
    ParticleListT::const_iterator pi; // constant because not modifying list
    double vMag, vBiggest = 0;
    double fMag, fBiggest = 0;

    // take advantage of the fact that all particles have the same mass
    double maxForce = kMaxAcceleration * Particle::mass;

    for (pi = particles.begin(); pi != particles.end(); ++pi)
    {
        Particle& p = **pi;
        assert(&p != NULL);

        vMag = p.v.GetLength();
        vBiggest = max(vMag, vBiggest);

        fMag = p.f.GetLength();
        fBiggest = max(fMag, fBiggest);
    }

    if (vBiggest > kMaxVelocity)
    {
        for (pi = particles.begin(); pi != particles.end(); ++pi)
        {
            (*pi)->v *= kMaxVelocity / vBiggest;
        }
    }

    if (fBiggest > maxForce)
    {
        for (pi = particles.begin(); pi != particles.end(); ++pi)
        {
            (*pi)->f *= maxForce / fBiggest;
        }
    }
}

void ParticleSystem::DdtParticlesStateToArray(double *xdot)
{
    ParticleListT::const_iterator pi; // constant because not modifying list

    for (pi = particles.begin(); pi != particles.end(); ++pi)
    {
        Particle& p = **pi;
        assert(&p != NULL);

        int i;

        // copy d/dt x(t) = v(t) into xdot
        for(i=0; i<3; i++)
            *(xdot++) = p.v[i];

        // copy d/dt v(t) = f(t) / m into xdot
        for(i=0; i<3; i++)
            *(xdot++) = p.f[i] / p.mass;
    }
}

void ParticleSystem::SetSpringConstant(double inKS)
{
    SpringConstraint::SetSpringConstant(inKS);
}

void ParticleSystem::SetSpringDampingConstant(double inKD)
{
    SpringConstraint::SetSpringDampingConstant(inKD);
}

double ParticleSystem::GetSpringConstant(void)
{
    return SpringConstraint::GetSpringConstant();
}

double ParticleSystem::GetSpringDampingConstant(void)
{
    return SpringConstraint::GetSpringDampingConstant();
}

/******************************************************************************/
