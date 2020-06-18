#include "learnopengl/model.h"
#include "ParticleSystem.h"
#include "learnopengl/mesh.h"
#include "learnopengl/shader.h"
#include "TIMER.h"
#include "GLM.h"
#include <QVector2D>
#include <QVector3D>
#include <QMatrix4x4>
#include <QPoint>
#include "raypicking.h"
class Obj2MassSpring 
{
public:
	//Obj2MassSpring(Model& mod) :model(mod)
	//{
	//	ps = new ParticleSystem();
	//};
	Obj2MassSpring(GLMmodel* mod,Model &model1) :glmmodel(mod),model(model1)
	{
		ps = new ParticleSystem();
		ray = new RayPicking();
	};
	void MassSpring();
	void SetParticleSystem(ParticleSystem* inPs) { ps = inPs; }
	void updatePS();
	void MouseDrag(int point, VECTOR3D position);
	int MouseClicked(VECTOR3D position);
	void RayPick(QPoint pos,QMatrix4x4 modma,QMatrix4x4 proma,QMatrix4x4 viewma, QVector3D posit, int width, int height,bool drag);
private:
	Model& model;
	GLMmodel* glmmodel;
	//Mesh *mesh;
	ParticleSystem* ps;
	TIMER timer;
	//for raypickings
	RayPicking* ray;
	//QMatrix4x4 modelMtr, projectMtr,viewMtr;
	//QVector3D pos;
	QVector<QVector<QVector3D>> vec_triangles;

};