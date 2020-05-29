#pragma once
#include "BALL.h"
#include <iostream>
#include <qvector3d.h>
#include <QVector4D>
#include "SPRING.h"
#include <glad/glad.h> 
#include <QDebug>
#include "TIMER.h"
#include "COLOR.h"
#include <GL/glu.h>
#include "VECTOR3D.h"
//#include <helper_cuda.h>
//extern "C"
//#include <cuda_runtime.h>
//#include <device_launch_parameters.h>
class MassSpringCuda
{
public:
	MassSpringCuda() {};
	MassSpringCuda(float width, float height) {
		//set viewport
		if (height == 0)
			height = 1;

		//Set up projection matrix
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluPerspective(45.0f, (float)width / height, 1.0f, 100.0f);

		//Load identity modelview
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		//Shading states
		QVector4D backgroundColor(0.0f, 0.0f, 0.0f, 0.0f);
		const QVector4D white(1.0f, 1.0f, 1.0f, 1.0f);
		const float whitef[4] = { 1,1,1,1 };
		glShadeModel(GL_SMOOTH);
		glClearColor(backgroundColor.x(), backgroundColor.y(), backgroundColor.z(), backgroundColor.w());
		glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
		glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

		//Depth states
		glClearDepth(1.0f);
		glDepthFunc(GL_LEQUAL);
		glEnable(GL_DEPTH_TEST);

		//Set up light
		glLightfv(GL_LIGHT1, GL_POSITION, whitef);
		glLightfv(GL_LIGHT1, GL_DIFFUSE, whitef);
		float color[4] = { 0.2,0.2,0.2,0.2 };
		glLightfv(GL_LIGHT1, GL_AMBIENT, color);
		glLightfv(GL_LIGHT1, GL_SPECULAR, whitef);
		glEnable(GL_LIGHT1);

		//Use 2-sided lighting
		glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, true);
	};
	~MassSpringCuda();
	bool DemoInit();
	void ResetCloth();

	void UpdateFrame();
	void RenderFrame(double currentTime, double timePassed);
	void setfixed(int point, bool isfix);
private:
	GLfloat* ToFloatPoint(QVector3D vec);
	//Grid complexity. This is the number of balls across and down in the model
	const int gridSize = 13;
	COLOR backgroundColor = COLOR(0.0f, 0.0f, 0.0f, 0.0f);
	//2 arrays of balls
	int numBalls;
	BALL* balls1 = NULL;
	BALL* balls2 = NULL;

	//Pointers to the arrays. One holds the balls for the current frame, and one holds those
	//for the next frame
	BALL* currentBalls = NULL;
	BALL* nextBalls = NULL;

	//Gravity
	//QVector3D gravity=QVector3D(0.0f, -0.98f, 0.0f);
	VECTOR3D gravity = VECTOR3D(0.0f, -0.98f, 0.0f);
	//Values given to each spring
	float springConstant = 15.0f;
	float naturalLength = 1.0f;

	//Values given to each ball
	float mass = 0.01f;

	//Damping factor. Velocities are multiplied by this
	float dampFactor = 0.9f;

	//Array of springs
	int numSprings;
	SPRING* springs = NULL;

	//What do we want to draw?
	bool drawBalls = false, drawSprings = false;
	bool drawTriangles = false, drawPatches = true;

	//Sphere
	float sphereRadius = 4.0f;

	//Floor texture
	GLuint floorTexture;

	//How tesselated is the patch?
	int patchTesselation = 5;
	TIMER timer;
};