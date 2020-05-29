//////////////////////////////////////////////////////////////////////////////////////////
//	BALL.h
//	Class for a ball in our "ball and spring" cloth model
//	Downloaded from: www.paulsprojects.net
//	Created:	12th January 2003
//
//	Copyright (c) 2006, Paul Baker
//	Distributed under the New BSD Licence. (See accompanying file License.txt or copy at
//	http://www.paulsprojects.net/NewBSDLicense.txt)
//////////////////////////////////////////////////////////////////////////////////////////	

#ifndef BALL_H
#define BALL_H
#include <QVector3D>
#include "VECTOR3D.h"
//Make sure these structures are packed to a multiple of 4 bytes (sizeof(float)),
//since glMap2 takes strides as integer multiples of 4 bytes.
//How consistent...
#pragma pack(push)
#pragma pack(4)

class BALL
{
public:
	//QVector3D position;
	//QVector3D velocity;

	//float mass;

	////Is this ball held in position?
	//bool fixed;

	////Vertex normal for this ball
	//QVector3D normal;
	VECTOR3D position;
	VECTOR3D velocity;

	float mass;

	//Is this ball held in position?
	bool fixed;

	//Vertex normal for this ball
	VECTOR3D normal;
};

#pragma pack(pop)

#endif	//BALL_H
