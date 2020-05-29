//////////////////////////////////////////////////////////////////////////////////////////
//	SPRING.h
//	Class for a spring in our "ball and spring" cloth model
//	Downloaded from: www.paulsprojects.net
//	Created:	12th January 2003
//
//	Copyright (c) 2006, Paul Baker
//	Distributed under the New BSD Licence. (See accompanying file License.txt or copy at
//	http://www.paulsprojects.net/NewBSDLicense.txt)
//////////////////////////////////////////////////////////////////////////////////////////	

#ifndef SPRING_H
#define SPRING_H

#include "BALL.h"

class SPRING
{
public:
	//Indices of the balls at either end of the spring
	int ball1;
	int ball2;

	//Tension in the spring
	float tension;

	float springConstant;
	float naturalLength;

	SPRING()	:	ball1(-1), ball2(-1)
	{}
	~SPRING()
	{}
};

#endif	//SPRING_H