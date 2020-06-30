extern "C"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <iostream>
#include <helper_cuda.h>
#include "Windows.h"
#include <math.h>
using namespace std;
#define gridSize 40

//The first (gridSize-1)*gridSize springs go from one ball to the next,
//excluding those on the right hand edge
void main()
{
	//The first (gridSize-1)*gridSize springs go from one ball to the next,
	//excluding those on the right hand edge
	for (int i = 0; i < gridSize; ++i)
	{
		for (int j = 0; j < gridSize - 1; ++j)
		{
			currentSpring->ball1 = i * gridSize + j;
			currentSpring->ball2 = i * gridSize + j + 1;

			currentSpring->springConstant = springConstant;
			currentSpring->naturalLength = naturalLength;

			++currentSpring;
		}
	}

	//The next (gridSize-1)*gridSize springs go from one ball to the one below,
	//excluding those on the bottom edge
	for (int i = 0; i < gridSize - 1; ++i)
	{
		for (int j = 0; j < gridSize; ++j)
		{
			currentSpring->ball1 = i * gridSize + j;
			currentSpring->ball2 = (i + 1) * gridSize + j;

			currentSpring->springConstant = springConstant;
			currentSpring->naturalLength = naturalLength;

			++currentSpring;
		}
	}

	//The next (gridSize-1)*(gridSize-1) go from a ball to the one below and right
	//excluding those on the bottom or right
	for (int i = 0; i < gridSize - 1; ++i)
	{
		for (int j = 0; j < gridSize - 1; ++j)
		{
			currentSpring->ball1 = i * gridSize + j;
			currentSpring->ball2 = (i + 1) * gridSize + j + 1;

			currentSpring->springConstant = springConstant;
			currentSpring->naturalLength = naturalLength * sqrt(2.0f);

			++currentSpring;
		}
	}

	//The next (gridSize-1)*(gridSize-1) go from a ball to the one below and left
	//excluding those on the bottom or right
	for (int i = 0; i < gridSize - 1; ++i)
	{
		for (int j = 1; j < gridSize; ++j)
		{
			currentSpring->ball1 = i * gridSize + j;
			currentSpring->ball2 = (i + 1) * gridSize + j - 1;

			currentSpring->springConstant = springConstant;
			currentSpring->naturalLength = naturalLength * sqrt(2.0f);

			++currentSpring;
		}
	}

	//The first (gridSize-2)*gridSize springs go from one ball to the next but one,
	//excluding those on or next to the right hand edge
	for (int i = 0; i < gridSize; ++i)
	{
		for (int j = 0; j < gridSize - 2; ++j)
		{
			currentSpring->ball1 = i * gridSize + j;
			currentSpring->ball2 = i * gridSize + j + 2;

			currentSpring->springConstant = springConstant;
			currentSpring->naturalLength = naturalLength * 2;

			++currentSpring;
		}
	}

	//The next (gridSize-2)*gridSize springs go from one ball to the next but one below,
	//excluding those on or next to the bottom edge
	for (int i = 0; i < gridSize - 2; ++i)
	{
		for (int j = 0; j < gridSize; ++j)
		{
			currentSpring->ball1 = i * gridSize + j;
			currentSpring->ball2 = (i + 2) * gridSize + j;

			currentSpring->springConstant = springConstant;
			currentSpring->naturalLength = naturalLength * 2;

			++currentSpring;
		}
	}
	//Calculate the tensions in the springs
//计算弹簧压力
	for (int i = 0; i < numSprings; ++i)
	{//弹簧长度 = 前一个弹簧的位置 - 下一个弹簧的位置
		float springLength = (currentBalls[springs[i].ball1].position -
			currentBalls[springs[i].ball2].position).GetLength();
		//伸展 = 现在长度 - 原始长度
		float extension = springLength - springs[i].naturalLength;
		//压力 = 弹性系数 * 伸展 / 自然长度
		springs[i].tension = springs[i].springConstant * extension / springs[i].naturalLength;
	}
	//Loop through springs
	for (int j = 0; j < numSprings; ++j)
	{
		//If this ball is "ball1" for this spring, add the tension to the force
		if (springs[j].ball1 == i)
		{
			VECTOR3D tensionDirection = currentBalls[springs[j].ball2].position -
				currentBalls[i].position;
			tensionDirection.Normalize();

			//把力加上去
			force += springs[j].tension * tensionDirection;
		}

		//Similarly if the ball is "ball2"
		if (springs[j].ball2 == i)
		{
			VECTOR3D tensionDirection = currentBalls[springs[j].ball1].position -
				currentBalls[i].position;
			tensionDirection.Normalize();

			force += springs[j].tension * tensionDirection;
		}
	}
	//计算加速度
//Calculate the acceleration
	VECTOR3D acceleration = force / currentBalls[i].mass;


	//更新速度
	//Update velocity
	nextBalls[i].velocity = currentBalls[i].velocity + acceleration *
		(float)timePassedInSeconds;
	//减少速度
	//Damp the velocity
	nextBalls[i].velocity *= dampFactor;


	//Calculate new position
	//计算新的位置
	nextBalls[i].position = currentBalls[i].position +
		(nextBalls[i].velocity + currentBalls[i].velocity) * (float)timePassedInSeconds / 2;
	//Check against sphere (at origin)
//计算球体
	if (nextBalls[i].position.GetSquaredLength() < sphereRadius * 1.08f * sphereRadius * 1.08f)
		nextBalls[i].position = nextBalls[i].position.GetNormalized() *
		sphereRadius * 1.08f;

	//Check against floor
	if (nextBalls[i].position.y < -8.5f)
		nextBalls[i].position.y = -8.5f;
	//Calculate the normals on the current balls
	for (int i = 0; i < gridSize - 1; ++i)
	{
		for (int j = 0; j < gridSize - 1; ++j)
		{
			VECTOR3D& p0 = currentBalls[i * gridSize + j].position;
			VECTOR3D& p1 = currentBalls[i * gridSize + j + 1].position;
			VECTOR3D& p2 = currentBalls[(i + 1) * gridSize + j].position;
			VECTOR3D& p3 = currentBalls[(i + 1) * gridSize + j + 1].position;

			VECTOR3D& n0 = currentBalls[i * gridSize + j].normal;
			VECTOR3D& n1 = currentBalls[i * gridSize + j + 1].normal;
			VECTOR3D& n2 = currentBalls[(i + 1) * gridSize + j].normal;
			VECTOR3D& n3 = currentBalls[(i + 1) * gridSize + j + 1].normal;

			//Calculate the normals for the 2 triangles and add on
			VECTOR3D normal = (p1 - p0).CrossProduct(p2 - p0);

			n0 += normal;
			n1 += normal;
			n2 += normal;

			normal = (p1 - p2).CrossProduct(p3 - p2);

			n1 += normal;
			n2 += normal;
			n3 += normal;
		}
	}

	//Normalize normals
	for (int i = 0; i < numBalls; ++i)
		currentBalls[i].normal.Normalize();
}
}