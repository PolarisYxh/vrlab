//////////////////////////////////////////////////////////////////////////////////////////
//	TIMER.cpp
//	Functions for timer class
//	Downloaded from: www.paulsprojects.net
//	Created:	20th July 2002
//
//	Copyright (c) 2006, Paul Baker
//	Distributed under the New BSD Licence. (See accompanying file License.txt or copy at
//	http://www.paulsprojects.net/NewBSDLicense.txt)
//////////////////////////////////////////////////////////////////////////////////////////
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "TIMER.h"
#include <time.h>
void TIMER::Reset()
{
	startTime=clock();
}

double TIMER::GetTime()
{
	if(isPaused)
		return pauseTime-startTime;
	else
		return clock() -startTime;
}

void TIMER::Pause()
{
	//Only pause if currently unpaused
	if(isPaused)
		return;

	isPaused=true;
	pauseTime= clock();
}

void TIMER::Unpause()
{
	//Only unpause if currently paused
	if(!isPaused)
		return;

	isPaused=false;

	//Update start time to reflect pause
	startTime+=(clock() -pauseTime);
}