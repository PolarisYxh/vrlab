// CArcBall.h: interface for the CArcBall class.
//
//////////////////////////////////////////////////////////////////////

#ifndef ArcBall_H
#define ArcBall_H

#pragma once

#include <QPoint>
#include <QVector3D>
#include <QVector4D>
#include <QQuaternion>
#include <QMatrix4x4>
#include <QKeyEvent>

class CArcBall  
{
public:
	void reset();
	CArcBall();
	virtual ~CArcBall();

	/* Public routines */
	void place(QPoint &vecPos, double dRadius);
	void setMouse(QPoint &vecNow);
	void startDragging();
	void finishDragging();

	void processMouseMovement(const QPoint& p);
	void press(const QPoint& p);
	QMatrix4x4 getRotatonMatrix();
	//QMatrix4x4 getMatrix4x4();
	float getDragDirection();

//private:
public:
	QVector3D convertMouseToSphere(QPoint vec2Mouse);
	QQuaternion getQuaternion(QVector3D vec3From, QVector3D vec3To);
	
    double        m_dRadius;

	QPoint     m_vec2Center;
	QPoint     m_vec2Now, m_vec2Down,m_vec2Last;
	QVector3D       m_vec3To,  m_vec3From;
	QQuaternion   m_quatNow, m_quatDown, m_quatDrag;

    bool m_bDragging;
};

#endif // !defined(_CArcBall_H__INCLUDED_)
