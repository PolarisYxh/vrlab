// CArcBall.cpp: implementation of the CArcBall class.
//
//////////////////////////////////////////////////////////////////////

#include "ArcBall.h"
#include <math.h>

#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[]=__FILE__;
#define new DEBUG_NEW
#endif

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CArcBall::CArcBall() :
m_bDragging(false),
m_dRadius(800.0),
m_vec2Center(0.0,0.0),
m_vec2Down(0.0,0.0),
m_vec2Now(0.0,0.0),
m_vec2Last(0.0,0.0),
m_vec3From(0.0, 0.0, 0.0),
m_vec3To(0.0, 0.0, 0.0)
{
	
	
}

CArcBall::~CArcBall()
{
	
}

void CArcBall::place(QPoint &vec2Center, double dRadius)
{
	m_vec2Center = vec2Center;
	m_dRadius = dRadius;
}

void CArcBall::setMouse(QPoint &vec2Now)
{
	m_vec2Last = m_vec2Now;
	m_vec2Now = vec2Now;
}


/* Begin drag sequence. */
void CArcBall::startDragging()
{
    m_bDragging = true;
    m_vec2Down   = m_vec2Now;
}


/* Stop drag sequence. */
void CArcBall::finishDragging()
{
    m_bDragging = false;
    m_quatDown  = m_quatNow;
}
void CArcBall::press(const QPoint &p)
{
	m_vec2Down = p;
}
void CArcBall::processMouseMovement(const QPoint& p)
{
		m_vec3From = convertMouseToSphere(m_vec2Down);
        m_vec3To   = convertMouseToSphere(p);
		m_quatDrag = getQuaternion(m_vec3From, m_vec3To);
		m_quatNow  = m_quatDrag * m_quatDown;
		m_quatDown = m_quatNow;
	
}
/* Return rotation matrix defined by controller use. */
QMatrix4x4 CArcBall::getRotatonMatrix()
{
	return QMatrix4x4(m_quatNow.toRotationMatrix());
}
/* Convert 2D window coordinates to coordinates on the 3D unit sphere. */
QVector3D CArcBall::convertMouseToSphere(QPoint vec2Mouse)//QPoint的值是int型
{
	QVector2D vec2UnitMouse((((double)vec2Mouse.x() - (double)m_vec2Center.x())) / m_dRadius, (((double)vec2Mouse.y() - (double)m_vec2Center.y())) / m_dRadius);
    double dDragRadius = vec2UnitMouse.lengthSquared();

	QVector3D vec3BallMouse(QVector4D(vec2UnitMouse.x(), vec2UnitMouse.y(), 0.0,1.0));

    if (dDragRadius > 1.0) {
		// the mouse position was outside the sphere
		// -> map the mouse position to the circle
		vec3BallMouse /= sqrt(dDragRadius);
		vec3BallMouse[2] = 0.0;
    } else {
		// compute the z-value of the unit sphere
		vec3BallMouse[2] = sqrt(1.0 - dDragRadius);
    }
	//vec3BallMouse.normalize();
    return (vec3BallMouse);
}


/* Construct a unit quaternion from two points on unit sphere */
QQuaternion
CArcBall::getQuaternion(QVector3D vecFrom, QVector3D vecTo)
{
	return QQuaternion(
		vecFrom[0] * vecTo[0] + vecFrom[1] * vecTo[1] + vecFrom[2] * vecTo[2],
		vecFrom[1] * vecTo[2] - vecFrom[2]*vecTo[1],
		vecFrom[2] * vecTo[0] - vecFrom[0]*vecTo[2],
		vecFrom[0] * vecTo[1] - vecFrom[1]*vecTo[0]
		);
}


void CArcBall::reset()
{
	m_bDragging = false;
    m_quatDown=QQuaternion(0.0, 0.0, 0.0, 1.0);
    m_quatNow = QQuaternion(0.0, 0.0, 0.0, 1.0);
	m_vec2Now  = m_vec2Center;
	m_vec2Down = m_vec2Center;
}

float CArcBall::getDragDirection()
{   
	//static CVector viewUpDir(0.f,1.f,0.f,0.f);
	//static CVector viewRightDir(0.f,0.f,1.f,0.f);
	//CMatrix matViewRotation = getRotatonMatrix();
	//viewUpDir  = matViewRotation * viewUpDir;
 //   viewRightDir = matViewRotation * viewRightDir;
	//CVector dragDirection = viewRightDir*(m_vec3To[0]-m_vec3From[0]) - viewUpDir*(m_vec3To[1]-m_vec3From[1]);
   
	m_vec3From = convertMouseToSphere(m_vec2Down);
	QVector3D tempvec3From = convertMouseToSphere(m_vec2Last);
    m_vec3To   = convertMouseToSphere(m_vec2Now);
	QVector3D temp = m_vec3To - tempvec3From;
	
	float dis = float(temp.length());
	if(m_vec2Now.x()>m_vec2Last.x())
		dis = -dis;

	return dis;
}