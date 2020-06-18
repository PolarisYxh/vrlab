//////////////////////////////////////////////////////////////////////////////////////////
//	VECTOR3D.h
//	Class declaration for a 3d vector
//	Downloaded from: www.paulsprojects.net
//	Created:	20th July 2002
//	Modified:	8th November 2002	-	Changed Constructor layout
//									-	Some speed Improvements
//									-	Corrected Lerp
//				7th January 2003	-	Added QuadraticInterpolate
//
//	Copyright (c) 2006, Paul Baker
//	Distributed under the New BSD Licence. (See accompanying file License.txt or copy at
//	http://www.paulsprojects.net/NewBSDLicense.txt)
//////////////////////////////////////////////////////////////////////////////////////////	

#ifndef VECTOR3D_H
#define VECTOR3D_H

class VECTOR3D
{
public:
	//constructors
	VECTOR3D(void)	:	x(0.0f), y(0.0f), z(0.0f)
	{}

	VECTOR3D(double newX, double newY, double newZ)	:	x(newX), y(newY), z(newZ)
	{}

	VECTOR3D(const double * rhs)	:	x(*rhs), y(*(rhs+1)), z(*(rhs+2))
	{}

	VECTOR3D(const VECTOR3D & rhs)	:	x(rhs.x), y(rhs.y), z(rhs.z)
	{}

	~VECTOR3D() {}	//empty

	void Set(double newX, double newY, double newZ)
	{	x=newX;	y=newY;	z=newZ;	}
	
	//Accessors kept for compatibility
	void SetX(double newX) {x = newX;}
	void SetY(double newY) {y = newY;}
	void SetZ(double newZ) {z = newZ;}

	double GetX() const {return x;}	//public accessor functions
	double GetY() const {return y;}	//inline, const
	double GetZ() const {return z;}

	void LoadZero(void)
	{	x=y=z=0.0f;	}
	void LoadOne(void)
	{	x=y=z=1.0f;	}
	
	//vector algebra
	VECTOR3D CrossProduct(const VECTOR3D & rhs) const
	{	return VECTOR3D(y*rhs.z - z*rhs.y, z*rhs.x - x*rhs.z, x*rhs.y - y*rhs.x);	}

	double DotProduct(const VECTOR3D & rhs) const
	{	return x*rhs.x + y*rhs.y + z*rhs.z;	}
	
	void Normalize();
	VECTOR3D GetNormalized() const;
	
	double GetLength() const
	{	return (double)sqrt((x*x)+(y*y)+(z*z));	}
	
	double GetSquaredLength() const
	{	return (x*x)+(y*y)+(z*z);	}
	double distanceToPoint(const VECTOR3D& point) const
	{
		VECTOR3D tem(x - point.GetX(), y - point.GetY(), z - point.GetZ());
		return tem.GetLength();
	}
	double distanceToPointSqr(const VECTOR3D& point) const
	{
		VECTOR3D tem(x - point.GetX(), y - point.GetY(), z - point.GetZ());
		return tem.GetSquaredLength();
	}
	//rotations
	void RotateX(double angle);
	VECTOR3D GetRotatedX(double angle) const;
	void RotateY(double angle);
	VECTOR3D GetRotatedY(double angle) const;
	void RotateZ(double angle);
	VECTOR3D GetRotatedZ(double angle) const;
	void RotateAxis(double angle, const VECTOR3D & axis);
	VECTOR3D GetRotatedAxis(double angle, const VECTOR3D & axis) const;

	//pack to [0,1] for color
	void PackTo01();
	VECTOR3D GetPackedTo01() const;

	//linear interpolate
	VECTOR3D lerp(const VECTOR3D & v2, double factor) const
	{	return (*this)*(1.0f-factor) + v2*factor;	}

	VECTOR3D QuadraticInterpolate(const VECTOR3D & v2, const VECTOR3D & v3, double factor) const
	{	return (*this)*(1.0f-factor)*(1.0f-factor) + 2*v2*factor*(1.0f-factor) + v3*factor*factor;}


	//overloaded operators
	//binary operators
	VECTOR3D operator+(const VECTOR3D & rhs) const
	{	return VECTOR3D(x + rhs.x, y + rhs.y, z + rhs.z);	}
	
	VECTOR3D operator-(const VECTOR3D & rhs) const
	{	return VECTOR3D(x - rhs.x, y - rhs.y, z - rhs.z);	}

	VECTOR3D operator*(const double rhs) const
	{	return VECTOR3D(x*rhs, y*rhs, z*rhs);	}
	
	VECTOR3D operator/(const double rhs) const
	{	return (rhs==0.0f) ? VECTOR3D(0.0f, 0.0f, 0.0f) : VECTOR3D(x / rhs, y / rhs, z / rhs);	}

	//multiply by a float, eg 3*v
	friend VECTOR3D operator*(float scaleFactor, const VECTOR3D & rhs);

	//Add, subtract etc, saving the construction of a temporary
	void Add(const VECTOR3D & v2, VECTOR3D & result)
	{
		result.x=x+v2.x;
		result.y=y+v2.y;
		result.z=z+v2.z;
	}

	void Subtract(const VECTOR3D & v2, VECTOR3D & result)
	{
		result.x=x-v2.x;
		result.y=y-v2.y;
		result.z=z-v2.z;
	}

	bool operator==(const VECTOR3D & rhs) const;
	bool operator!=(const VECTOR3D & rhs) const
	{	return !((*this)==rhs);	}

	//self-add etc
	void operator+=(const VECTOR3D & rhs)
	{	x+=rhs.x;	y+=rhs.y;	z+=rhs.z;	}

	void operator-=(const VECTOR3D & rhs)
	{	x-=rhs.x;	y-=rhs.y;	z-=rhs.z;	}

	void operator*=(const double rhs)
	{	x*=rhs;	y*=rhs;	z*=rhs;	}
	
	void operator/=(const double rhs)
	{	if(rhs==0.0f)
			return;
		else
		{	x/=rhs; y/=rhs; z/=rhs;	}
	}
	double& operator[](int index)
	{
		if (index == 0)
			return x;
		else if (index == 1)
			return y;
		else if (index == 2)
			return z;
		else
			throw "index out of range£¡";
	}

	//unary operators
	VECTOR3D operator-(void) const {return VECTOR3D(-x, -y, -z);}
	VECTOR3D operator+(void) const {return *this;}

	//cast to pointer to a (double *) for glVertex3fv etc
	operator double* () const {return (double*) this;}
	//operator double* () const { return (double*)this; }
	//operator const double* () const {return (const double*) this;}
	//operator const double* () const { return (const double*)this; }
	//member variables
	double x;
	double y;
	double z;
};

#endif	//VECTOR3D_H