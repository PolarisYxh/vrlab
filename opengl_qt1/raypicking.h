#ifndef RAYPICKING_H
#define RAYPICKING_H

#include <QOpenGLFunctions_3_3_Core>
#include <QVector3D>
#include <QVector4D>
#include <QVector>
#include <QDebug>
#include <QMatrix4x4>
#include <QtAlgorithms>

class RayPicking{
public:
  RayPicking();
  ~RayPicking();
  void addRay(float x, float y);
  int picking(float x, float y, QVector<QVector<QVector3D>> vec_triangles, QVector3D& ray_dir);
  void drawRay();
  void transData(const QMatrix4x4& projection, const QMatrix4x4& view, const QVector3D& cameraPos);
private:
  QVector3D getRay(float x, float y);
  QVector<float> checkTriCollision(QVector3D ray, QVector<QVector<QVector3D>> vec_triangles); //检查射线与一堆三角面的碰撞关系
  float checkSingleTriCollision(QVector3D ray, QVector<QVector3D> vec_triangle); //检查射线与一个三角面的碰撞关系
  QOpenGLFunctions_3_3_Core *core;
  GLuint vertexVBO;
  QVector<QVector3D> vec_ray;
  QMatrix4x4 projection, view;
  QVector3D cameraPos;
};

#endif // RAYPICKING_H
