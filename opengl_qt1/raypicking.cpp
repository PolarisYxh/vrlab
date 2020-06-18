#include "raypicking.h"

RayPicking::RayPicking(): vertexVBO(0){
  core = QOpenGLContext::currentContext()->versionFunctions<QOpenGLFunctions_3_3_Core>();
}

RayPicking::~RayPicking(){
  if(vertexVBO != 0)
    core->glDeleteBuffers(1, &vertexVBO);
}

void RayPicking::addRay(float x, float y){
  if(vertexVBO != 0)
    core->glDeleteBuffers(1, &vertexVBO);

  QVector3D ray_dir = getRay(x, y);

  this->vec_ray.push_back(cameraPos);
  this->vec_ray.push_back(cameraPos + ray_dir*100.0f);


  core->glGenBuffers(1, &vertexVBO);
  core->glBindBuffer(GL_ARRAY_BUFFER, vertexVBO);
  core->glBufferData(GL_ARRAY_BUFFER, sizeof(QVector3D) * vec_ray.size(), &vec_ray[0], GL_DYNAMIC_DRAW);
}

int RayPicking::picking(float x, float y, QVector<QVector<QVector3D> > vec_triangles,QVector3D &ray_dir){
  /*
   * 传进来一堆三角形，判断射线 和 这些三角形的碰撞情况，如果碰撞，选距离cameraPos最近的三角形的索引，进行返回
  */
  ray_dir = getRay(x, y);
//  qDebug() << "ray: " << ray_dir;
  QVector<float> vec_t = checkTriCollision(ray_dir, vec_triangles);

  /*
   * 比较t的大小，找出最小的一个，即距离camerapos最近的一个三角形面
   *
   */
  QVector<float> vec_comt;
  for(auto iter: vec_t){
    if(iter > 0.0f)
      vec_comt.push_back(iter);
  }

  qSort(vec_comt.begin(), vec_comt.end()); //从小到大排序
  if(vec_comt.isEmpty())
    return -1;
  else{
    return vec_t.indexOf(vec_comt[0]);
  }
}

void RayPicking::drawRay(){
  if(vertexVBO != 0){
    core->glBindBuffer(GL_ARRAY_BUFFER, vertexVBO);
    core->glEnableVertexAttribArray(0);
    core->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);

    core->glDrawArrays(GL_LINES, 0, vec_ray.size());
  }
}

void RayPicking::transData(const QMatrix4x4 &projection, const QMatrix4x4 &view, const QVector3D& cameraPos){
  this->projection = projection;
  this->view = view;
    this->cameraPos = cameraPos;
}

QVector3D RayPicking::getRay(float x, float y){
  /*
   * 计算射线
   */
	QVector3D ray_nds = QVector3D(x, y, 1.0f);
	QVector4D ray_clip = QVector4D(ray_nds, 1.0f);
	QVector4D ray_eye = projection.inverted() * ray_clip;
	QVector4D ray_world = view.inverted() * ray_eye;

	if (ray_world.w() != 0.0f) { //齐次坐标转 笛卡尔坐标
		ray_world.setX(ray_world.x() / ray_world.w());
		ray_world.setY(ray_world.y() / ray_world.w());
		ray_world.setZ(ray_world.z() / ray_world.w());
	}

	QVector3D ray_dir = (QVector3D(ray_world.x(), ray_world.y(), ray_world.z()) - cameraPos).normalized();

	return ray_dir;
}

QVector<float> RayPicking::checkTriCollision(QVector3D ray, QVector<QVector<QVector3D> > vec_triangles){
  QVector<float> vec_t;

//  qDebug() << "tris: " << vec_triangles;
  for(int i = 0; i < vec_triangles.size(); ++i){
    float t = checkSingleTriCollision(ray, vec_triangles[i]);
    vec_t.push_back(t);
  }

//  qDebug() << "vec_t: "<< vec_t;
  return vec_t;
}

float RayPicking::checkSingleTriCollision(QVector3D ray, QVector<QVector3D> vec_triangle){
  QVector3D E1 =  vec_triangle[1] - vec_triangle[0];
  QVector3D E2 =  vec_triangle[2] - vec_triangle[0];

  QVector3D P = QVector3D::crossProduct(ray, E2);
  float det = QVector3D::dotProduct(P, E1);
  QVector3D T;
  if(det > 0)
     T = this->cameraPos - vec_triangle[0];
  else{
     T = vec_triangle[0] - this->cameraPos;
     det = -det;
  }

  if(det < 0.00001f) //表示射线与三角面所在的平面平行，返回不相交
    return -1.0f;

  /******* 相交则判断 交点是否落在三角形面内 *********/
  float u = QVector3D::dotProduct(P, T);
  if(u < 0.0f || u > det)
    return -1.0f;

  QVector3D Q = QVector3D::crossProduct(T, E1);
  float v = QVector3D::dotProduct(Q, ray);
  if(v < 0.0f || u+v > det)
    return -1.0f;

  float t = QVector3D::dotProduct(Q, E2);
  if(t < 0.0f)
    return -1.0f;

  return t/det;
}
