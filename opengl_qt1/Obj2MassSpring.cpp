#include "Obj2MassSpring.h"
void Obj2MassSpring::MassSpring()
{
	ps->StartConstructingSystem();
	ps->ClearSystem();
	model.meshes[0].vertices.clear();
	for (int i = 0; i <= glmmodel->numvertices; i++)//第0个是空点
	{
		//qDebug() << glmmodel->vertices[3 * i] << " " << glmmodel->vertices[3 * i + 1] << " " << glmmodel->vertices[3 * i + 2];
		Vertex v;
		v.Position=glm::vec3(glmmodel->vertices[3 * i], glmmodel->vertices[3 * i + 1], glmmodel->vertices[3 * i + 2]);
		model.meshes[0].vertices.push_back(v);
		ps->AddParticle(glmmodel->vertices[3 * i], glmmodel->vertices[3 * i + 1], glmmodel->vertices[3 * i + 2]);
	}
	model.meshes[0].indices.clear();
	for (int i = 0; i < glmmodel->numtriangles; i++)
	{
		GLMtriangle& triangle = glmmodel->triangles[i];
		for (int j = 0; j < 3; j++)
		{
			Vertex& v = model.meshes[0].vertices[triangle.vindices[j]];
			v.TexCoords = glm::vec2(glmmodel->texcoords[2 * triangle.tindices[j]], glmmodel->texcoords[2 * triangle.tindices[j] + 1]);//stbi读入的图片y轴0坐标与opengl规定的相反
			v.Normal = glm::vec3(glmmodel->normals[3 * triangle.nindices[j]], glmmodel->normals[3 * triangle.nindices[j] + 1], glmmodel->normals[3 * triangle.nindices[j] + 2]);
		}
		GLuint index0 = triangle.vindices[0];
		GLuint index1 = triangle.vindices[1];
		GLuint index2 = triangle.vindices[2];
		model.meshes[0].indices.push_back(index0);
		model.meshes[0].indices.push_back(index1);
		model.meshes[0].indices.push_back(index2);
		double lenth1 = (ps->GetParticle(index0).x - ps->GetParticle(index1).x).GetLength();
		double lenth2 = (ps->GetParticle(index1).x - ps->GetParticle(index2).x).GetLength();
		double lenth3 = (ps->GetParticle(index2).x - ps->GetParticle(index0).x).GetLength();
		ps->AddSpringConstraint(index0, index1,lenth1);
		ps->AddSpringConstraint(index1, index2,lenth2);
		ps->AddSpringConstraint(index2, index0,lenth3);
		//qDebug() <<i<< " " << triangle.vindices[1] << " " <<triangle.tindices[1]<<" "<< model.meshes[0].vertices[triangle.vindices[1]].Position.x<<" "<<glmmodel->texcoords[2 * triangle.tindices[1]];
	}
	////添加mesh内部的静顶点作为支架
	//for (int i = 0; i <= glmmodel->numvertices; i++)
	//{
	//	Vertex v;
	//	glm::mat4 model = glm::mat4(1.0f);
	//	v.Position = glm::scale(model,glm::vec3(0.8,0.8,0.8))*glm::vec4(glmmodel->vertices[3 * i], glmmodel->vertices[3 * i + 1], glmmodel->vertices[3 * i + 2],1.0);
	//	glm::vec3 v1 = v.Position;
	//	ps->AddParticle(v1.x, v1.y, v1.z);
	//	ps->AddNailConstraint(glmmodel->numvertices +i);
	//	ps->AddSpringConstraint(i, glmmodel->numvertices + i);//将静止点和对应动质点间加弹簧约束
	//}
	//for (int i = 0; i < glmmodel->numtriangles; i++)//四面体化
	//{
	//	GLMtriangle& triangle = glmmodel->triangles[i];
	//	Vertex& v0 = model.meshes[0].vertices[triangle.vindices[0]];
	//	Vertex& v1 = model.meshes[0].vertices[triangle.vindices[1]];
	//	Vertex& v2 = model.meshes[0].vertices[triangle.vindices[2]];
	//	glm::vec3 center = (v0.Position + v1.Position + v2.Position) / glm::vec3(3,3,3);
	//	glm::vec3 point = center-glm::vec3(v0.Normal+v1.Normal+v2.Normal)/glm::vec3(3,3,3);
	//	ps->AddParticle(point.x, point.y, point.z);
	//	ps->AddNailConstraint(glmmodel->numvertices + i);
	//	ps->AddSpringConstraint(triangle.vindices[0], glmmodel->numvertices + i);
	//	ps->AddSpringConstraint(triangle.vindices[1], glmmodel->numvertices + i);
	//	ps->AddSpringConstraint(triangle.vindices[2], glmmodel->numvertices + i);
	//	//qDebug() << i;
	//}
	model.meshes[0].ChangeVertex();
	Particle::SetMass(0.1);
	ps->FinishConstructingSystem();
	ps->gravity = 0.0;
	ps->drag = 1.0;
	ps->SetSpringConstant(1);
	ps->SetSpringDampingConstant(1);
	timer.Reset();
}
void Obj2MassSpring::updatePS()
{
	static double prevTime = 0;
	double currTime = timer.GetTime()/1000;
	ps->AdvanceSimulation(prevTime, currTime);
	prevTime = currTime;
	//更新位置
	int p = 0;
	for (int j = 0; j < model.meshes[0].vertices.size(); j++)
	{
		Vertex& v = model.meshes[0].vertices[j];
		Particle& tem=ps->GetParticle(p);
		v.Position.x = tem.x[0];
		v.Position.y = tem.x[1];
		v.Position.z = tem.x[2];
		p++;
	}
	model.meshes[0].ChangeVertex();
	//ps->Draw();
}
void Obj2MassSpring::MouseDrag(int point,VECTOR3D position)
{
	Particle& tem = ps->GetParticle(point);
	tem.x = VECTOR3D(0,0,0);

	qDebug() <<point<<" "<< ps->GetParticle(point).x.x << " " << ps->GetParticle(point).x.y << " " << ps->GetParticle(point).x.z;
}
void Obj2MassSpring::RayPick(QPoint pos, QMatrix4x4 modma, QMatrix4x4 proma, QMatrix4x4 viewma, QVector3D campos,int width,int height,bool drag)//点击后获取点击的顶点并且给该点施加力
{
	ray->transData(proma, viewma, campos);
	vec_triangles.clear();
	for (int i = 0; i < glmmodel->numtriangles; i++)
	{
		GLMtriangle& triangle = glmmodel->triangles[i];
		QVector<QVector3D> tri;
		for (int j = 0; j < 3; j++)
		{
			glm::vec3 p=model.meshes[0].vertices[triangle.vindices[j]].Position;
			QVector3D tem(p.x,p.y,p.z);
			tri.push_back(modma * tem);
		}
		vec_triangles.push_back(tri);
	}
	//GLint viewport[4];
	//glGetIntegerv(GL_VIEWPORT, viewport);
	float x = 2.0f * pos.x() / width - 1.0f; //视口坐标转 NDC坐标
	float y = 1.0f - (2.0f * pos.y() / height);
	QVector3D dir;
	int index = ray->picking(x, y, vec_triangles,dir);
	dir = modma.inverted() * dir;
	VECTOR3D vdir;
	if(!drag)
		vdir=VECTOR3D(dir.x(), dir.y(), dir.z());
	else
		vdir = VECTOR3D(-dir.x(), -dir.y(), -dir.z());
	ps->StartConstructingSystem();
	ps->AddMouseForceConstraint(glmmodel->triangles[index].vindices[0],0.6*vdir);
	ps->AddMouseForceConstraint(glmmodel->triangles[index].vindices[1], 0.6 * vdir);
	ps->AddMouseForceConstraint(glmmodel->triangles[index].vindices[2], 0.6 * vdir);
	ps->FinishConstructingSystem();
}
int Obj2MassSpring::MouseClicked(VECTOR3D position)
{
	int closepoint=ps->GetClosestParticle(position);
	ps->StartConstructingSystem();
	ps->AddMouseSpringConstraint(7000);
	ps->FinishConstructingSystem();
	return closepoint;
}