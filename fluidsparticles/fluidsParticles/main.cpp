#include <stdio.h>
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <helper_math.h>
#include <GL/freeglut.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <cuda_profiler_api.h>
#include "global.h"
#include "SPH.h"
#include "FLIP.h"
#include "MC.h"
#include "Utility.h"
#include "Voronoi3D.h"

float rot[2];
int mousePos[2]={-1,-1};
bool mouse_left_down = false;
float zoom = 0.35f;

static int frame_id = 0;
bool running = false;

SPH *sph;
FLIP *flip;
MC *mc;
//Voronoi3D *voro;
int particleNum = 50*32*32;//17256;
float3 spaceSize = make_float3(1.0, 1.0, 1.0);
// vbo variables
GLuint particlesVBO;
GLuint particlesColorVBO;
GLuint flipVBO;
GLuint flipColorVBO;
GLuint m_particles_program;
GLuint m_flip_program;
GLuint voroVertexVBO;
GLuint voroColorVBO;
int m_window_h=700;
int m_fov = 30;
float particle_radius = 0.01f;
float flip_radius = 0.0025f;
bool isRenderSPH=1;

struct log frame_log;

void createVBO(GLuint* vbo, unsigned int length)
{
	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	// initialize buffer object
	glBufferData(GL_ARRAY_BUFFER, length, 0, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// register buffer object with CUDA
	cudaGLRegisterBufferObject(*vbo);
}

void deleteVBO(GLuint* vbo)
{
	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);

	cudaGLUnregisterBufferObject(*vbo);

	*vbo = NULL;
}

namespace particle_attributes {
	enum {
		POSITION,
		COLOR,
		SIZE,
	};
}

void onClose()
{
	cudaDeviceReset();
	cudaProfilerStop();
	exit(0);
}

void initParticlesProgram()
{
	m_particles_program = glCreateProgram();
	glBindAttribLocation(m_particles_program, particle_attributes::SIZE, "pointSize");
	Utility::attachAndLinkProgram(m_particles_program, Utility::loadShaders("particles.vert", "particles.frag"));
	m_flip_program = glCreateProgram();
	glBindAttribLocation(m_flip_program, particle_attributes::SIZE, "pointSize");
	Utility::attachAndLinkProgram(m_flip_program, Utility::loadShaders("particles.vert", "particles.frag"));
}

void init()
{
	sph = new SPH(particleNum, spaceSize);
	flip = new FLIP(sph);
	mc = new MC(sph);
	//voro = new Voronoi3D(sph);

	createVBO(&particlesVBO, sizeof(float3)*(sph->num_fluid + sph->num_staticBoundary));
	createVBO(&particlesColorVBO, sizeof(float3)*(sph->num_fluid + sph->num_staticBoundary));
	createVBO(&flipVBO, sizeof(float3)*(flip->num_flip));
	createVBO(&flipColorVBO, sizeof(float3)*(flip->num_flip));
// 	createVBO(&voroVertexVBO, sizeof(float3)*2*voro->para_h->voro_max_edge);
// 	createVBO(&voroColorVBO, sizeof(float3)*2*voro->para_h->voro_max_edge);
	initParticlesProgram();

	FILE*fp=fopen("density.txt", "w");
	fclose(fp);
	fp = fopen("energy.txt", "w");
	fclose(fp);
	return;
}

void mouse(int button,int state,int x,int y)
{
	if(GLUT_DOWN==state)
	{
		if(GLUT_LEFT_BUTTON==button)
		{
			mouse_left_down = true;
			mousePos[0] = x;
			mousePos[1] = y;
		}
		else if(GLUT_RIGHT_BUTTON==button)
		{
		}
	}
	else
	{
		mouse_left_down = false;
	}
	return;
}

void motion(int x, int y)
{
	int dx,dy;
	if(-1==mousePos[0] && -1==mousePos[1])
	{
		mousePos[0] = x;
		mousePos[1] = y;
		dx=dy=0;
	}
	else
	{
		dx = x-mousePos[0];
		dy = y-mousePos[1];
	}
	if(mouse_left_down)
	{
		rot[0] += (dy * 180.0f) / 720.0f;
		rot[1] += (dx * 180.0f) / 720.0f;
	}

	mousePos[0] = x;
	mousePos[1] = y;

	glutPostRedisplay();
	return;
}

void keyboard(unsigned char key,int x,int y)
{
	switch (key)
	{
	case ' ':
		running = !running;
		break;
	case ',':
		zoom*=1.2f;
		break;
	case '.':
		zoom/=1.2f;
		break;
	case 'q':
	case 'Q':
		onClose();
		break;
	case 'r':
	case 'R':
		rot[0] = rot[1] = 0;
		zoom = 0.35f;
		break;
	case 'n':
	case 'N':
		frame_id++;
		sprintf(frame_log.str[frame_log.ptr++], "Frame %04d", frame_id);
		sph->step();
//		voro->DT();
		frame_log.output();
		break;
	case 'm':
	case 'M':
		mc->apply_MC("123.stl");
		frame_log.output();
		break;
	case 'e':
	case 'E':
		break;
	case 'p':
	case 'P':
		isRenderSPH = !isRenderSPH;
		break;
	default:
		;
	}
}

extern "C" 	void generate_dots(float3* pos, float3* posColor, SPH *sph);
extern "C" 	void generate_dots_flip(float3* dot, float3* posColor, FLIP *flip);
//extern "C" void generate_voro(float3 *pos, float3 *posColor, Voronoi3D *voro);

// void renderVORO()
// {
// 	// map OpenGL buffer object for writing from CUDA
// 	float3 *dptr;
// 	float3 *cptr;
// 	cudaGLMapBufferObject((void**)&dptr, voroVertexVBO);
// 	cudaGLMapBufferObject((void**)&cptr, voroColorVBO);
// 
// 	// calculate the lines' position
// 	generate_voro(dptr, cptr, voro);
// 
// 	// unmap buffer object
// 	cudaGLUnmapBufferObject(voroVertexVBO);
// 	cudaGLUnmapBufferObject(voroColorVBO);
// 
// 
// 	glBindBuffer(GL_ARRAY_BUFFER, voroVertexVBO);
// 	glVertexPointer(3, GL_FLOAT, 0, 0);
// 	glEnableClientState(GL_VERTEX_ARRAY);
// 
// 	glBindBuffer(GL_ARRAY_BUFFER, voroColorVBO);
// 	glColorPointer(3, GL_FLOAT, 0, 0);
// 	glEnableClientState(GL_COLOR_ARRAY);
// 
// 	glDrawArrays(GL_LINES, 0, voro->num_edge_h*2);
// 
// 	glDisableClientState(GL_VERTEX_ARRAY);
// 	glDisableClientState(GL_COLOR_ARRAY);
// 	return;
// }

void renderSPH()
{
	// map OpenGL buffer object for writing from CUDA
	float3 *dptr;
	float3 *cptr;
	cudaGLMapBufferObject((void**)&dptr, particlesVBO);
	cudaGLMapBufferObject((void**)&cptr, particlesColorVBO);

	// calculate the lines' position
	generate_dots(dptr, cptr, sph);

	// unmap buffer object
	cudaGLUnmapBufferObject(particlesVBO);
	cudaGLUnmapBufferObject(particlesColorVBO);


	glBindBuffer(GL_ARRAY_BUFFER, particlesVBO);
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);

	glBindBuffer(GL_ARRAY_BUFFER, particlesColorVBO);
	glColorPointer(3, GL_FLOAT, 0, 0);
	glEnableClientState(GL_COLOR_ARRAY);

	glDrawArrays(GL_POINTS, 0, sph->num_fluid);

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
	return;
}

void renderFLIP()
{
	float3 *dptr;
	float3 *cptr;
	cudaGLMapBufferObject((void**)&dptr, flipVBO);
	cudaGLMapBufferObject((void**)&cptr, flipColorVBO);
	generate_dots_flip(dptr, cptr, flip);
	cudaGLUnmapBufferObject(flipVBO);
	cudaGLUnmapBufferObject(flipColorVBO);
	glBindBuffer(GL_ARRAY_BUFFER, flipVBO);
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, flipColorVBO);
	glColorPointer(3, GL_FLOAT, 0, 0);
	glEnableClientState(GL_COLOR_ARRAY);
	glDrawArrays(GL_POINTS, 0, flip->num_flip);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
	return;
}

void myDisplay(void)
{
	if(running)
	{
		frame_id++;
		sprintf(frame_log.str[frame_log.ptr++], "Frame %04d", frame_id);
		sph->step();
		//voro->DT();
		//system("cls");

// 		if (frame_id>1600) exit(0);
// 		char fn[32];
// 		if (frame_id%2==0) {
// 			sprintf(fn, "stl\\%04d.stl", frame_id/2);
// 			mc->apply_MC(fn);
// 		}
		frame_log.output();

		if (sph->outputSTL && frame_id%8==0) {
			static float3 *vec = (float3*)malloc(sizeof(float3)*sph->num_fluid);//от╢Ф
			static float *sca = (float*)malloc(sizeof(float)*sph->num_fluid);
			char filename[32];
			//sprintf(filename, "stl\\%04d.bin", frame_id/8);
			//FILE *fp = fopen(filename, "wb+");
			//fprintf(fp, "%d\n", sph->num_fluid);
			cudaMemcpy(vec, sph->pos_fluid, sizeof(float3)*sph->num_fluid, cudaMemcpyDeviceToHost);
			//fwrite(vec, sizeof(float3), sph->num_fluid, fp);
			cudaMemcpy(vec, sph->vel_fluid, sizeof(float3)*sph->num_fluid, cudaMemcpyDeviceToHost);
			//fwrite(vec, sizeof(float3), sph->num_fluid, fp);
			float vv = sph->para_h->sph_spacing;
			vv = vv*vv*vv;
			for (int i=0; i<sph->num_fluid; i++)
				sca[i] = vv;
			//fwrite(sca, sizeof(float), sph->num_fluid, fp);
			//fwrite(sca, sizeof(float), sph->num_fluid, fp);
			//fwrite(sca, sizeof(float), sph->num_fluid, fp);
			//fclose(fp);
		}
	}
// 	if(frame_id>200)
// 	{
// 		onClose();
// 	}
	////////////////////
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

	glViewport(0,0,m_window_h,m_window_h);
	glUseProgram(0);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(m_fov,m_window_h/m_window_h,0.01,100.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0,0,1.0/zoom,0,0,0,0,1,0);

	glPushMatrix();
	glRotatef(rot[0], 1.0f, 0.0f, 0.0f);
	glRotatef(rot[1], 0.0f, 1.0f, 0.0f);
	glLineWidth(3.0);
	////////////////////
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glutSolidCube(1.0);
	////////////////////
	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);
	glLineWidth(1.0);
	glPushMatrix();
	glTranslatef(-.5, -.5, -.5);
	//renderVORO();
	glPopMatrix();


	glUseProgram(m_particles_program);
	glUniform1f(glGetUniformLocation(m_particles_program, "pointScale"), m_window_h / tanf(m_fov*0.5f*(float)M_PI/180.0f));
	glUniform1f(glGetUniformLocation(m_particles_program, "pointRadius"), particle_radius);
	glEnable(GL_POINT_SPRITE_ARB);
	glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);
	glPushMatrix();
	glTranslatef(-.5, -.5, -.5);
	if(isRenderSPH)
		renderSPH();
	glPopMatrix();
	////////////////////
// 	glUseProgram(m_flip_program);
// 	glUniform1f(glGetUniformLocation(m_flip_program, "pointScale"), m_window_h / tanf(m_fov*0.5f*(float)M_PI/180.0f));
// 	glUniform1f(glGetUniformLocation(m_flip_program, "pointRadius"), flip_radius);
// 	glEnable(GL_POINT_SPRITE_ARB);
// 	glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
// 	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
// 	glDepthMask(GL_TRUE);
// 	glEnable(GL_DEPTH_TEST);
// 	glPushMatrix();
// 	glTranslatef(-.5, -.5, -.5);
// 	renderFLIP();
// 	glPopMatrix();

	////////////////////
	glPopMatrix();
	glutSwapBuffers();
	glutPostRedisplay();
}

int main(int argc, char* argv[])
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutInitWindowPosition(400, 0);
	glutInitWindowSize(m_window_h, m_window_h);
	glutCreateWindow("");
	glutDisplayFunc(&myDisplay);
	glutKeyboardFunc(&keyboard); 
	glutMouseFunc(&mouse);
	glutMotionFunc(&motion);

	glewInit();
	////////////////////
	init();
	////////////////////
	glutMainLoop();
	return 0;
}