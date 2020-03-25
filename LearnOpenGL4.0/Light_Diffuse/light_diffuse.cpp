#include <GL/glew.h>
#include <GL/freeglut.h>
#include <iostream>
#include <assert.h>

#include "GLSLShader.h"
#include "vbotorus.h"


#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform2.hpp>

#define GL_CHECK_ERRORS assert(glGetError()== GL_NO_ERROR)


//shader  
GLSLShader shader;

//顶点数组对象和顶点缓冲区对象
GLuint vaoID;
GLuint vboVerticesID;
GLuint vboIndicesID;

VBOTorus * torus;
glm::mat4 model;
glm::mat4 view;
glm::mat4 projection;



void Init()
{
	glClearColor(0.5f,0.5f,0.5f,1.0f);



	shader.LoadFromFile(GL_VERTEX_SHADER, "shader/diffuse.vert");
	shader.LoadFromFile(GL_FRAGMENT_SHADER, "shader/diffuse.frag");
	shader.CreateAndLinkProgram();
	shader.Use();
	GL_CHECK_ERRORS;


	glEnable(GL_DEPTH_TEST);
	torus = new VBOTorus(0.7f, 0.3f, 30, 30);

	model = glm::mat4(1.0f);
	model *= glm::rotate(-35.0f, glm::vec3(1.0f,0.0f,0.0f));
	model *= glm::rotate(35.0f, glm::vec3(0.0f,1.0f,0.0f));
	view = glm::lookAt(glm::vec3(0.0f,0.0f,2.0f), glm::vec3(0.0f,0.0f,0.0f), glm::vec3(0.0f,1.0f,0.0f));
	projection = glm::mat4(1.0f);


 

	shader.AddUniform("Kd");
	shader.AddUniform("Ld");
	shader.AddUniform("LightPosition");

 


	glUniform3f(shader("Kd"),0.9f, 0.5f, 0.3f);
	glUniform3f(shader("Ld"),1.0f, 1.0f, 1.0f);
	glm::vec4 v= view * glm::vec4(5.0f,5.0f,2.0f,1.0f) ;
	glUniform4f(shader("LightPosition"),v.x,v.y,v.z,v.w);



	shader.AddUniform("ModelViewMatrix");
	shader.AddUniform("NormalMatrix");
	shader.AddUniform("MVP");



	//查询uniform变量
	GLint nUniforms, maxLen;
	glGetProgramiv( shader.GetHandler(), GL_ACTIVE_UNIFORM_MAX_LENGTH,&maxLen);
	glGetProgramiv( shader.GetHandler(), GL_ACTIVE_UNIFORMS,&nUniforms);

	GLchar * name = (GLchar *) malloc( maxLen );

	GLint size, location;
	GLsizei written;
	GLenum type;
	printf(" Location | Name\n");
	printf("------------------------------------------------\n");
	for( int i = 0; i < nUniforms; ++i ) {
		glGetActiveUniform( shader.GetHandler(), i, maxLen, &written,
			&size, &type, name );
		location = glGetUniformLocation(shader.GetHandler(), name);
		printf(" %-8d | %s\n", location, name);
	}
	printf("------------------------------------------------\n");
	free(name);
}
//----------------------------------------------------------------------------

void display( void )
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 

	shader.Use();

	glm::mat4 m4 = view * model;

	glUniformMatrix4fv(shader("ModelViewMatrix"), 1, GL_FALSE, &m4[0][0]);
	glm::mat3 m3 = glm::mat3( glm::vec3(m4[0]), glm::vec3(m4[1]), glm::vec3(m4[2]) );
	glUniformMatrix3fv(shader("NormalMatrix"), 1, GL_FALSE, &m3[0][0]);
	glm::mat4 mvp = projection * m4;
	glUniformMatrix4fv(shader("MVP"), 1, GL_FALSE, &mvp[0][0]);


	torus->render();

	shader.UnUse();
	glutSwapBuffers();
}
//----------------------------------------------------------------------------

void reshape( int w, int h )
{
	glViewport(0,0,w,h);
	projection = glm::perspective(70.0f, (float)w/h, 0.3f, 100.0f);
}

//----------------------------------------------------------------------------

void mouse( int button, int state, int x, int y )
{
}

//----------------------------------------------------------------------------
void idle( void )
{

	glutPostRedisplay();
}

//----------------------------------------------------------------------------

void keyboard( unsigned char key, int x, int y )
{ 
}


//----------------------------------------------------------------------------
void OnShutdown()
{
	//销毁 shader
	shader.DeleteShaderProgram();

	//销毁vbo 和 vao
	glDeleteBuffers(1, &vboVerticesID);
	glDeleteBuffers(1, &vboIndicesID);
	glDeleteVertexArrays(1, &vaoID);

	std::cout<<"Shutdown successfull"<<std::endl;
}
int main( int argc, char **argv )
{
	glutInit( &argc, argv );
	glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH );
	glutInitWindowSize( 512, 512 );

	glutCreateWindow( "Basic FrameWork!!" );

	if ( glewInit() != GLEW_OK )
	{
		std::cerr << "Failed to initialize GLEW ... exiting" << std::endl;
		exit( EXIT_FAILURE );
	}

	Init();


	const GLubyte *renderer = glGetString( GL_RENDERER );
	const GLubyte *vendor = glGetString( GL_VENDOR );
	const GLubyte *version = glGetString( GL_VERSION );
	const GLubyte *glslVersion =glGetString( GL_SHADING_LANGUAGE_VERSION );
	GLint major, minor;
	glGetIntegerv(GL_MAJOR_VERSION, &major);
	glGetIntegerv(GL_MINOR_VERSION, &minor);
	printf("GL Vendor : %s\n", vendor);
	printf("GL Renderer : %s\n", renderer);
	printf("GL Version (string) : %s\n", version);//字符串形式
	printf("GL Version (integer) : %d.%d\n", major, minor);//整数形式
	printf("GLSL Version : %s\n", glslVersion);


	// 	GLint nExtensions;
	// 	glGetIntegerv(GL_NUM_EXTENSIONS, &nExtensions);
	// 	for( int i = 0; i < nExtensions; i++ )//输出所有支持扩展
	// 		printf("%s\n", glGetStringi( GL_EXTENSIONS, i ) );


	glutInitContextVersion(major, minor);//指定版本号

	glutCloseFunc(OnShutdown);
	glutDisplayFunc( display );
	glutKeyboardFunc( keyboard );
	glutReshapeFunc( reshape );
	glutMouseFunc( mouse );
	glutIdleFunc( idle );

	glutMainLoop();
	return 0;
}
