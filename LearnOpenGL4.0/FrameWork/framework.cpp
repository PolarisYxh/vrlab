#include <GL/glew.h>
#include <GL/freeglut.h>
#include <iostream>
#include <assert.h>
#include <cstdio>
#include <string>
#include "GLSLShader.h"


#define GL_CHECK_ERRORS assert(glGetError()== GL_NO_ERROR)


//shader  
GLSLShader shader;

//顶点数组对象和顶点缓冲区对象
GLuint vaoID;
GLuint vboVerticesID;
GLuint vboIndicesID;


void Init()
{


	glClearColor(0.5f,0.5f,0.5f,1.0f);


	glEnable(GL_DEBUG_OUTPUT);

 
	GL_CHECK_ERRORS;
		//load the shader
		shader.LoadFromFile(GL_VERTEX_SHADER, "shaders/basic.vert");
	shader.LoadFromFile(GL_FRAGMENT_SHADER, "shaders/basic.frag");
	//compile and link shader
	shader.CreateAndLinkProgram();


	GL_CHECK_ERRORS;




		/////////////////// Create the VBO ////////////////////
		float positionData[] = {
			-0.5f, -0.5f, 0.0f,
			0.5f, -0.5f, 0.0f,
			0.0f,  0.5f, 0.0f };

			float colorData[] = {
				1.0f, 0.0f, 0.0f,
				0.0f, 1.0f, 0.0f,
				0.0f, 0.0f, 1.0f };



				// Create and set-up the vertex array object
				glGenVertexArrays( 1, &vaoID );
				glBindVertexArray(vaoID);


				// Create and populate the buffer objects
				GLuint vboHandles[2];
				glGenBuffers(2, vboHandles);
				GLuint positionBufferHandle = vboHandles[0];
				GLuint colorBufferHandle = vboHandles[1];

				glBindBuffer(GL_ARRAY_BUFFER, positionBufferHandle);
				glBufferData(GL_ARRAY_BUFFER, 9 * sizeof(float), positionData, GL_STATIC_DRAW);

				glBindBuffer(GL_ARRAY_BUFFER, colorBufferHandle);
				glBufferData(GL_ARRAY_BUFFER, 9 * sizeof(float), colorData, GL_STATIC_DRAW);


				glEnableVertexAttribArray(0);  // Vertex position
				glEnableVertexAttribArray(1);  // Vertex color

				glBindBuffer(GL_ARRAY_BUFFER, positionBufferHandle);
				glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, 0, (GLubyte *)NULL );

				glBindBuffer(GL_ARRAY_BUFFER, colorBufferHandle);
				glVertexAttribPointer( 1, 3, GL_FLOAT, GL_FALSE, 0, (GLubyte *)NULL );







				//获得当前活动顶点属性和其索引
				GLint maxLength, nAttribs;
				glGetProgramiv(shader.GetHandler(), GL_ACTIVE_ATTRIBUTES,&nAttribs);
				glGetProgramiv(shader.GetHandler(), GL_ACTIVE_ATTRIBUTE_MAX_LENGTH,&maxLength);

				GLchar * name = (GLchar *) malloc( maxLength );
				GLint written, size, location;
				GLenum type;
				printf(" Index | Name\n");
				printf("------------------------------------------------\n");
				for( int i = 0; i < nAttribs; i++ )
				{
					glGetActiveAttrib( shader.GetHandler(), i, maxLength, &written,	&size, &type, name );
					location = glGetAttribLocation(shader.GetHandler(), name);
					printf(" %-5d | %s\n",location, name);
				}
				free(name);
				printf("------------------------------------------------\n");



}
//----------------------------------------------------------------------------

void display( void )
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 

	//bind the shader
	shader.Use();
	//glBindVertexArray(vaoID);
	glDrawArrays(GL_TRIANGLES, 0, 3 );
	//unbind the shader
	shader.UnUse();

	glutSwapBuffers();
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

void reshape( int width, int height )
{
	glViewport(0,0,width,height);
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
