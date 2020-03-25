#include "CoreFunctionWidget.h"
#include <QDebug>
#include <QFile>
static GLuint VBO, VAO, EBO;

CoreFunctionWidget::CoreFunctionWidget(QWidget* parent) : QOpenGLWidget(parent)
{

}

CoreFunctionWidget::~CoreFunctionWidget()
{
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
	//    glDeleteBuffers(1, &EBO);
}

void CoreFunctionWidget::initializeGL() {
	this->initializeOpenGLFunctions();

	bool success = shaderProgram.addShaderFromSourceFile(QOpenGLShader::Vertex, "./triangle.vert");
	if (!success) {
		qDebug() << "shaderProgram addShaderFromSourceFile failed!" << shaderProgram.log();
		return;
	}

	success = shaderProgram.addShaderFromSourceFile(QOpenGLShader::Fragment, "./triangle.frag");
	if (!success) {
		qDebug() << "shaderProgram addShaderFromSourceFile failed!" << shaderProgram.log();
		return;
	}

	success = shaderProgram.link();
	if (!success) {
		qDebug() << "shaderProgram link failed!" << shaderProgram.log();
	}

	//VAO，VBO数据部分
	float vertices[] = {
		0.5f,  0.5f, 0.0f,  // top right
		 0.5f, -0.5f, 0.0f,  // bottom right
		-0.5f, -0.5f, 0.0f,  // bottom left
		-0.5f,  0.5f, 0.0f   // top left
	};
	unsigned int indices[] = {  // note that we start from 0!
		0, 1, 3,  // first Triangle
		1, 2, 3   // second Triangle
	};

	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);
	// bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);  //顶点数据复制到缓冲

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);//告诉程序如何解析顶点数据
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);//取消VBO的绑定, glVertexAttribPointer已经把顶点属性关联到顶点缓冲对象了

//    remember: do NOT unbind the EBO while a VAO is active as the bound element buffer object IS stored in the VAO; keep the EBO bound.
//    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

//    You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
//    VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
	glBindVertexArray(0);   //取消VAO绑定

	//线框模式，QOpenGLExtraFunctions没这函数, 3_3_Core有
//    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
}

void CoreFunctionWidget::resizeGL(int w, int h) {
	glViewport(0, 0, w, h);
}

void CoreFunctionWidget::paintGL() {
	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	shaderProgram.bind();
	glBindVertexArray(VAO); // seeing as we only have a single VAO there's no need to bind it every time, but we'll do so to keep things a bit more organized
//    glDrawArrays(GL_TRIANGLES, 0, 6);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

	shaderProgram.release();
}
/*#include "CoreFunctionWidget.h"
#include <QDebug>
#include <QTimer>
#include <QKeyEvent>
#include <QDateTime>

// lighting
static QVector3D lightPos(1.2f, 1.0f, 2.0f);

CoreFunctionWidget::CoreFunctionWidget(QWidget* parent) : QOpenGLWidget(parent)
{
	camera = std::make_unique<Camera>(QVector3D(0.0f, 0.0f, 3.0f));
	m_bLeftPressed = false;

	m_pTimer = new QTimer(this);
	connect(m_pTimer, &QTimer::timeout, this, [=] {
		m_nTimeValue += 1;
		update();
	});
	m_pTimer->start(40);//25 fps

}

CoreFunctionWidget::~CoreFunctionWidget()
{
	glDeleteVertexArrays(1, &lightVAO);
	glDeleteVertexArrays(1, &cubeVAO);
	glDeleteBuffers(1, &VBO);
}

void CoreFunctionWidget::initializeGL() {
	this->initializeOpenGLFunctions();

	createShader();

	// set up vertex data (and buffer(s)) and configure vertex attributes
	// ------------------------------------------------------------------
	float vertices[] = {
		// positions          // normals           // texture coords
		-0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f,  0.0f,
		 0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f,  0.0f,
		 0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f,  1.0f,
		 0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f,  1.0f,
		-0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f,  1.0f,
		-0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f,  0.0f,

		-0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f,  0.0f,
		 0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f,  0.0f,
		 0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f,  1.0f,
		 0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f,  1.0f,
		-0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f,  1.0f,
		-0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f,  0.0f,

		-0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f,  0.0f,
		-0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  1.0f,  1.0f,
		-0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f,  1.0f,
		-0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f,  1.0f,
		-0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  0.0f,  0.0f,
		-0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f,  0.0f,

		 0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f,
		 0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  1.0f,  1.0f,
		 0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  0.0f,  1.0f,
		 0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  0.0f,  1.0f,
		 0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  0.0f,  0.0f,
		 0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f,

		-0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  0.0f,  1.0f,
		 0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  1.0f,  1.0f,
		 0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  1.0f,  0.0f,
		 0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  1.0f,  0.0f,
		-0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  0.0f,  0.0f,
		-0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  0.0f,  1.0f,

		-0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  0.0f,  1.0f,
		 0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  1.0f,  1.0f,
		 0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  1.0f,  0.0f,
		 0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  1.0f,  0.0f,
		-0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  0.0f,  0.0f,
		-0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  0.0f,  1.0f
	};

	// first, configure the cube's VAO (and VBO)
	glGenVertexArrays(1, &cubeVAO);
	glGenBuffers(1, &VBO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glBindVertexArray(cubeVAO);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
	glEnableVertexAttribArray(2);

	// second, configure the light's VAO (VBO stays the same; the vertices are the same for the light object which is also a 3D cube)
	glGenVertexArrays(1, &lightVAO);
	glBindVertexArray(lightVAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	// note that we update the lamp's position attribute's stride to reflect the updated buffer data
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	// load textures (we now use a utility function to keep the code more organized)
	// -----------------------------------------------------------------------------
	diffuseMap = loadTexture(":/container2.png");
	specularMap = loadTexture(":/container2_specular.png");

	// shader configuration
	// --------------------
	lightingShader.bind();
	lightingShader.setUniformValue("material.diffuse", 0);
	lightingShader.setUniformValue("material.specular", 1);
	lightingShader.release();

	// configure global opengl state
	// -----------------------------
	glEnable(GL_DEPTH_TEST);
}

void CoreFunctionWidget::resizeGL(int w, int h) {
	glViewport(0, 0, w, h);
}

void CoreFunctionWidget::paintGL() {
	// input
	// -----
	camera->processInput(0.5f);//speed

	// render
	// ------
	glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// be sure to activate shader when setting uniforms/drawing objects
	lightingShader.bind();
	lightingShader.setUniformValue("light.position", lightPos);
	lightingShader.setUniformValue("viewPos", camera->position);

	// light properties
	lightingShader.setUniformValue("light.ambient", QVector3D(0.2f, 0.2f, 0.2f));
	lightingShader.setUniformValue("light.diffuse", QVector3D(0.5f, 0.5f, 0.5f));
	lightingShader.setUniformValue("light.specular", QVector3D(1.0f, 1.0f, 1.0f));

	// material properties
	lightingShader.setUniformValue("material.shininess", 64.0f);

	// view/projection transformations
	QMatrix4x4 projection;
	projection.perspective(camera->zoom, 1.0f * width() / height(), 0.1f, 100.0f);
	QMatrix4x4 view = camera->getViewMatrix();
	lightingShader.setUniformValue("projection", projection);
	lightingShader.setUniformValue("view", view);

	// world transformation
	QMatrix4x4 model;
	lightingShader.setUniformValue("model", model);

	// bind diffuse map
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, diffuseMap);
	// bind specular map
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, specularMap);

	// render the cube
	glBindVertexArray(cubeVAO);
	glDrawArrays(GL_TRIANGLES, 0, 36);
	lightingShader.release();

	// also draw the lamp object
	lampShader.bind();
	lampShader.setUniformValue("projection", projection);
	lampShader.setUniformValue("view", view);
	model = QMatrix4x4();
	model.translate(lightPos);
	model.scale(0.2f); // a smaller cube
	lampShader.setUniformValue("model", model);

	glBindVertexArray(lightVAO);
	glDrawArrays(GL_TRIANGLES, 0, 36);
	lampShader.release();
}

void CoreFunctionWidget::keyPressEvent(QKeyEvent* event)
{
	int key = event->key();
	if (key >= 0 && key < 1024)
		camera->keys[key] = true;
}

void CoreFunctionWidget::keyReleaseEvent(QKeyEvent* event)
{
	int key = event->key();
	if (key >= 0 && key < 1024)
		camera->keys[key] = false;
}

void CoreFunctionWidget::mousePressEvent(QMouseEvent* event)
{
	if (event->button() == Qt::LeftButton) {
		m_bLeftPressed = true;
		m_lastPos = event->pos();
	}
}

void CoreFunctionWidget::mouseReleaseEvent(QMouseEvent* event)
{
	Q_UNUSED(event);

	m_bLeftPressed = false;
}

void CoreFunctionWidget::mouseMoveEvent(QMouseEvent* event)
{
	int xpos = event->pos().x();
	int ypos = event->pos().y();

	int xoffset = xpos - m_lastPos.x();
	int yoffset = m_lastPos.y() - ypos;
	m_lastPos = event->pos();

	camera->processMouseMovement(xoffset, yoffset);
}

void CoreFunctionWidget::wheelEvent(QWheelEvent* event)
{
	QPoint offset = event->angleDelta();
	camera->processMouseScroll(offset.y() / 20.0f);
}

bool CoreFunctionWidget::createShader()
{
	bool success = lightingShader.addShaderFromSourceFile(QOpenGLShader::Vertex, ":/lighting_maps.vert");
	if (!success) {
		qDebug() << "shaderProgram addShaderFromSourceFile failed!" << lightingShader.log();
		return success;
	}

	success = lightingShader.addShaderFromSourceFile(QOpenGLShader::Fragment, ":/lighting_maps.frag");
	if (!success) {
		qDebug() << "shaderProgram addShaderFromSourceFile failed!" << lightingShader.log();
		return success;
	}

	success = lightingShader.link();
	if (!success) {
		qDebug() << "shaderProgram link failed!" << lightingShader.log();
	}

	success = lampShader.addShaderFromSourceFile(QOpenGLShader::Vertex, ":/lamp.vert");
	if (!success) {
		qDebug() << "shaderProgram addShaderFromSourceFile failed!" << lampShader.log();
		return success;
	}

	success = lampShader.addShaderFromSourceFile(QOpenGLShader::Fragment, ":/lamp.frag");
	if (!success) {
		qDebug() << "shaderProgram addShaderFromSourceFile failed!" << lampShader.log();
		return success;
	}

	success = lampShader.link();
	if (!success) {
		qDebug() << "shaderProgram link failed!" << lampShader.log();
	}

	return success;
}

uint CoreFunctionWidget::loadTexture(const QString& path)
{
	uint textureID;
	glGenTextures(1, &textureID);

	QImage image = QImage(path).convertToFormat(QImage::Format_RGBA8888).mirrored(true, true);
	if (!image.isNull()) {
		glBindTexture(GL_TEXTURE_2D, textureID);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width(), image.height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, image.bits());
		glGenerateMipmap(GL_TEXTURE_2D);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}

	return textureID;
}*/

/*#include "CoreFunctionWidget.h"
#include <QDebug>
#include <QFile>
#include <QtOpenGL>
#include <QSlider>
static GLuint VBO,VAO, EBO;

CoreFunctionWidget::CoreFunctionWidget(QWidget* parent) : QOpenGLWidget(parent)
{

}

CoreFunctionWidget::~CoreFunctionWidget()
{
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
	//    glDeleteBuffers(1, &EBO);
}
void CoreFunctionWidget::printContextInformation()
{
	QString glType;
	QString glVersion;
	QString glProfile;

	// 获取版本信息
	glType = (context()->isOpenGLES()) ? "OpenGL ES" : "OpenGL";
	glVersion = reinterpret_cast<const char*>(glGetString(GL_VERSION));

	// 获取 Profile 信息
#define CASE(c) case QSurfaceFormat::c: glProfile = #c; break
	switch (format().profile())
	{
		CASE(NoProfile);
		CASE(CoreProfile);
		CASE(CompatibilityProfile);
	}
#undef CASE

	qDebug() << qPrintable(glType) << qPrintable(glVersion) << "(" << qPrintable(glProfile) << ")";
}
void CoreFunctionWidget::initializeGL() {
	this->initializeOpenGLFunctions();
	//qDebug() << "11111111111111111";
	/*glClearColor(0.0f, 0.0f, 0.0f, 0.5f);				// Black Background
	glShadeModel(GL_SMOOTH);							// Enable Smooth Shading
//	glShadeModel(GL_FLAT);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	//	glCullFace(GL_BACK);
	//	glEnable(GL_CULL_FACE);

	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	///////////////////////////////////////////
//	glAlphaFunc(GL_GEQUAL,0.1f);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE); //字体显示要求,不能少
	glDepthFunc(GL_LEQUAL);
	glReadBuffer(GL_FRONT);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1); //不能去掉,否则会非法操作,不知为什么

	float fogColor[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
	glDisable(GL_FOG);						// Turn on fog
	glFogi(GL_FOG_MODE, GL_LINEAR);			// Set the fog mode to LINEAR (Important)
	glFogfv(GL_FOG_COLOR, fogColor);		// Give OpenGL our fog color
	glFogf(GL_FOG_START, 0.0);				// Set the start position for the depth at 0
	glFogf(GL_FOG_END, 30.0);				// Set the end position for the detph at 50
	// Now we tell OpenGL that we are using our fog extension for per vertex
	// fog calculations.  For each vertex that needs fog applied to it we must
	// use the glFogCoordfEXT() function with a depth value passed in.
	// These flags are defined in main.h and are not apart of the normal opengl headers.
	glFogi(GL_FOG_COORDINATE_SOURCE_EXT, GL_FOG_COORDINATE_EXT);

	glActiveTextureARB = (PFNGLACTIVETEXTUREARBPROC)wglGetProcAddress("glActiveTextureARB");
	glMultiTexCoord2fARB = (PFNGLMULTITEXCOORD2FARBPROC)wglGetProcAddress("glMultiTexCoord2fARB");

	// We should have our multitexturing functions defined and ready to go now, but let's make sure
	// that the current version of OpenGL is installed on the machine.  If the extension functions
	// could not be found, our function pointers will be NULL.
	if (!glActiveTextureARB || !glMultiTexCoord2fARB)
	{
		// Print an error message and quit.
		//MessageBox(NULL, , "Error", MB_OK);
		qDebug() << "Your current setup does not support multitexturing";
		//PostQuitMessage(0);
		return;
	}
	printContextInformation();
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);*/
	//m_Scene.Init();
/*}

void CoreFunctionWidget::resizeGL(int w, int h) {
	glViewport(0, 0, w, h);
}

void CoreFunctionWidget::paintGL() {
	m_Scene.Draw();
	
	glBegin(GL_TRIANGLES);
	glNormal3f(0, -1, 0.707);
	glVertex3f(0.5f, -1.f, 1.f);
	glVertex3f(-0.5, 0.5, 0);
	glVertex3f(-0.5, -0.5, 0);
	glEnd();
	
	/*glBegin(GL_QUADS);
	//glNormal3f(0, 0, -1);
	glVertex3f(0.5f, -0.5f, 0.0f);
	glVertex3f(-0.5, 0.5, 0);
	glVertex3f(-0.5, -0.5, 0);
	glVertex3f(1, -1, 0);
	glEnd();
	
	glBegin(GL_TRIANGLES);
	glNormal3f(1, 0, 0.707);
	glVertex3f(1, -1, 0);
	glVertex3f(1, 1, 0);
	glVertex3f(0, 0, 1.2);
	glEnd();
	glBegin(GL_TRIANGLES);
	glNormal3f(0, 1, 0.707);
	glVertex3f(1, 1, 0);
	glVertex3f(-1, 1, 0);
	glVertex3f(0, 0, 1.2);
	glEnd();
	/*glBegin(GL_TRIANGLES);
	glNormal3f(-1, 0, 0.707);
	glVertex3f(-1, 1, 0);
	glVertex3f(-1, -1, 0);
	glVertex3f(0, 0, 1.2);
	glEnd();
	//	glTranslatef (0.0, 0.0, -10.0);
	//	auxSolidTeapot(1.5);
	//	glTranslatef (0.0, 0.0, -100.0);
	//	glBindTexture(GL_TEXTURE_2D, m_Txt1.GetTxtID());
	//	m_md2.Animate();
}
*/

