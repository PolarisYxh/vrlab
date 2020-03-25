#include "QtFunctionWidget1.h"

QtFunctionWidget1::QtFunctionWidget1( std::string MP, std::string vSP, std::string fSP,QWidget* parent) :
	ModelPath(MP), vShaderPath(vSP), fShaderPath(fSP),QOpenGLWidget(parent)
{
	camera = std::make_unique<Camera>(QVector3D(0.0f, 0.0f, 30.0f));
	m_bLeftPressed = false;

	m_pTimer = new QTimer(this);
	connect(m_pTimer, &QTimer::timeout, this, [=] {
		m_nTimeValue += 1;
		update();
	});
	m_pTimer->start(40);
}

QtFunctionWidget1::~QtFunctionWidget1() {
	makeCurrent();

	vbo.destroy();
	vao.destroy();

	delete texture1;
	delete texture2;

	doneCurrent();
}

void QtFunctionWidget1::initializeGL() {
	//this->initializeOpenGLFunctions();
	if (!gladLoadGL())//glad及其相关函数只有在initializeGL中才生效
	{
		qDebug() << "Failed to init glad!";
	}
	// configure global opengl state
	// -----------------------------
	glEnable(GL_DEPTH_TEST);
	ourShader=new Shader(vShaderPath.c_str(), fShaderPath.c_str());
	ourModel=new Model(ModelPath.c_str());
}

void QtFunctionWidget1::resizeGL(int w, int h) {
	glViewport(0, 0, w, h);
}

void QtFunctionWidget1::paintGL() {
	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // also clear the depth buffer now!

	camera->processInput(1.0f);

	ourShader->use();

	// view/projection transformations
	QMatrix4x4 projection;
	projection.perspective(camera->zoom, 1.0f * width() / height(), 0.1f, 100.f);
	QMatrix4x4 view =camera->getViewMatrix();
	ourShader->setMat4("projection", projection);
	ourShader->setMat4("view", view);

	// render the loaded model
	QMatrix4x4 model,model1;
	model1= arcball.getRotatonMatrix();
	model.setToIdentity();
	model.translate(-15.0f, -1.0f, 0.0f);// translate it down so it's at the center of the scene
	model.scale(0.2f, 0.2f, 0.2f);// it's a bit too big for our scene, so scale it down
	model*=arcball.getRotatonMatrix();
	ourShader->setMat4("model", model);
	ourModel->Draw(*ourShader);
}

void QtFunctionWidget1::keyPressEvent(QKeyEvent* event)
{
	int key = event->key();
	if (key >= 0 && key < 1024)
		camera->keys[key] = true;
}

void QtFunctionWidget1::keyReleaseEvent(QKeyEvent* event)
{
	int key = event->key();
	if (key >= 0 && key < 1024)
		camera->keys[key] = false;
}

void QtFunctionWidget1::mousePressEvent(QMouseEvent* event)
{
	if (event->button() == Qt::LeftButton) {
		m_bLeftPressed = true;
		m_lastPos = event->pos();
	}
	else if (event->button() == Qt::RightButton) {
		m_bRightPressed = true;
		//m_lastPos = event->pos();
		arcball.press(event->pos());
	}
}

void QtFunctionWidget1::mouseReleaseEvent(QMouseEvent* event)
{
	Q_UNUSED(event);

	m_bLeftPressed = false;
	m_bRightPressed = false;
}

void QtFunctionWidget1::mouseMoveEvent(QMouseEvent* event)
{
	if (m_bLeftPressed) {
		int xpos = event->pos().x();
		int ypos = event->pos().y();

		int xoffset = xpos - m_lastPos.x();
		int yoffset = m_lastPos.y() - ypos;
		m_lastPos = event->pos();
		camera->processMouseMovement(xoffset, yoffset);
	}
	else if (m_bRightPressed) {
		arcball.processMouseMovement(event->pos());
		//m_lastPos = event->pos();
		arcball.press(event->pos());
	}
}
void QtFunctionWidget1::wheelEvent(QWheelEvent* event)
{
	QPoint offset = event->angleDelta();
	camera->processMouseScroll(offset.y() / 20.0f);
}