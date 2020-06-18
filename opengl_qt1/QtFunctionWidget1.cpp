#include "QtFunctionWidget1.h"

QtFunctionWidget1::QtFunctionWidget1( std::string MP, std::string vSP, std::string fSP,QWidget* parent) :
	ModelPath(MP), vShaderPath(vSP), fShaderPath(fSP),QOpenGLWidget(parent)
{
	//camera = std::make_unique<Camera>(QVector3D(0.0f, 0.0f, 20.0f));
	camera= std::make_unique<Camera>(QVector3D(0.0f, 0.0f, 20.0f));
	m_bLeftPressed = false;
	m_bRightPressed = false;
	m_CtrlPressed = false;
	m_CtrlMove = false;
	m_pTimer = new QTimer(this);
	connect(m_pTimer, &QTimer::timeout, this, [=] {
		m_nTimeValue += 1;
		update();
	});
	m_pTimer->start(40);
}

QtFunctionWidget1::~QtFunctionWidget1() {
	makeCurrent();

	//vbo.destroy();
	//vao.destroy();

	//delete texture1;
	//delete texture2;

	doneCurrent();
}

void QtFunctionWidget1::initializeGL() {
	//this->initializeOpenGLFunctions();
	// configure global opengl state
	// -----------------------------
	if (!gladLoadGL())//glad及其相关函数只有在initializeGL中才生效
	{
		qDebug() << "Failed to init glad!";
	}
	////load and show model
	glEnable(GL_DEPTH_TEST);
	ourShader=new Shader(vShaderPath.c_str(), fShaderPath.c_str());
	ourModel=new Model(ModelPath);
	ObjModel=glmReadOBJ(ModelPath.c_str());
	
	ms=new Obj2MassSpring(ObjModel,*ourModel);
	ms->MassSpring();
	////mass spring simulate
	//m =new MassSpring(width(), height());
	//m->DemoInit();
	//cm=new MassSpringCuda(width(), height());
	//cm->DemoInit();
}

void QtFunctionWidget1::resizeGL(int w, int h) {
	glViewport(0, 0, w, h);
}

void QtFunctionWidget1::paintGL() {
	rendermodel();
	//m->UpdateFrame();//for massspring simulate
	//cm->UpdateFrame();
	//change ourModel vertex's position
	//glmDraw(ObjModel, GLM_TEXTURE);
}
void QtFunctionWidget1::rendermodel()//for load and show model
{
	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // also clear the depth buffer now!
	ms->updatePS();//change ourModel vertex's position
	camera->processInput(1.0f);//wasd键盘输入
	ourShader->use();
	ourShader->setVec4("Mat.aAmbient", 0.9f, 0.5f, 0.3f,1);
	ourShader->setVec4("Mat.aDiffuse", 0.9f, 0.5f, 0.3f, 1);
	ourShader->setVec4("Mat.aSpecular", 1.f, 1.f, 1.f, 1);
	//light
	ourShader->setVec3("light.specular", 1.0f, 1.0f, 1.0f);
	ourShader->setVec3("light.diffuse", 1.0f, 1.0f, 1.0f);
	ourShader->setVec3("light.ambient", 1.0f, 1.0f, 1.0f);
	//ourShader->setVec3("light.position", glm::vec3(-13,2,2));
	ourShader->setVec3("light.position", glm::vec3(2, 5., 10));
	ourShader->setVec3("viewPos", camera->position.x(), camera->position.y(), camera->position.z());
	ourShader->setFloat("shininess", 100);
	// view/projection transformations
	QMatrix4x4 projection;
	QMatrix4x4 model;
	projection.perspective(camera->zoom, 1.0f * width() / height(), 0.1f, 100.f);
	QMatrix4x4 view =camera->getViewMatrix();
	ourShader->setMat4("projection", projection);
	ourShader->setMat4("view", view);

	// render the loaded model
	model.setToIdentity();
	//model.translate(-15.0f, -1.0f, 0.0f);// translate it down so it's at the center of the scene
	//model.scale(0.2f, 0.2f, 0.2f);// it's a bit too big for our scene, so scale it down
	model.translate(0.0f, 0.0f, 0.0f);// translate it down so it's at the center of the scene
	model.scale(1.f, 1.f, 1.f);// it's a bit too big for our scene, so scale it down
	model*=arcball.getRotatonMatrix();
	curmodel = model;
	curprojection = projection;
	curview = view;
	curposition = camera->position;
	ourShader->setMat4("model", model);
	GLfloat mouseZ;
	ourModel->Draw(*ourShader);
	//glReadPixels(0, 0, 5, 5, GL_DEPTH_COMPONENT, GL_FLOAT, &mouseZ);
	
}
void QtFunctionWidget1::keyPressEvent(QKeyEvent* event)
{
	int key = event->key();
	if (key >= 0 && key < 1024)
		camera->keys[key] = true;
	switch (event->key())
	{
		case Qt::Key_1:
			//m->setfixed(168,false);
		case Qt::Key_Control:
		{
			m_CtrlPressed = true;
			qDebug() << "ctrl press";
		}

	}
}

void QtFunctionWidget1::keyReleaseEvent(QKeyEvent* event)
{
	int key = event->key();
	if (key >= 0 && key < 1024)
		camera->keys[key] = false;
	switch (event->key())
	{
		case Qt::Key_Control:
		{
			m_CtrlPressed = false;
			qDebug() << "ctrl release";
		}	
	}
}

void QtFunctionWidget1::mousePressEvent(QMouseEvent* event)
{
	if (event->button() == Qt::LeftButton) {
		if (m_CtrlPressed)
		{
			ms->RayPick(event->pos(), curmodel, curprojection, curview, curposition, width(), height(),false);
			/*UnProjectUtilities WorldP(curmodel,curprojection);
			rendermodel();
			WorldP.GetMousePosition(event->pos().x(), event->pos().y(), initposition);*/
			//initpoint=ms->MouseClicked(initposition);
			//m_CtrlMove = true;
			//WorldP.GetMouseRay(event->pos.x(), event->pos.y(), nearpos, farpos);
			//ms->MouseClicked(position,farpos);
		}
		else
		{
			m_bLeftPressed = true;
			m_lastPos = event->pos();
		}
	}
	else if (event->button() == Qt::RightButton) {
		if (m_CtrlPressed)
		{
			ms->RayPick(event->pos(), curmodel, curprojection, curview, curposition, width(), height(),true);
			/*UnProjectUtilities WorldP(curmodel,curprojection);
			rendermodel();
			WorldP.GetMousePosition(event->pos().x(), event->pos().y(), initposition);*/
			//initpoint=ms->MouseClicked(initposition);
			//m_CtrlMove = true;
			//WorldP.GetMouseRay(event->pos.x(), event->pos.y(), nearpos, farpos);
			//ms->MouseClicked(position,farpos);
		}
		else
		{
			m_bRightPressed = true;
			//m_lastPos = event->pos();
			QPoint tem(event->pos().x(), height()-event->pos().y());
			arcball.press(event->pos());
		}
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
		arcball.processMouseMovement(QPoint(event->pos().x(), height()-event->pos().y()));
		//m_lastPos = event->pos();
		arcball.press(QPoint(event->pos().x(), height() - event->pos().y()));
	}
}
void QtFunctionWidget1::wheelEvent(QWheelEvent* event)
{
	QPoint offset = event->angleDelta();
	camera->processMouseScroll(offset.y() / 20.0f);
}