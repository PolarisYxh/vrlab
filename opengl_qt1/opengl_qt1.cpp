#include "opengl_qt1.h"

opengl_qt1::opengl_qt1(QWidget *parent)
	: QMainWindow(parent)
{
	gladLoadGL();
	ui.setupUi(this);
	//myopengl = new CoreFunctionWidget(this);
	//myopengl = new QtFunctionWidget(this);
	//std::string a = "./Resources/objects/dog/dog.obj";
	//std::string a = "./Resources/objects/nanosuit/nanosuit.obj";
	std::string a = "./SoftDeformCode/liver model/liver.obj";
	//std:: string b = "./shader/1.model_loading.vs";
	//std::string c = "./shader/1.model_loading.fs";//single texture
	//std::string b = "./shader/textures.vs";
	//std::string c= "./shader/textures.fs";//mix 2 textures
	std::string b = "./shader/textures-phong.vs";
	std::string c = "./shader/textures-phong.fs";//texture-phong1�����㷨��vert�У���gouraudЧ����texture-phong�㷨��frag�У���phongЧ��
	myopengl = new QtFunctionWidget1(a.c_str(), b.c_str(), c.c_str(),this);
	myopengl->setFocusPolicy(Qt::StrongFocus);//or set Qt::ClickFocus for keyboard event
	//���ֹ�����
	QWidget* widget = new QWidget();
	totalLayout = new QHBoxLayout();
	leftLayout = new QVBoxLayout();
	middleLayout = new QVBoxLayout();
	rightLayout = new QVBoxLayout();

	middleLayout->addWidget(myopengl);

	//���ò��ֹ�����
	totalLayout->addLayout(leftLayout);
	totalLayout->addLayout(middleLayout);
	totalLayout->addLayout(rightLayout);
	totalLayout->setStretchFactor(leftLayout, 0);
	totalLayout->setStretchFactor(middleLayout, 835);
	totalLayout->setStretchFactor(rightLayout, 0);

	widget->setLayout(totalLayout);
	setCentralWidget(widget);
}
