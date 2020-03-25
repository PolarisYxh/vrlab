#include "opengl_qt1.h"

opengl_qt1::opengl_qt1(QWidget *parent)
	: QMainWindow(parent)
{
	gladLoadGL();
	ui.setupUi(this);
	//myopengl = new CoreFunctionWidget(this);
	//myopengl = new QtFunctionWidget(this);
	std::string a = "./Resources/objects/dog/dog.stl";
	std:: string b = "./shader/1.model_loading.vs";
	std::string c = "./shader/1.model_loading.fs";
	myopengl = new QtFunctionWidget1(a.c_str(), b.c_str(), c.c_str(),this);
	//布局管理器
	QWidget* widget = new QWidget();
	totalLayout = new QHBoxLayout();
	leftLayout = new QVBoxLayout();
	middleLayout = new QVBoxLayout();
	rightLayout = new QVBoxLayout();

	middleLayout->addWidget(myopengl);

	//设置布局管理器
	totalLayout->addLayout(leftLayout);
	totalLayout->addLayout(middleLayout);
	totalLayout->addLayout(rightLayout);
	totalLayout->setStretchFactor(leftLayout, 95);
	totalLayout->setStretchFactor(middleLayout, 835);
	totalLayout->setStretchFactor(rightLayout, 60);

	widget->setLayout(totalLayout);
	setCentralWidget(widget);
}
