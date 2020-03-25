#pragma once
#include "QtFunctionWidget1.h"
#include <QtWidgets/QMainWindow>
#include "ui_opengl_qt1.h"
//#include "CoreFunctionWidget.h"
//#include "QtFunctionWidget.h"

#include <QHBoxLayout>
class opengl_qt1 : public QMainWindow
{
	Q_OBJECT

public:
	opengl_qt1(QWidget *parent = Q_NULLPTR);

private:
	Ui::opengl_qt1Class ui;
	//×Ü²¼¾Ö
	QHBoxLayout* totalLayout;
	QVBoxLayout* leftLayout;
	QVBoxLayout* middleLayout;
	QVBoxLayout* rightLayout;
	//CoreFunctionWidget *myopengl;
	QtFunctionWidget1 *myopengl;
};
