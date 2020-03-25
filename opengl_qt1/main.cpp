#include "opengl_qt1.h"
#include <QtWidgets/QApplication>
int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	opengl_qt1 w;
	w.show();
	return a.exec();
}
