/********************************************************************************
** Form generated from reading UI file 'opengl_qt1.ui'
**
** Created by: Qt User Interface Compiler version 5.12.3
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_OPENGL_QT1_H
#define UI_OPENGL_QT1_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_opengl_qt1Class
{
public:
    QAction *actioninput;
    QAction *actionroll;
    QAction *actionroll_2;
    QWidget *centralWidget;
    QMenuBar *menuBar;
    QMenu *menufile;
    QMenu *menucontrol;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *opengl_qt1Class)
    {
        if (opengl_qt1Class->objectName().isEmpty())
            opengl_qt1Class->setObjectName(QString::fromUtf8("opengl_qt1Class"));
        opengl_qt1Class->resize(742, 512);
        actioninput = new QAction(opengl_qt1Class);
        actioninput->setObjectName(QString::fromUtf8("actioninput"));
        actionroll = new QAction(opengl_qt1Class);
        actionroll->setObjectName(QString::fromUtf8("actionroll"));
        actionroll_2 = new QAction(opengl_qt1Class);
        actionroll_2->setObjectName(QString::fromUtf8("actionroll_2"));
        centralWidget = new QWidget(opengl_qt1Class);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        opengl_qt1Class->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(opengl_qt1Class);
        menuBar->setObjectName(QString::fromUtf8("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 742, 26));
        menufile = new QMenu(menuBar);
        menufile->setObjectName(QString::fromUtf8("menufile"));
        menucontrol = new QMenu(menuBar);
        menucontrol->setObjectName(QString::fromUtf8("menucontrol"));
        opengl_qt1Class->setMenuBar(menuBar);
        mainToolBar = new QToolBar(opengl_qt1Class);
        mainToolBar->setObjectName(QString::fromUtf8("mainToolBar"));
        opengl_qt1Class->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(opengl_qt1Class);
        statusBar->setObjectName(QString::fromUtf8("statusBar"));
        opengl_qt1Class->setStatusBar(statusBar);

        menuBar->addAction(menufile->menuAction());
        menuBar->addAction(menucontrol->menuAction());
        menufile->addAction(actioninput);
        menucontrol->addAction(actionroll_2);

        retranslateUi(opengl_qt1Class);

        QMetaObject::connectSlotsByName(opengl_qt1Class);
    } // setupUi

    void retranslateUi(QMainWindow *opengl_qt1Class)
    {
        opengl_qt1Class->setWindowTitle(QApplication::translate("opengl_qt1Class", "opengl_qt1", nullptr));
        actioninput->setText(QApplication::translate("opengl_qt1Class", "input", nullptr));
        actionroll->setText(QApplication::translate("opengl_qt1Class", "roll", nullptr));
        actionroll_2->setText(QApplication::translate("opengl_qt1Class", "roll", nullptr));
        menufile->setTitle(QApplication::translate("opengl_qt1Class", "file", nullptr));
        menucontrol->setTitle(QApplication::translate("opengl_qt1Class", "control", nullptr));
    } // retranslateUi

};

namespace Ui {
    class opengl_qt1Class: public Ui_opengl_qt1Class {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_OPENGL_QT1_H
