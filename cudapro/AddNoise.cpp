//���ó����ԭͼ�����ӽ�������
//ͼ��ģ����ӽ���������ͨ�������ȡ���ص���Ǹ�����Ϊ�����ȵ���ʵ�ֵ�

#include <cstdlib>
#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace cv;
using namespace std;

Mat addSaltNoise(const Mat srcImage, int n);
double generateGaussianNoise(double m, double sigma);
Mat addGaussianNoise(Mat& srcImag);
int addnoise()
{
	Mat srcImage = imread("nature_monte.bmp");
	if (!srcImage.data)
	{
		cout << "����ͼ������" << endl;
		system("pause");
		return -1;
	}
	imshow("ԭͼ��", srcImage);
	Mat dstImage = addSaltNoise(srcImage, 3000);
	//Mat dstImage = addGaussianNoise(srcImage);
	//imshow("��ӽ���������ͼ��", dstImage);
	//imshow("��Ӹ�˹������ͼ��", dstImage);
	//�洢ͼ��
	imwrite("nature_monte_salty.bmp", dstImage);
	waitKey();
	return 0;
}
//���ɸ�˹����
double generateGaussianNoise(double mu, double sigma)
{
	//����Сֵ
	const double epsilon = numeric_limits<double>::min();
	static double z0, z1;
	static bool flag = false;
	flag = !flag;
	//flagΪ�ٹ����˹�������X
	if (!flag)
		return z1 * sigma + mu;
	double u1, u2;
	//�����������
	do
	{
		u1 = rand() * (1.0 / RAND_MAX);
		u2 = rand() * (1.0 / RAND_MAX);
	} while (u1 <= epsilon);
	//flagΪ�湹���˹�������
	z0 = sqrt(-2.0 * log(u1)) * cos(2 * CV_PI * u2);
	z1 = sqrt(-2.0 * log(u1)) * sin(2 * CV_PI * u2);
	return z0 * sigma + mu;
}

//Ϊͼ����Ӹ�˹����
Mat addGaussianNoise(Mat& srcImag)
{
	Mat dstImage = srcImag.clone();
	int channels = dstImage.channels();
	int rowsNumber = dstImage.rows;
	int colsNumber = dstImage.cols * channels;
	//�ж�ͼ���������
	if (dstImage.isContinuous())
	{
		colsNumber *= rowsNumber;
		rowsNumber = 1;
	}
	for (int i = 0; i < rowsNumber; i++)
	{
		for (int j = 0; j < colsNumber; j++)
		{
			//��Ӹ�˹����
			int val = dstImage.ptr<uchar>(i)[j] +
				generateGaussianNoise(2, 0.8) * 32;
			if (val < 0)
				val = 0;
			if (val > 255)
				val = 255;
			dstImage.ptr<uchar>(i)[j] = (uchar)val;
		}
	}
	return dstImage;
}
Mat addSaltNoise(const Mat srcImage, int n)
{
	Mat dstImage = srcImage.clone();
	for (int k = 0; k < n; k++)
	{
		//���ȡֵ����
		int i = rand() % dstImage.rows;
		int j = rand() % dstImage.cols;
		//ͼ��ͨ���ж�
		if (dstImage.channels() == 1)
		{
			dstImage.at<uchar>(i, j) = 255;		//������
		}
		else
		{
			dstImage.at<Vec3b>(i, j)[0] = 255;
			dstImage.at<Vec3b>(i, j)[1] = 255;
			dstImage.at<Vec3b>(i, j)[2] = 255;
		}
	}
	for (int k = 0; k < n; k++)
	{
		//���ȡֵ����
		int i = rand() % dstImage.rows;
		int j = rand() % dstImage.cols;
		//ͼ��ͨ���ж�
		if (dstImage.channels() == 1)
		{
			dstImage.at<uchar>(i, j) = 0;		//������
		}
		else
		{
			dstImage.at<Vec3b>(i, j)[0] = 0;
			dstImage.at<Vec3b>(i, j)[1] = 0;
			dstImage.at<Vec3b>(i, j)[2] = 0;
		}
	}
	return dstImage;
}