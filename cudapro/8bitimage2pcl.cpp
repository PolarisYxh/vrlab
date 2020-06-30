#include <iostream>
#include <fstream>
using namespace std;
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Geometry>

#include <boost/format.hpp> // for formating strings
 #include <pcl/point_types.h>
 #include <pcl/io/pcd_io.h>
 #include <pcl/visualization/pcl_visualizer.h>
 int main( int argc, char** argv )
 {
 vector<cv::Mat> colorImgs, depthImgs; // 彩色图和深度图
 vector<Eigen::Isometry3d> poses; // 相机位姿

 ifstream fin("./pose.txt");
 int num=0;
 fin>>num; 
cout<<num<<endl;
 if (!fin)
 {
 cerr<<"请在有 pose.txt 的目录下运行此程序"<<endl;
 return 1;
 }

 for ( int i=0; i<num; i++ )
 {
 boost::format fmt( "./%s/%d.%s" ); //图像文件格式 ./color/1.png
 string colorname=(fmt%"color"%(i+1)%"png").str();
 string depthname=(fmt%"depth"%(i+1)%"png").str();
 colorImgs.push_back( cv::imread(colorname));
 depthImgs.push_back( cv::imread( depthname, 2 )); // 使用 -1 
 //cv::imshow ( "image", depthImgs[i] ); 
 cout<<depthImgs[i].type()<<"  "<<depthImgs[i].cols<<"  "<<depthname<<"  "<<colorImgs[i].type()<<" ";
 double data[7] = {0};
 for ( auto& d:data )//获取pose.txt的四元数
{
 fin>>d;
 cout<<d;
}
 Eigen::Quaterniond q( data[6], data[3], data[4], data[5] );
 Eigen::Isometry3d T(q);
 T.pretranslate( Eigen::Vector3d( data[0], data[1], data[2] ));
 poses.push_back( T );
 }
/*for(int i=0;i<depthImgs[0].rows;i++)遍历单通道8uc1
    {
       for(int j=0;j<depthImgs[0].cols;j++)
	{
      char nm=0;
if((unsigned char)depthImgs[0].at<unsigned char>(i,j)!=0)
	nm=(unsigned char)depthImgs[0].at<unsigned char>(i,j);//<<(int)depthImgs[0].at<float>(i,j)[1]<<" "<<(int)depthImgs[0].at<float>(i,j)[2]<<" ";
	}
}*/
/*for(int i=0;i<depthImgs[0].rows;i++)
    {
       for(int j=0;j<depthImgs[0].cols;j++)
	{
      char nm=0;
if((int)depthImgs[0].at<cv::Vec3b>(i,j)[0]!=0)
	cout<<(int)depthImgs[0].at<cv::Vec3b>(i,j)[0]<<(int)depthImgs[0].at<cv::Vec3b>(i,j)[1]<<" "<<(int)depthImgs[0].at<cv::Vec3b>(i,j)[2]<<" ";
	}
}*/
//unsigned int *vec=new unsigned int[depthImgs[0].rows*depthImgs[0].cols];
/*for(int i=0;i<depthImgs[0].rows;i++)
{
       for(int j=0;j<depthImgs[0].cols;j++)
	{
      if((unsigned int)depthImgs[0].at<unsigned char>(i,j)!=0)
	cout<<(unsigned int)depthImgs[0].ptr<unsigned char> ( i )[j]<<" ";//unsigned short 16bit
	}
}*/
 // 计算点云并拼接
 // 相机内参
 double cx = 840 ;
 double cy = 525.00000 ;
 double fx = 4763.87464;
 double fy = 4763.87464;
 double zfar=7.15;
 double depthScale = 255.0/zfar;//真实的kinect返回的数据是mm单位的，所以转为m，
   /* double cx = 325.5;
    double cy = 253.5;
    double fx = 518.0;
    double fy = 519.0;
    double depthScale = 1000.0;*/

 cout<<"正在将图像转换为点云..."<<endl;
 // 定义点云使用的格式：这里用的是 XYZRGB
 typedef pcl::PointXYZRGB PointT;
 typedef pcl::PointCloud<PointT> PointCloud;

 // 新建一个点云
 PointCloud::Ptr pointCloud( new PointCloud );
 for ( int i=0; i<num; i++ )
 {
 cout<<"转换图像中: "<<i+1<<endl;
 cv::Mat color = colorImgs[i];
 cv::Mat depth = depthImgs[i];
 Eigen::Isometry3d T = poses[i];
 for ( int v=0; v<color.rows; v++ )
 for ( int u=0; u<color.cols; u++ )
 {
 unsigned int d = depth.ptr<unsigned char> ( v )[u]; // 深度值
 if ( d==0 ) continue; // 为 0 表示没有测量到
 
 Eigen::Vector3d point;
 point[2] = double(d)/depthScale;
 cout<<point[2]<<" ";
 point[0] = (u-cx)*point[2]/fx;
 point[1] = (v-cy)*point[2]/fy;
 Eigen::Vector3d pointWorld = T*point;//外参矩阵求解

 PointT p ;
 p.x = pointWorld[0];
 p.y = pointWorld[1];
 p.z = pointWorld[2];
 p.b = color.data[ v*color.step+u*color.channels() ];
 p.g = color.data[ v*color.step+u*color.channels()+1 ];
 p.r = color.data[ v*color.step+u*color.channels()+2 ];
 pointCloud->points.push_back( p );
  }
}

pointCloud->is_dense = false;
cout<<"点云共有"<<pointCloud->size()<<"个点."<<endl;
pcl::io::savePCDFileBinary("map.pcd", *pointCloud );
return 0;
}
