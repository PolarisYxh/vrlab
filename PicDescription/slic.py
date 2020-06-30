import math
import sys
from turtle import Shape

import matplotlib.pyplot as plt
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import seaborn as sns
from pycuda.compiler import SourceModule
from scipy.interpolate import griddata  # 插值
from scipy.spatial import Delaunay
from skimage import color, io
from skimage.segmentation import mark_boundaries, slic


def Laplas():
    return
def getD(vals,vecs,Msize,index):
    D=np.zeros(Msize)
    for j in range(0,Msize):
        s=0
        for k in range(0,Msize):
            s+=((vecs[index][k]-vecs[j][k])/vals[k])**2
        D[j]=np.sqrt(s)
    D=D/np.max(D)
    return D

def interPolateCuda(TriIndex,DPoint):#TriIndex=M,是图的邻接矩阵
    mod = SourceModule("""
    __global__ void matrix_cal(float *dest, float *a, float *b, int length)
    {
        int i = threadIdx.x;//+ blockDim.x * blockIdx.x;//顶点i
        float sum = 0;
        for(int k=0;k<length;k++)//step 1
        {
            if(k==i)continue;
            if(b[i][k]!=0)
            {
                int num=30*((a[i*4]-a[k*4])*(a[i*4]-a[k*4])+(a[i*4+1]-a[k*4+1])*(a[i*4+1]-a[k*4+1]));//插值点数目
                float* interPoint=malloc(num*4*sizeof(float))
                for(int m=0;m<num;m++)
                {
                    for(int n=0;n<4;n++)
                        interPoint[m*4+n]=(a[k*4]-a[i*4])/(float)num*m+a[i*4];
                }
            }
        }
        for(int k=0;k<length;k++)//step 2
        {

        }
    }
    """)
    matrix_mul = mod.get_function("matrix_cal")
    matrix_mul(cuda.Out(D),cuda.Out(),cuda.In(DPoint), cuda.In(TriIndex),TriIndex.shape[0], block=(TriIndex.shape[0],1,1))   #

def interPolate(m,n):
    Vm=point3d[m]
    Vn=point3d[n]
    k=(Vm-Vn)**2//(1/30)
    for i in range(0,k):
        t1=np.concatenate(((Vn-Vm)/k*i+Vm,(D[n]-D[m])/k*i+D[m]),axis=1)
        t=np.concatenate((t,t1),axis=0)
    return
if  __name__ == '__main__':
    img = io.imread("2.jpg")#nature_monte.bmp
    #slic
    #gray_img=color.rgb2gray(img)
    segments = slic(img, n_segments=600, compactness=10)
    points2d=[]
    points3d=[]
    rgb_mean=[]
    MtrSize=np.max(segments)+1#顶点数目
    #三角化和make manifest mesh,循环各个mask
    for i in range(0,MtrSize):
        mask_point=np.where(segments==i)
        seg_img=img[mask_point[0][:],mask_point[1]]
        z=np.mean(seg_img)#z值为rgb的平均值
        rgb=np.mean(seg_img,axis=0)
        rgb_mean.append(rgb)
        p=np.mean(mask_point,axis=1)
        points2d.append(p)#用于平面显示的点集
        points3d.append([p[0],p[1],z])
    #获取三角形索引
    delaunay=Delaunay(points2d).simplices
    #构造双调和距离场
    point2d=np.array([[pt[0],pt[1]] for pt in points2d])
    point3d=np.array([[pt[0],pt[1],pt[2]] for pt in points3d])
    rgb=np.array(rgb_mean)
    M=np.zeros((MtrSize,MtrSize))
    A=np.zeros((MtrSize,MtrSize))
    #遍历索引值，计算M和A矩阵
    for indice in delaunay:
        i=indice[0]
        j=indice[1]
        k=indice[2]
        A[i][i]+=1
        A[j][j]+=1
        A[k][k]+=1
        a=np.sqrt(np.sum((point2d[i] - point2d[j])**2)+np.sum(np.absolute(rgb[i]-rgb[j]))**2)
        b=np.sqrt(np.sum((point2d[j] - point2d[k])**2)+np.sum(np.absolute(rgb[j]-rgb[k]))**2)
        c=np.sqrt(np.sum((point2d[i] - point2d[k])**2)+np.sum(np.absolute(rgb[i]-rgb[k]))**2)
        r1=math.acos((a*a-b*b-c*c)/(-2*b*c))#bc夹角1
        if(M[i][j]==0):
            M[i][j]=1/math.tan(r1)
        elif(M[j][i]==0):
            M[j][i]=1/math.tan(r1)
        r2=math.acos((b*b-a*a-c*c)/(-2*a*c))#ac夹角2
        if(M[j][k]==0):
            M[j][k]=1/math.tan(r2)
        elif(M[k][j]==0):
            M[k][j]=1/math.tan(r2)
        r3=math.acos((c*c-a*a-b*b)/(-2*a*b))#夹角3
        if(M[i][k]==0):
            M[i][k]=1/math.tan(r3)
        elif(M[k][i]==0):
            M[k][i]=1/math.tan(r3)
    M=-(np.transpose(M)+M)
    for i in range(0,MtrSize):
        M[i][i]=-np.sum(M[i])
    L=np.dot(np.linalg.inv(A),M)
    e_vals,e_vecs = np.linalg.eig(L)#e_vecs是列向量

    #计算D矩阵及绘制热度图
    D=getD(e_vals,e_vecs,MtrSize,268)
    heatImg=np.zeros((img.shape[0],img.shape[1]))
    for i in range(0,MtrSize):
        mask_point=np.where(segments==i)
        heatImg[mask_point[0],mask_point[1]]=D[i]
    sns.heatmap(heatImg,cmap="Spectral")
    #make T structure
    DPoint = [point2d,D]
    T=np.zeros()
    for indices in delaunay:
        pass

    plt.show()

    # plt.xlim((0, 700))
    # plt.ylim((0, 500))
    #plt.triplot(point[:,1], point[:,0], delaunay, linewidth=1.5)#绘制2维mesh
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot_trisurf(point3d[:,1], point3d[:,0],point3d[:,2], linewidth=0.2, antialiased=True)#绘制3维mesh

    # out=mark_boundaries(img,segments)#给img按segments mask画边界线
    # plt.title("n_segments=600")
    # plt.xlabel('x coordinate')
    # plt.ylabel('y coordinate')
    #plt.imshow(img)
    #plt.imshow(out)



#def drawIsoLine():  # 用库函数插值并画等高线
    # y = np.linspace(0,img.shape[0]-1,img.shape[0])
    # x = np.linspace(0,img.shape[1]-1,img.shape[1])
    # grid_x, grid_y = np.meshgrid(x,y)#0到5 1000个数
    # # 用cubic方法插值
    # grid_z = griddata(point2d, D, (grid_x, grid_y), method='linear')
    # # 等值面图绘制
    # plt.contourf(grid_y, grid_x, grid_z)#f填充颜色
    # contour=plt.contour(grid_y, grid_x, grid_z,8,colors='k')#画等高线
    # plt.clabel(contour,fontsize=10,colors='k')
    # #只画z=20和40的线，并将颜色设置为黑色
    # #contour = plt.contour(X,Y,Z,[20,40],colors='k')
    # #等高线上标明z（即高度）的值，字体大小是10，颜色分别是黑色和红色
    # #plt.clabel(contour,fontsize=10,colors=('k','r'))