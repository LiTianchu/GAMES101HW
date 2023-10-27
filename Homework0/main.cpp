#include<cmath>
#include<eigen3/Eigen/Core>
#include<eigen3/Eigen/Dense>
#include<iostream>

int main(){

    // Basic Example of cpp
    std::cout << "Example of cpp \n";
    float a = 1.0, b = 2.0;
    std::cout << a << std::endl;
    std::cout << a/b << std::endl;
    std::cout << std::sqrt(b) << std::endl;
    std::cout << std::acos(-1) << std::endl;
    std::cout << std::sin(30.0/180.0*acos(-1)) << std::endl;

    // Example of vector
    std::cout << "Example of vector \n";
    // vector definition
    Eigen::Vector3f v(1.0f,2.0f,3.0f);
    Eigen::Vector3f w(1.0f,0.0f,0.0f);
    // vector output
    std::cout << "Example of output \n";
    std::cout << v << std::endl;
    // vector add
    std::cout << "Example of add \n";
    std::cout << v + w << std::endl;
    // vector scalar multiply
    std::cout << "Example of scalar multiply \n";
    std::cout << v * 3.0f << std::endl;
    std::cout << 2.0f * v << std::endl;

    // Example of matrix
    std::cout << "Example of matrix \n";
    // matrix definition
    Eigen::Matrix3f i,j;
    i << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0;
    j << 2.0, 3.0, 1.0, 4.0, 6.0, 5.0, 9.0, 7.0, 8.0;
    // matrix output
    std::cout << "Example of output of matrix i\n";
    std::cout << i << std::endl;
    std::cout << "Example of output of matrix j\n";
    std::cout << j << std::endl;
    // matrix add i + j
    std::cout << "Example of output of i+j \n";
    std::cout << i+j <<std::endl;
    // matrix scalar multiply i * 2.0
    std::cout << "Example of scalar multiple of i*2 \n";
    std::cout << i*2.0 << std::endl;
    // matrix multiply i * j
    std::cout << "Example of output of i*j \n";
    std::cout << i*j << std::endl;
    // matrix multiply vector i * v
    std::cout << "Example of output of i*v \n";
    std::cout << i*v << std::endl;

    //hw0: 给定一个点 P=(2,1), 将该点绕原点先逆时针旋转45◦，再平移(1,2), 计算出变换后点的坐标（要求用齐次坐标进行计算）
    float pi = std::acos(-1);
    float rotateDeg = 45.0f/180.0f * pi;

    Eigen::Vector3f p(2.0f, 1.0f, 1.0f); //define the point in homogeneous coord
    Eigen::Matrix3f r;
    r << std::cos(rotateDeg), -std::sin(rotateDeg), 0, std::sin(rotateDeg), std::cos(rotateDeg), 0, 0, 0, 1; //rotation matrix 45 degree anti clockwise
    Eigen::Matrix3f t;
    t << 1.0f, 0, 1.0f, 0, 1.0f, 2.0f, 0, 0, 1.0f; // translation matrix by (1,2)

    Eigen::Matrix3f m;
    m = t*r; //pre-compute the transformation matrix
    std::cout << "Transformation Matrix: " << std::endl;
    std::cout << m << std::endl;
    std::cout << "Transformed Point:" << std::endl;
    std::cout << m*p << std::endl;

    return 0;
}