//
// Created by LEI XU on 4/27/19.
//

#ifndef RASTERIZER_TEXTURE_H
#define RASTERIZER_TEXTURE_H
#include "global.hpp"
#include <eigen3/Eigen/Eigen>
#include <opencv2/opencv.hpp>
class Texture{
private:
    cv::Mat image_data;

public:
    Texture(const std::string& name)
    {
        image_data = cv::imread(name);
        cv::cvtColor(image_data, image_data, cv::COLOR_RGB2BGR);
        width = image_data.cols;
        height = image_data.rows;
    }

    int width, height;

    Eigen::Vector3f getColor(float u, float v)
    {
        auto u_img = u * width;
        auto v_img = (1 - v) * height;
        auto color = image_data.at<cv::Vec3b>(v_img, u_img);
        return Eigen::Vector3f(color[0], color[1], color[2]);
    }

    // Eigen::Vector3f getColorBilinear(float u, float v){
    //     int w1 = int(u * width), h1 = int(v * height); //bottom left pixel
    //     int w2 = w1 + 1, h2 = h1; //bottom right pixel
    //     int w3 = w1, h3 = h1 + 1; //top left pixel
    //     int w4 = w1 + 1, h4 = h1 + 1; //top right pixel

    //     Eigen::Vector3f bottom_left_color = getColor(w1/width, h1/height); 
    //     Eigen::Vector3f bottom_right_color = getColor(w2/width, h2/height);
    //     Eigen::Vector3f top_left_color = getColor(w3/width, h3/height);
    //     Eigen::Vector3f top_right_color = getColor(w4/width, h4/height);

    //     //horizontal interpolation
    //     Eigen::Vector3f bottom_point = bottom_left_color + (bottom_right_color-bottom_left_color) * (u * width - w1);
    //     Eigen::Vector3f top_point = top_left_color + (top_right_color - top_left_color) * (u * width - w1);

    //     //vertical interpolation
    //     Eigen::Vector3f final_point = bottom_point + (top_point - bottom_point) * (v*height - h1);
    //     return final_point;
    // }



};
#endif //RASTERIZER_TEXTURE_H
