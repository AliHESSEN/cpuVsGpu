#pragma once

#include <opencv2/imgproc.hpp> // For bildebehandlingsfunksjoner som subtract
#include <opencv2/highgui.hpp> // For vindush�ndtering og GUI-funksjoner som imshow


void subtractionAlgoCPU(const cv::Mat& image1, const cv::Mat& image2);

