#pragma once

#include <opencv2/imgproc.hpp>   // For bildebehandlingsfunksjoner som cvtColor og Canny
#include <opencv2/highgui.hpp>   // For vindush�ndtering og GUI-funksjoner som imshow


void edgeDetectionAlgoCPU(const cv::Mat& image1);