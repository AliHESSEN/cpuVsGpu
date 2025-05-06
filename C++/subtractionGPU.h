#pragma once

#include <opencv2/core/cuda.hpp> // Grunnleggende st�tte for CUDA-akselerasjon
#include <opencv2/cudaarithm.hpp>// GPU-baserte aritmetiske operasjoner
#include <opencv2/highgui.hpp>   // For vindush�ndtering og GUI-funksjoner som imshow
#include <opencv2/imgproc.hpp>   // Bildebehandlingsfunksjoner for CPU-basert prosessering


void subtractionAlgoGPU(const cv::Mat& image1, const cv::Mat& image2);