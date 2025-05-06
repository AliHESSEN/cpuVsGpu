#pragma once

#include <opencv2/core/cuda.hpp>      // Grunnleggende st�tte for CUDA-akselerasjon
#include <opencv2/cudaimgproc.hpp>    // GPU-basert bildebehandling
#include <opencv2/cudawarping.hpp>    // GPU-basert bildeendring og transformasjon (hvis n�dvendig)
#include <opencv2/cudaarithm.hpp>     // Grunnleggende GPU-baserte aritmetiske og logiske operasjoner
#include <opencv2/cudafilters.hpp>    // GPU-basert bildefiltrering
#include <opencv2/highgui.hpp>        // For vindush�ndtering og GUI-funksjoner som imshow
#include <opencv2/imgproc.hpp>        // Bildebehandling inkludert fargekonverteringer (for CPU-delen)


void edgeDetectionAlgoGPU(const cv::Mat& image1);
