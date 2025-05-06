#pragma once

#include <opencv2/imgproc.hpp>  // Inkludering av bildebehandlingsfunksjoner som fargeomforming og histogramutjevning.
#include <opencv2/highgui.hpp>  // Inkludering av funksjoner for � vise bilder og h�ndtere brukerinteraksjon med OpenCV-vinduer.


void histogramEqualizationAlgoCPU(const cv::Mat& image1);