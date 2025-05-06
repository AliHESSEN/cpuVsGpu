#include <iostream>
#include "histEqCPU.h"

/**
 * @brief Performs histogram equalization on an input image using both GPU and CPU.
 *
 * This function takes an input image and performs histogram equalization on it
 * using CPU. It measures the time taken for the histogram equalization
 * and displays the original image along with the processed results.
 *
 * @param image1 The input image for histogram equalization.
 *
 * @details The function follows these steps:

 * - Converts the input image to grayscale on CPU using OpenCV's cvtColor function.
 * - Performs histogram equalization on CPU using OpenCV's equalizeHist function.
 * - Measures the time taken for CPU histogram equalization.
 * - Displays the original image and the processed results using OpenCV's imshow function.
 */

void histogramEqualizationAlgoCPU(const cv::Mat& image1)
{
    
    cv::Mat resultCpuHistEq; // skal bruke denne variabeln for � resultat bildet etter histeq
    cv::TickMeter cpuTimeHistEq; // skal brukes for � m�le tiden

    
    cpuTimeHistEq.start(); // starter m�lingen
    cv::cvtColor(image1, resultCpuHistEq, cv::COLOR_BGR2GRAY); // gj�r om til greyscale tatt fra chatgpt
    cv::equalizeHist(resultCpuHistEq, resultCpuHistEq); // kj�rer histeq funksjonen p� bildet
    cpuTimeHistEq.stop(); // stopper m�lingen av tid

    // gj�r om tid til sekunder ved � bruke 1e-3
    double timeUsedOnCpuHistEq = cpuTimeHistEq.getTimeMilli() * 1e-3; 
    std::cout << "Time spent by CPU for histogram equalization: " << timeUsedOnCpuHistEq << " seconds \n" << std::endl; // skriver ut tiden
    std::cout << "----------------------------------------------------------------------------------------" << std::endl; 

    
    cv::imshow("Original Image 1", image1); // originalbildet
    cv::imshow("CPU-Processed Image (Hist. Equalization)", resultCpuHistEq); // viser resultatet her
    cv::waitKey(0); 
    cv::destroyAllWindows(); 
}
