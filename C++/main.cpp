


//#include <opencv2/dnn.hpp>

#include "additionGPU.h"
#include "additionCPU.h"
#include "histEqGPU.h"
#include "histEqCPU.h"
#include "edgeDetectionGPU.h"
#include "edgeDetectionCPU.h"
#include "subtractionGPU.h"
#include "subtractionCPU.h"
//#include "gausblurGPU.h"
//#include "blobDetGPU.h"
//#include "shapeDetectionCPU.h"
#include <iostream>




cv::Mat resizeImage(const cv::Mat& inputImage, const cv::Size& newSize) // resize function
{
    cv::Mat resizedImage;
    cv::resize(inputImage, resizedImage, newSize);
    return resizedImage;
}



int main() 

{
   

    
    cv::Mat image1 = cv::imread(""); // legg til path
    cv::Mat image2 = cv::imread(""); 
    cv::Mat image3 = cv::imread(""); 
    cv::Mat image4 = cv::imread(""); 

    // ser om noen ikke fungerer
    if (image1.empty() || image2.empty() || image3.empty() || image4.empty()) {
        std::cout << "Could not load all images from the file system." << std::endl;
        return -1;
    }

    cv::Size newSizeImage = cv::Size(1020, 720); 
    image1 = resizeImage(image1, newSizeImage);
    image2 = resizeImage(image2, newSizeImage);
    image3 = resizeImage(image3, newSizeImage);
    image4 = resizeImage(image4, newSizeImage);


    // run the functions here:

    // Addition algos:

    //additionAlgoGPU(image1, image2, image3, image4); // GPU
    //additionAlgoCPU(image1, image2, image3, image4); //CPU


    // histogramEqualization algos:
  
    //histogramEqualizationAlgoGPU(image1); // GPU
    //histogramEqualizationAlgoCPU(image1); // CPU

    // edge detection algos:

    //edgeDetectionAlgoGPU(image1);
    //edgeDetectionAlgoCPU(image1);
        

    
    
    // Kjør funskjoner 100 ganger
    //for (int antall = 0; antall < 100; ++antall) 
    //{
    //    //std::cout << " " << antall << std::endl;
    //    edgeDetectionAlgoCPU(image1);
    //}

    // subtraction algos:

    //subtractionAlgoGPU(image1, image3);
   // subtractionAlgoCPU(image1, image3);

    // blur algos:

   // gaussianBlurAlgoGPU(image1);

    
    //blob detection on GPU:

   //blobDetectionAlgoGPU(image1);


    // shape detection algo:

    //shapeDetectionAlgoCPU(image2);

    
    // video tracking :

    //roadTrackingCPU(); // CPU
    //roadTrackingGPU(); // GPU

    return 0;


}
