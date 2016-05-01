#include <cstdio>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <sstream>

#include "descriptorLBPH.h"
#include "pedestrianRecognizer.h"

using namespace cv;


int main(int argc, char** argv ) {

    if ( argc < 3 ) {
        std::cerr << "Usage: " << argv[ 0 ] << " imageFile modelFile " << std::endl;
        return( -1 );
    }
    std::string imageFile = std::string(argv[1]);
    std::string modelFile = std::string(argv[2]);

    Mat img;
    img = imread(imageFile.c_str(), 0);

    if ( !img.data ) {
        std::cerr << "Error: Could not open or find the image" << std::endl;
        return -1;
    }

    PedestrianRecognizer model(modelFile);

    double prob = model.pedestrianProbability(img);
    std::cout << "pedestrian probability: " << prob << std::endl;
    // Create a window for display img.
    namedWindow( "Display window", WINDOW_AUTOSIZE );
    imshow( "Display window", img );
    waitKey(0);
    return 0;
}
