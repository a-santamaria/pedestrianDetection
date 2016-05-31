#include <cstdio>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <sstream>

#include "pedestrianDetector.h"

using namespace cv;


int main(int argc, char** argv ) {

    if ( argc < 4 ) {
        std::cerr << "Usage: " << argv[ 0 ] << " imageFile modelFile threshold" << std::endl;
        return( -1 );
    }
    std::string imageFile = std::string(argv[1]);
    std::string modelFile = std::string(argv[2]);
    double threshold = std::atof(argv[3]);

    Mat img;
    img = imread(imageFile.c_str(), 0);

    if ( !img.data ) {
        std::cerr << "Error: Could not open or find the image" << std::endl;
        return -1;
    }

    PedestrianDetector pd(img, modelFile, threshold);

    std::vector< std::pair<Point2d, Point2d> > vec = pd.getBoxes();

    for(int i = 0; i < vec.size(); i++) {
        std::cout << "que " << vec[i].first << ", "<< vec[i].second<< std::endl;
        rectangle(img, vec[i].first, vec[i].second, 200);
    }

    // Create a window for display img.
    namedWindow( "Display window", WINDOW_AUTOSIZE );
    imshow( "Display window", img );
    waitKey(0);
    return 0;
}
