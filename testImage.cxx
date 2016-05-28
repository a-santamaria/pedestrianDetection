#include <cstdio>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <sstream>

#include "PedestrianDetector.h"

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

    PedestrianDetector pd(img, modelFile);

    std::vector< std::pair<Point2d, Point2d> > vec = pd.getBoxes();

    for(int i = 0; i < vec.size(); i++) {
        rectangle(img, vec[i].first, vec[i].second, 255);
    }

    // Create a window for display img.
    namedWindow( "Display window", WINDOW_AUTOSIZE );
    imshow( "Display window", img );
    waitKey(0);
    return 0;
}
