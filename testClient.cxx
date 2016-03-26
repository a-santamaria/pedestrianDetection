#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char** argv ) {

    if ( argc < 2 ) {
        std::cerr << "Usage: " << argv[ 0 ] << " image_file" << std::endl;
        return( -1 );
    }

    Mat img;
    //read in gray scale

    img = imread(argv[1], 0);

    if ( !img.data ) {
        std::cerr << "Error: Could not open or find the image" << std::endl;
        return -1;
    }

    // Create a window for display img.
    namedWindow( "Display window", WINDOW_AUTOSIZE );
    imshow( "Display window", img );
    waitKey(0);

    /*
    Mat dst;
    std::stringstream ss( argv[ 1 ] );
    std::string basename;
    getline( ss, basename, '.' );

    imwrite( basename + "_salida.png", dst );
    */
    return 0;
}
