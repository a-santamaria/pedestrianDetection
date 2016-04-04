#include <cstdio>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;

static void read_csv(const std::string& filename, std::vector<Mat>& images);

int main(int argc, char** argv ) {

    if ( argc < 2 ) {
        std::cerr << "Usage: " << argv[ 0 ] << " csv_file" << std::endl;
        return( -1 );
    }
    std::string fn_csv = std::string(argv[1]);
    std::vector<Mat> images;

    try {
        read_csv(fn_csv, images);
    } catch (cv::Exception& e) {
        std::cerr << "Error opening file \"" << fn_csv;
        std::cerr << "\". Reason: " << e.msg << std::endl;
        return 1;
    }
    std::cout << "leí " << images.size() << " imagenes" << std::endl;

    std::string basename = "Test/negative128x64_";
    for (int i = 0; i < images.size(); i++) {
        int x = 400;
        int y = 400;
        if(images[i].size().height < 528)
            y = 10;
        if(images[i].size().width < 464)
            x = 10;
        int CROPPING_WIDTH = 64;
        int CROPPING_HEIGHT = 128;
        Mat dst = images[i](Rect(x, y, CROPPING_WIDTH, CROPPING_HEIGHT));
        std::cout << "recorte " << i << std::endl;

        imwrite( basename + std::to_string(i) + ".png" , dst );
    }
    return 0;
}

static void read_csv(const std::string& filename, std::vector<Mat>& images) {
    char separator = ';';
    std::ifstream file(filename.c_str(), std::ifstream::in);
    if (!file) {
        std::cerr <<  "No valid input file" << std::endl;
    }
    std::string line, path;
    while (getline(file, line)) {
        std::stringstream liness(line);
        getline(liness, path);
        if(!path.empty()) {
            images.push_back(imread(path.c_str(), 1));
            if( !images[images.size()-1].data ) {
                std::cerr << "Error reading file " << path << std::endl;
            }
        }
    }
}
