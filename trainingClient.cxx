#include <cstdio>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <sstream>

#include "descriptorLBPH.h"
#include "pedestrianRecognizer.h"
#include "mlpack/core.hpp"
#include "mlpack/methods/logistic_regression/logistic_regression.hpp"
// #include "/home/alfredo/Documents/mlpack-2.0.1/build/include/mlpack/core.hpp"
// #include "/home/alfredo/Documents/mlpack-2.0.1/build/include/mlpack/methods/logistic_regression/logistic_regression.hpp"
using namespace cv;
using namespace mlpack;
using namespace mlpack::regression;
using namespace mlpack::optimization;

static void read_csv(const std::string& filename, std::vector<Mat>& images,
                            std::vector<int>& labels);

int main(int argc, char** argv ) {

    if ( argc < 3 ) {
        std::cerr << "Usage: " << argv[ 0 ] << " csv_file modelFile" << std::endl;
        return( -1 );
    }
    std::string fn_csv = std::string(argv[1]);
    std::string modelFile = std::string(argv[2]);
    std::vector<Mat> images;
    std::vector<int> labels;

    try {
        read_csv(fn_csv, images, labels);
    } catch (cv::Exception& e) {
        std::cerr << "Error opening file \"" << fn_csv;
        std::cerr << "\". Reason: " << e.msg << std::endl;
        return 1;
    }
    std::cout << "leí " << images.size() << " imagenes" << std::endl;
    PedestrianRecognizer model(modelFile);
    model.train(images, labels);

    // arma::mat x;
    // arma::Row<size_t> y;
    LogisticRegression<> model2(0,0);

    return 0;
}

static void read_csv(const std::string& filename, std::vector<Mat>& images,
                            std::vector<int>& labels) {
    char separator = ';';
    std::ifstream file(filename.c_str(), std::ifstream::in);
    if (!file) {
        std::cerr <<  "No vad input file" << std::endl;
    }
    std::string line, path, classlabel;
    while (getline(file, line)) {
        std::stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path.c_str(), 0));
            if( !images[images.size()-1].data ) {
                std::cerr << "Error reading file " << path << std::endl;
            }
            labels.push_back(std::atoi(classlabel.c_str()));
        }
    }
}
