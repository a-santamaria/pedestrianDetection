#include <cstdio>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <sstream>

#include "mlpack/core.hpp"
#include "mlpack/methods/logistic_regression/logistic_regression.hpp"
#include "descriptorLBPH.h"
#include "pedestrianRecognizer.h"

using namespace cv;
using namespace mlpack;
using namespace mlpack::regression;

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

    // PedestrianRecognizer model(modelFile);

    // double prob = model.pedestrianProbability(img);

    LogisticRegression<> model(0,0);
    data::Load(modelFile, "logReg_model", model);
    std::cout << "cargo" << std::endl;
    arma::mat predictors(DescriptorLBPH::desSize, 1);
    arma::Row<size_t> responses(1);

    DescriptorLBPH dlbp = DescriptorLBPH(img);
    for(int j = 0; j < DescriptorLBPH::desSize; j++) {
        //TODO revisar que si sea en este orden
        predictors(j, 0) = dlbp.descriptor[j];
    }
    //std::cout << "predictors " << predictors << std::endl;
    model.Predict(predictors, responses);
    double prob = responses(0);
    std::cout << "responses " << responses << std::endl;
    std::cout << "pedestrian probability: " << prob << std::endl;
    // Create a window for display img.
    namedWindow( "Display window", WINDOW_AUTOSIZE );
    imshow( "Display window", img );
    waitKey(0);
    return 0;
}
