#include <cstdio>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <sstream>

#include "descriptorLBPH.h"
#include "pedestrianRecognizer.h"
#include "mlpack/core.hpp"
#include "mlpack/methods/logistic_regression/logistic_regression.hpp"
#include "mlpack/methods/logistic_regression/logistic_regression_function.hpp"
#include "mlpack/core/optimizers/sgd/sgd.hpp"

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

    //TODO revisar que si sea en este orden
    arma::mat regressors(DescriptorLBPH::desSize, images.size());
    arma::Row<size_t> responses(DescriptorLBPH::desSize);

    std::vector<DescriptorLBPH> descriptors;
    for (int i = 0; i < images.size(); i++) {
        DescriptorLBPH dlbp = DescriptorLBPH(images[i]);
        for(int j = 0; j < DescriptorLBPH::desSize; j++) {
            //TODO revisar que si sea en este orden
            regressors(j, i) = dlbp.descriptor[j];
            //std::cout << "regresor ("<<j<<","<<i<<") "<< regressors(j, i) << " ";
        }

        responses(i) = labels[i];
        //std::cout << std::endl << " lable: " << responses(i) << std::endl;
    }


    LogisticRegression<> model(0,0);
    model.Parameters() = arma::zeros<arma::vec>(regressors.n_rows);

    LogisticRegressionFunction<> lrf(regressors, responses, model.Parameters());
    SGD<LogisticRegressionFunction<>> sgdOpt(lrf);
    //TODO set these
    // sgdOpt.MaxIterations() = 30;
    // sgdOpt.Tolerance() = 0.3;
    // sgdOpt.StepSize() = 10;

    model.Train(sgdOpt);
    std::string outputModelFile = "outputModelFile.txt";
    data::Save(outputModelFile, "logReg_model", model, false);
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
