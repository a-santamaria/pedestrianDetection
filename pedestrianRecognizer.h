#ifndef PEDESTRIAN_RECOGNIZER_H
#define PEDESTRIAN_RECOGNIZER_H

#include <opencv2/opencv.hpp>
#include "descriptorLBPH.h"
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>

#define INF DBL_MAX
#define EPS 1e-3

using namespace cv;

class PedestrianRecognizer {
private:
    /** model **/
    double model [6196];
    const int modelSize = 6196;
    int treshold;
    /** gradient descent alpha (lerning ratio) **/
    int alpha_GD = 1; //TODO cross validation

    string modelFileName;

    /**
     * one step of gradiant descent to minimize the cost of the model
    **/
    void gradiantDescentStep(std::vector<DescriptorLBPH>& descriptors,
                             std::vector<int>& labels,
                             std::vector<double>& descriptorsEst);
    /**
     * total cost of current model
    **/
    double totalLoss(std::vector<double>& descriptorsEst,
                        std::vector<int>& labels);
    /**
     * cost of especific descriptor estimated by est
     * est: [0..1]
     * label: {0,1}
    **/
    double loss(double est, int label);
    /**
     * probability that descriptor is a pedestrian
     * return [0..1]
    **/
    double estimateDescriptor(DescriptorLBPH & descriptor);

    /**
     * ligistic function
    **/
    double sigmoid(double x);

    void writeModelToFile();
    void readModelFromFile();

    void initModel();

public:
    PedestrianRecognizer();

    PedestrianRecognizer(std::string _modelFileName);

    /**
     * logistic regression lerning algo
    **/
    void train(std::vector<Mat>& images, std::vector<int>& labels);
};

#endif
