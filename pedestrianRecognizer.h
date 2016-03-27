#ifndef FACE_RECOGNIZER_H
#define FACE_RECOGNIZER_H

#include <opencv2/opencv.hpp>
#include "descriptorLBPH.h"
#include <cmath>

#define INF DBL_MAX
#define EPS 1e-2

using namespace cv;

class PedestrianRecognizer {
private:
    /** model **/
    double model [6196];
    const int modelSize = 6196;
    int treshold;
    /** gradient descent alpha (lerning ratio) **/
    //TODO cross validation
    int alpha_GD = 0.05;
public:
    PedestrianRecognizer();

    /**
     * logistic regression lerning algo
    **/
    void train(std::vector<Mat> images, std::vector<int> lables);
    /**
     * one step of gradiant descent to minimize the cost of the model
    **/
    void gradiantDescentStep(std::vector<DescriptorLBPH>& descriptors,
                                std::vector<int>& lables);
    /**
     * total cost of current model
    **/
    double totalLoss(std::vector<DescriptorLBPH>& descriptors,
                        std::vector<int>& lables);
    /**
     * cost of especific descriptor estimated by est
     * est: [0..1]
     * lable: {0,1}
    **/
    double loss(double est, int lable);
    /**
     * probability that descriptor is a pedestrian
     * return [0..1]
    **/
    double estimateDescriptor(DescriptorLBPH & descriptor);

};

#endif
