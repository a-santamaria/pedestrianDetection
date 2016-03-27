#include "pedestrianRecognizer.h"

PedestrianRecognizer::PedestrianRecognizer() {
    treshold = 0.5;
    memset(model, 0, sizeof(model));
    //TODO search for file of model if any
}

void PedestrianRecognizer::train(std::vector<Mat> images,
                                    std::vector<int> lables) {
    //create descriptros
    std::vector<DescriptorLBPH> descriptors;
    for (int i = 0; i < images.size(); i++) {
        descriptors.push_back( DescriptorLBPH(images[i]) );
    }

    double prevCost = INF;
    double cost = totalLoss(descriptors, lables);
    while ( abs(prevCost - cost) >  EPS) {
        prevCost = cost;
        gradiantDescentStep(descriptors, lables);
        cost = totalLoss(descriptors, lables);
    }
}

void PedestrianRecognizer::gradiantDescentStep(
                                std::vector<DescriptorLBPH>& descriptors,
                                std::vector<int>& lables) {
    for (int i = 0; i < modelSize; i++) {
        double delta_i = 0;
        for (int j = 1; j < descriptors.size(); j++) {
            double est = estimateDescriptor(descriptors[j]);
            delta_i += (est - lables[j]) * descriptors[j].getDescriptorAt(j);
        }
        delta_i = (delta_i / (double)descriptors.size());
        model[i] = model[i] - ( alpha_GD * delta_i);
    }
}

double PedestrianRecognizer::totalLoss(
                                std::vector<DescriptorLBPH>& descriptors,
                                    std::vector<int>& lables) {
    double cost = 0;
    for (int i = 0; i < descriptors.size(); i++) {
        double est = estimateDescriptor(descriptors[i]);
        cost += loss(est, lables[i]);
    }
    return -( cost / (double)descriptors.size() );
}

double PedestrianRecognizer::loss(double est, int lable) {
    //lable: 0 | 1
    return -( ( lable * log(est) ) + ( (1-lable) * log(1-est) ) );
}

double PedestrianRecognizer::estimateDescriptor(DescriptorLBPH & descriptor) {
    double prediction = 0;
    for (int i = 0; i < modelSize; i++) {
        prediction += model[i] * descriptor.getDescriptorAt(i);
    }
}
