#include "pedestrianRecognizer.h"

PedestrianRecognizer::PedestrianRecognizer() {
    srand (time(0));
    treshold = 0.5;
    for (int i = 0; i < modelSize; i++) {
        if(rand()%2 == 0)
            model[i] = ((double) rand() / (RAND_MAX));
        else
            model[i] = -((double) rand() / (RAND_MAX));
    }
    std::cout << "al principio" << std::endl;
    std::cout << model[1] << std::endl;
    //TODO search for file of model if any
}

void PedestrianRecognizer::train(std::vector<Mat>& images,
                                    std::vector<int>& labels) {
    //create descriptros
    std::vector<DescriptorLBPH> descriptors;
    for (int i = 0; i < images.size(); i++) {
        descriptors.push_back( DescriptorLBPH(images[i]) );
    }

    double prevCost = INF;
    double cost = totalLoss(descriptors, labels);
    std::cout << "costo inicial " << cost << std::endl;
    while ( fabs(prevCost - cost) >  EPS) {
        std::cout << "diff " << fabs(prevCost - cost)<< std::endl;
        prevCost = cost;
        gradiantDescentStep(descriptors, labels);
        cost = totalLoss(descriptors, labels);
        std::cout << "costo " << cost << std::endl;
    }
    std::cout << " sali diff " << fabs(prevCost - cost)<< std::endl;
}

void PedestrianRecognizer::gradiantDescentStep(
                                std::vector<DescriptorLBPH>& descriptors,
                                std::vector<int>& labels) {
    for (int i = 0; i < modelSize; i++) {
        if(i % 100 == 0)
            std::cout << "voy modelo " << i << std::endl;
        double delta_i = 0;
        for (int j = 1; j < descriptors.size(); j++) {
            double est = estimateDescriptor(descriptors[j]);
            delta_i += (est - labels[j]) * descriptors[j].getDescriptorAt(j);
        }
        /*std::cout << "delta antes de dividir" << delta_i;
        std::cout << " size " << descriptors.size() << std::endl;
        */
        delta_i = (delta_i / (double)descriptors.size());
        /*std::cout << "division " << delta_i << std::endl;
        std::cout << "por alpha " << alpha_GD * delta_i << std::endl;
        */
        double anterior = model[i];
        model[i] = model[i] - ( alpha_GD * delta_i);
    }
}

double PedestrianRecognizer::totalLoss(
                                std::vector<DescriptorLBPH>& descriptors,
                                    std::vector<int>& labels) {
    double cost = 0;
    for (int i = 0; i < descriptors.size(); i++) {
        double est = estimateDescriptor(descriptors[i]);
        cost += loss(est, labels[i]);
    }
    return -( cost / (double)descriptors.size() );
}

double PedestrianRecognizer::loss(double est, int label) {
    //label: 0 | 1
    return -( ( label * log(est) ) + ( (1-label) * log(1-est) ) );
}

double PedestrianRecognizer::estimateDescriptor(DescriptorLBPH & descriptor) {
    double prediction = 0;
    for (int i = 0; i < modelSize; i++) {
        prediction += model[i] * descriptor.getDescriptorAt(i);
    }
    double result = 1 / (1 + exp(prediction));
    return result;
}
