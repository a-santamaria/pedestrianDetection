#include "pedestrianRecognizer.h"

PedestrianRecognizer::PedestrianRecognizer() {
    treshold = 0.5;
    initModel();
}

PedestrianRecognizer::PedestrianRecognizer(std::string _modelFileName) {
    //TODO read model from file if already created
    modleFileName = _modelFileName;
    treshold = 0.5;
    initModel();
}

void PedestrianRecognizer::initModel() {
    srand (time(0));
    for (int i = 0; i < modelSize; i++) {
        if(rand()%2 == 0)
            model[i] = ((double) rand() / (RAND_MAX));
        else
            model[i] = -((double) rand() / (RAND_MAX));
    }
    std::cout << "al principio" << std::endl;
    std::cout << model[1] << std::endl;
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
        int voy = 0;
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
    writeModelToFile();
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
    return  - ( ( label     * log( est )   ) +
                ( (1-label) * log( 1-est ) ) );
}

double PedestrianRecognizer::estimateDescriptor(DescriptorLBPH & descriptor) {
    double prediction = 0;
    for (int i = 0; i < modelSize; i++) {
        prediction += model[i] * descriptor.getDescriptorAt(i);
    }
    return sigmoid(prediction);
}

void PedestrianRecognizer::writeModelToFile() {
    /** filewriter **/
    std::ofstream outf( modelFileName.c_str(),  std::ios::out | std::ios::binary);
    outf.write( (char*)&modelSize, sizeof(int) );
    outf.write( (char*)model, sizeof(double)*modelSize);
    outf.close();
}

void PedestrianRecognizer::readModelFromFile() {
    /** filereader **/
    std::ifstream in( modelFileName.c_str(),  std::ios::in | std::ios::binary);
    in.read( (char*)&modelSize, sizeof(int) );
    in.read( (char*)model, sizeof(double)*modelSize);
    in.close();
}

double PedestrianRecognizer::sigmoid(double x) {
    double e = 2.718281828;
    return 1.0 / (1.0 + exp(-x));
}
