#include "pedestrianRecognizer.h"

PedestrianRecognizer::PedestrianRecognizer() {
    treshold = 0.5;
    initModel();
}

PedestrianRecognizer::PedestrianRecognizer(std::string _modelFileName) {
    //TODO read model from file if already created
    modelFileName = _modelFileName;
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
    std::vector<double> descriptorsEst;
    for (int i = 0; i < images.size(); i++) {
        descriptors.push_back( DescriptorLBPH(images[i]) );
        int last = descriptors.size()-1;
        descriptorsEst.push_back( estimateDescriptor(descriptors[last]) );
    }

    double prevCost = INF;
    double cost = INF;
    int iteration = 0;
    do {
        prevCost = cost;

        prevCost = cost;
        gradiantDescentStep(descriptors, labels, descriptorsEst);
        cost = totalLoss(descriptorsEst, labels);
        std::cout << "------- iteration: "<< iteration++;
        std::cout << " costo: " << cost << std::endl;
        std::cout << "diff " << fabs(prevCost - cost)<< std::endl;
    } while ( fabs(prevCost - cost) >  EPS);

    std::cout << " sali diff " << fabs(prevCost - cost)<< std::endl;
}

void PedestrianRecognizer::gradiantDescentStep(
                                std::vector<DescriptorLBPH>& descriptors,
                                std::vector<int>& labels,
                                std::vector<double>& descriptorsEst) {
    // calculate estimation of descriptors with current model
    for (int j = 1; j < descriptors.size(); j++) {
        descriptorsEst[j] = estimateDescriptor(descriptors[j]);
    }

    for (int i = 0; i < modelSize; i++) {
        if(i % 1000 == 0)
            std::cout << "voy modelo " << i << std::endl;
        double delta_i = 0.0;
        for (int j = 1; j < descriptors.size(); j++) {
            double est = descriptorsEst[j];
            delta_i += (est - labels[j]) * descriptors[j].getDescriptorAt(j);
        }
        /*std::cout << "delta antes de dividir" << delta_i;
        std::cout << " size " << descriptors.size() << std::endl;
        */
        delta_i = (delta_i / (double)descriptors.size());
        /*std::cout << "division " << delta_i << std::endl;
        std::cout << "por alpha " << alpha_GD * delta_i << std::endl;
        */
        model[i] = model[i] - ( alpha_GD * delta_i);
    }
    writeModelToFile();
}

double PedestrianRecognizer::totalLoss(
                                std::vector<double>& descriptorsEst,
                                    std::vector<int>& labels) {
    double cost = 0;
    for (int i = 0; i < descriptorsEst.size(); i++) {
        double est = descriptorsEst[i];
        cost += loss(est, labels[i]);
    }
    return -( cost / (double)descriptorsEst.size() );
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
