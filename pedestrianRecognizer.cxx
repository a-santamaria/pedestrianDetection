#include "pedestrianRecognizer.h"

PedestrianRecognizer::PedestrianRecognizer() {
    threshold = 0.5;
    initModel();
}

PedestrianRecognizer::PedestrianRecognizer(std::string _modelFileName) {
    //TODO read model from file if already created
    modelFileName = _modelFileName;
    threshold = 0.5;
    if(!readModelFromFile()) {
        std::cout << "no modle file" << std::endl;
        initModel();
    }
}

void PedestrianRecognizer::initModel() {
    srand (time(0));
    for (int i = 0; i < modelSize; i++) {
        if(rand()%2 == 0)
            model[i] = ((double) rand() / (RAND_MAX));
        else
            model[i] = -((double) rand() / (RAND_MAX));
    }
}

double PedestrianRecognizer::pedestrianProbability(const Mat& img) {
    DescriptorLBPH descriptor(img);
    return estimateDescriptor(descriptor);
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
        double delta_i = 0.0;
        for (int j = 1; j < descriptors.size(); j++) {
            double est = descriptorsEst[j];
            delta_i += (est - labels[j]) * descriptors[j].getDescriptorAt(j);
        }
        delta_i = (delta_i / (double)descriptors.size());
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

bool PedestrianRecognizer::readModelFromFile() {
    /** filereader **/
    std::ifstream in( modelFileName.c_str(),  std::ios::in | std::ios::binary);
    if(!in) return false;
    in.read( (char*)&modelSize, sizeof(int) );
    in.read( (char*)model, sizeof(double)*modelSize);
    in.close();
    // printModel();
    return true;
}

void PedestrianRecognizer::printModel() {
    for (int i = 0; i < modelSize; i++) {
        std::cout << model[i] << std::endl;
    }
}

double PedestrianRecognizer::sigmoid(double x) {
    double e = 2.718281828;
    return 1.0 / (1.0 + exp(-x));
}
