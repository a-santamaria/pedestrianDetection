#include "pedestrianDetector.h"
#include "pedestrianRecognizer.h"

PedestrianDetector::PedestrianDetector()
{
}

PedestrianDetector::PedestrianDetector(const Mat& _img,
                                std::string _modelFileName) : img(_img)
{
    detect(_modelFileName);
}

void PedestrianDetector::detect(std::string _modelFileName)
{
    Mat_<uchar> dst;

    PedestrianRecognizer pr(_modelFileName);
    for(int lev = 1; lev <= pyrHeight; lev++) {
        Size s(lev*img.cols/pyrHeight , lev*img.rows/pyrHeight );
        pyrDown(img, dst, s);


        for(int i = 0; i < dst.cols-dx; i+=dx) {
            for(int j = 0; j < dst.rows-dy; j+=dy) {
                Rect roi(i, j, i+64, j+128);
                Mat window = img(roi);
                double prob = pr.pedestrianProbability(window);
                if(prob >= threshold) {
                    //TODO translate with level of pyr
                    boxes.push_back(
                        std::make_pair(Point2d(i, j), Point2d(i+64, j+128)));
                }
            }
        }

    }
}
