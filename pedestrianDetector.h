#ifndef PEDESTRIAN_DETECTOR_H
#define PEDESTRIAN_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;

class PedestrianDetector {
private:
    /** imput image **/
    const Mat_<uchar> img;
    double pyrHeight = 8;
    int dx = 32;
    int dy = 64;
    double threshold = 0.9;
    std::vector< std::pair<Point2d, Point2d> > boxes;
public:
    PedestrianDetector();
    PedestrianDetector(const Mat& _img, std::string _modelFileName);
    std::vector< std::pair<Point2d, Point2d> > getBoxes();

private:
    void detect(std::string _modelFileName);
};

#endif
