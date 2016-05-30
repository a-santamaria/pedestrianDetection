#ifndef PEDESTRIAN_DETECTOR_H
#define PEDESTRIAN_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <queue>
#include <algorithm>

using namespace cv;

struct PreDeteccion
{
    double prob;
    Point2d p, q;

    PreDeteccion();
    PreDeteccion(double _prob, Point2d _p, Point2d _q);
    bool operator<(const PreDeteccion& other) const;
};

class PedestrianDetector {
private:
    /** imput image **/
    const Mat_<uchar> img;
    double pyrHeight = 8;
    int dx = 16;
    int dy = 16;
    double threshold = 0.9;
    std::vector< std::pair<Point2d, Point2d> > boxes;
public:
    PedestrianDetector();
    PedestrianDetector(const Mat& _img, std::string _modelFileName);
    std::vector< std::pair<Point2d, Point2d> > getBoxes();

private:
    void detect(std::string _modelFileName);
    void refinarCandidatos(std::priority_queue<PreDeteccion>& pq);
    double overlap(std::pair<Point2d, Point2d>& b1,
                    std::pair<Point2d, Point2d>& b2);
};

#endif
