#ifndef LOCAL_BIANRY_PATTERN
#define LOCAL_BIANRY_PATTERN

#include <opencv2/opencv.hpp>

using namespace cv;

class PedestrianRecognizer {
private:
    const Mat_<uchar> img;

    // hardcoded 8-neighbour case
    static uchar uniform[256];

   // 8-neighbours
   int xs[8] = { -1,  0,  1, -1, 1, -1, 0, 1 };
   int ys[8] = { -1, -1, -1,  0, 0,  1, 1, 1 };

public:
    Mat_<uchar> imgLBP;
    PedestrianRecognizer();
    PedestrianRecognizer(const Mat _img);
    uchar lbp(int x, int y);
    void calculate();
};

#endif
