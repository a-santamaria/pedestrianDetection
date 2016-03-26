#ifndef LOCAL_BIANRY_PATTERN
#define LOCAL_BIANRY_PATTERN

#include <opencv2/opencv.hpp>

using namespace cv;

class PedestrianRecognizer {
private:
    /** imput image **/
    const Mat_<uchar> img;
    /**
    * image desciptor: normalized histograms concatenated values between 0..1
    * 59 posibilities * 105 blocks = 6195 + 1 -> first position = 1
    **/
    double descriptor [6196];
    const int w_with = 64;
    const int w_height = 128;
    // hardcoded 8-neighbour case
    static uchar uniform[256];

   // 8-neighbours
   const int xs[8] = { -1,  0,  1, -1, 1, -1, 0, 1 };
   const int ys[8] = { -1, -1, -1,  0, 0,  1, 1, 1 };

public:
    Mat_<uchar> imgLBP;
    PedestrianRecognizer();
    PedestrianRecognizer(const Mat _img);
    uchar lbp(int x, int y);
    void createHistogram();
    void calculateHistogram(int x1, int x2, int y1, int y2,
                                 double* histNormalized);
};

#endif
