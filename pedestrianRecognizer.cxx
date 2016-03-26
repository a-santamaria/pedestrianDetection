#include "PedestrianRecognizer.h"

// hardcoded 8-neighbour case
uchar PedestrianRecognizer::uniform[256] = {
    0,1,2,3,4,58,5,6,7,58,58,58,8,58,9,10,11,58,58,58,58,58,58,58,12,58,58,58,
    13,58,14,15,16,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,17,58,58,58,58,
    58,58,58,18,58,58,58,19,58,20,21,22,58,58,58,58,58,58,58,58,58,58,58,58,58,
    58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,23,58,58,58,58,58,58,
    58,58,58,58,58,58,58,58,58,24,58,58,58,58,58,58,58,25,58,58,58,26,58,27,28,
    29,30,58,31,58,58,58,32,58,58,58,58,58,58,58,33,58,58,58,58,58,58,58,58,58,
    58,58,58,58,58,58,34,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
    58,58,58,58,58,58,58,58,58,58,58,58,58,35,36,37,58,38,58,58,58,39,58,58,58,
    58,58,58,58,40,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,41,42,43,58,44,
    58,58,58,45,58,58,58,58,58,58,58,46,47,48,58,49,58,58,58,50,51,52,58,53,54,
    55,56,57
};


PedestrianRecognizer::PedestrianRecognizer() {}

PedestrianRecognizer::PedestrianRecognizer(const Mat _img) : img(_img) {
    imgLBP = Mat::zeros( img.size().height, img.size().width, CV_8U );

    calculate();
}

uchar PedestrianRecognizer::lbp(int x, int y) {
    uchar c = 0;
    uchar curr = img(x, y);
    for(int k = 0; k < 8; k++) {
        if( img(x+xs[k], y+ys[k]) >=  curr) {
            c |= (1 << k);
        }
    }
    return c;
}

void PedestrianRecognizer::calculate() {
    int hist[59];
    memset(hist, sizeof(hist), 0);
    for (int i = 1; i < img.size().height-1; i++) {
        for (int j = 1; j < img.size().width-1; j++) {
            imgLBP(i, j) = lbp(i, j);
            uchar uv = lbp(i,j);
            hist[ PedestrianRecognizer::uniform[uv] ] ++;
        }
    }
}
