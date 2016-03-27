#include "descriptorLBPH.h"

// hardcoded 8-neighbour case
uchar DescriptorLBPH::uniform[256] = {
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


DescriptorLBPH::DescriptorLBPH() {}

DescriptorLBPH::DescriptorLBPH(const Mat& _img) : img(_img) {
    imgLBP = Mat::zeros( img.size().height, img.size().width, CV_8U );
    createHistogram();
}

uchar DescriptorLBPH::lbp(int x, int y) {
    uchar c = 0;
    if(x == 0 || y == 0 || x == w_with-1 || y == w_height-1)
        return c;

    uchar curr = img(x, y);
    for(int k = 0; k < 8; k++) {
        if( img(x+xs[k], y+ys[k]) >=  curr) {
            c |= (1 << k);
        }
    }
    return c;
}

void DescriptorLBPH::createHistogram() {
    descriptor[0] = 1;
    int curr = 1;
    for(int i = 0; i < w_height-8; i += 8) {
        for(int j = 0; j < w_with-8; j += 8) {
            calculateHistogram(i, i+15, j, j+15, descriptor+curr);
            curr += 59;
        }
    }

    std::cout << "termine size " << curr << std::endl;

}

void DescriptorLBPH::calculateHistogram(int x1, int x2, int y1, int y2,
                                         double* histNormalized) {
    int hist[59];
    memset(hist, 0, sizeof(hist));
    int total = 0;
    for (int i = x1; i < x2; i++) {
        for (int j = y1; j < y2; j++) {
            imgLBP(i, j) = lbp(i, j);
            uchar uv = lbp(i,j);
            hist[ DescriptorLBPH::uniform[uv] ] ++;
            total++;
        }
    }
    for(int i = 0; i < 59; i++){
        histNormalized[i] = (double)((double)hist[i] / (double)total);
    }
}

double DescriptorLBPH::getDescriptorAt(int i) {
    return descriptor[i];
}
