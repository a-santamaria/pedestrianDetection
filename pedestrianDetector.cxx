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
    std::cout << "original size " << img.cols << ", " << img.cols << std::endl;
    Mat temp = img;
    //dst = img;
    for(int lev = 0; lev <= pyrHeight; lev++) {

        if(lev != 0) {
            pyrDown(temp, dst);
            temp = dst;
        }

        std::cout << "dest size " << temp.cols << ", " << temp.cols << std::endl;
        if(temp.cols < 64 || temp.rows < 128) break;

        for(int i = 0; i < temp.cols-64; i+=dx) {
            for(int j = 0; j < temp.rows-128; j+=dy) {

                Rect roi(i, j, 64, 128);
                Mat window = temp(roi);
                double prob = pr.pedestrianProbability(window);
                if(prob >= threshold) {
                    std::cout << j*(1<<lev)  << std::endl;
                    namedWindow( "Display window1", WINDOW_AUTOSIZE );
                    imshow( "Display window1", window );
                    std::cout << "prob " << prob << std::endl;
                    waitKey(0);
                    boxes.push_back(
                        std::make_pair(
                            Point2d( i*(1<<lev)     , j*(1<<lev)       ),
                            Point2d( (i+64)*(1<<lev), (j+128)*(1<<lev) )
                        )
                    );
                }
            }
        }

    }
}

std::vector< std::pair<Point2d, Point2d> >
PedestrianDetector::getBoxes()
{
    return boxes;
}
