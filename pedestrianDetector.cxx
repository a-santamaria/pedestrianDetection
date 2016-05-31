#include "pedestrianDetector.h"
#include "pedestrianRecognizer.h"

#include "mlpack/core.hpp"
#include "mlpack/methods/logistic_regression/logistic_regression.hpp"

using namespace cv;
using namespace mlpack;
using namespace mlpack::regression;

PreDeteccion::PreDeteccion() {}

PreDeteccion::PreDeteccion(double _prob, Point2d _p, Point2d _q) :
                            prob(_prob), p(_p), q(_q) {}

bool PreDeteccion::operator<(const PreDeteccion& other) const
{
    return this->prob < other.prob;
}

PedestrianDetector::PedestrianDetector() {}

PedestrianDetector::PedestrianDetector(const Mat& _img,
                                std::string _modelFileName) : img(_img)
{
    detect(_modelFileName);
}

void PedestrianDetector::detect(std::string _modelFileName)
{
    Mat_<uchar> dst;

    //PedestrianRecognizer pr(_modelFileName);
    LogisticRegression<> model(0,0);
    data::Load(_modelFileName, "logReg_model", model);

    arma::mat predictors(DescriptorLBPH::desSize, 1);



    std::cout << "original size " << img.cols << ", " << img.rows << std::endl;
    Mat temp = img;
    //dst = img;
    std::priority_queue<PreDeteccion> preDetecciones;
    for(int lev = pyrHeight; lev > 0; lev--) {

        if(lev != pyrHeight) {
            Size s( lev*img.cols/pyrHeight, lev*img.rows/pyrHeight);
            GaussianBlur(temp, temp, Size(5,5), 0);
            resize(temp, temp, s);
            // pyrDown(temp, dst);
            // temp = dst;
        }

        std::cout << "dest size " << temp.cols << ", " << temp.rows << std::endl;
        if(temp.cols < 64 || temp.rows < 128) break;

        for(int i = 0; i < temp.cols-64; i+=dx) {
            for(int j = 0; j < temp.rows-128; j+=dy) {

                Rect roi(i, j, 64, 128);
                Mat window = temp(roi);

                DescriptorLBPH dlbp = DescriptorLBPH(window);
                for(int j = 0; j < DescriptorLBPH::desSize; j++) {
                    //TODO revisar que si sea en este orden
                    predictors(j, 0) = dlbp.descriptor[j];
                }
                arma::Row<size_t> responses;
                model.Predict(predictors, responses, 0.995);
                //std::cout << "predictions " << responses << std::endl;
                if(responses(0) == 1) {
                    boxes.push_back( std::make_pair(
                        Point2d( i*pyrHeight/lev  , j*pyrHeight/lev       ),
                        Point2d( (i+64)*pyrHeight/lev, (j+128)*pyrHeight/lev ) )
                    );
                }
                // double prob = pr.pedestrianProbability(window);
                // // if(temp.cols < 240) {
                // //     namedWindow( "Display window1", WINDOW_AUTOSIZE );
                // //     imshow( "Display window1", window );
                // //     waitKey(0);
                // //     std::cout << "prob " << prob << std::endl;
                // // }
                //
                // if(prob >= threshold) {
                //     // namedWindow( "Display window1", WINDOW_AUTOSIZE );
                //     // imshow( "Display window1", window );
                //     // std::cout << "prob " << prob << std::endl;
                //     // waitKey(0);
                //     preDetecciones.push(
                //         PreDeteccion(
                //             prob,
                //             Point2d( i*pyrHeight/lev  , j*pyrHeight/lev       ),
                //             Point2d( (i+64)*pyrHeight/lev, (j+128)*pyrHeight/lev )
                //         )
                //     );
                //     // boxes.push_back(
                //     //     std::make_pair(
                //     //         Point2d( i*pyrHeight/lev  , j*pyrHeight/lev       ),
                //     //         Point2d( (i+64)*pyrHeight/lev, (j+128)*pyrHeight/lev )
                //     //     )
                //     // );
                // }
            }
        }

    }
    //refinarCandidatos(preDetecciones);
}

void
PedestrianDetector::refinarCandidatos(std::priority_queue<PreDeteccion>& pq)
{
    while(!pq.empty())
    {
        PreDeteccion curr = pq.top();
        pq.pop();
        bool add = true;
        std::pair<Point2d, Point2d> currBox(curr.p, curr.q);
        for(int i = 0; i < boxes.size(); i++)
        {
            if( overlap(boxes[i], currBox) > 0.2 )
            {
                add = false;
                break;
            }
        }
        if(add)
            boxes.push_back(currBox);
    }
}

double
PedestrianDetector::overlap(std::pair<Point2d, Point2d>& a,
                            std::pair<Point2d, Point2d>& b)
{
    double sa = (a.second.x - a.first.x) * (a.second.y - a.first.y);
    double sb = (b.second.x - b.first.x) * (b.second.y - b.first.y);
    double lx = min(a.second.x, b.second.x) - max(a.first.x, b.first.x);
    double ly = min(a.second.y, b.second.y) - max(a.first.y, b.first.y);

    double si = max( 0.0, lx * max( 0.0, ly ) );
    double s = sa + sb - si;
    return si / s;
}

std::vector< std::pair<Point2d, Point2d> >
PedestrianDetector::getBoxes()
{
    return boxes;
}
