#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <memory>

using namespace caffe;

int main()
{
    std::string model_file = "/home/hosery/git/caffe/examples/mnist/my_lenet_train_test.prototxt";
    std::string weights_file = "/home/hosery/git/caffe/examples/mnist/lenet_iter_10000.caffemodel";
    //initialize from a trained network
    //std::shared_ptr<Net<float> > pretrained_net;
    //pretrained_net.reset(new Net<float>(model_file, TEST));
    //pretrained_net->CopyTrainedLayersFrom(weights_file);

    //create solver
    std::string solver_file = "/home/hosery/git/caffe/experiment/lenet_solver.prototxt";
    SolverParameter solver_param;
    caffe::ReadSolverParamsFromTextFileOrDie(solver_file, &solver_param);
    Caffe::set_mode(Caffe::CPU);
    std::shared_ptr<caffe::Solver<float> >
      solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));;

    //copy params from the trained network
    //Net<float> *net = solver->net();
    //copy weight from pretrained layer
    solver->net()->CopyTrainedLayersFrom(weights_file);
    //((ExtendedImageLayer *)solver->net()->layers()[0].get())->setLabel(0);

    //start optimization
    solver->Solve();

    //get first layer of network and convert to image
    shared_ptr<Blob<float>> image_blob = solver->net()->layers()[0]->blobs()[0];
    int target_width = image_blob->width();
    int target_height = image_blob->height();
    float* target_data = image_blob->mutable_cpu_data();
    cv::Mat curImage(target_height, target_width, CV_32FC1, target_data, target_width);

    double min, max;
    cv::minMaxLoc(curImage, &min, &max);
    curImage.convertTo(curImage,CV_8U,255.0/(max-min),-255.0*min/(max-min));

    cv::Size size(500,500);
    cv::resize(curImage,curImage,size);

    //display optimized image
    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display window",curImage);

    cv::waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}