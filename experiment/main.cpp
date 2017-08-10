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
    Caffe::set_mode(Caffe::GPU);
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

    //display optimized image
}