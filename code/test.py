#import convolutional_mlp, dbn, logistic_cg, logistic_sgd, mlp, rbm, SdA_loops, SdA
import convolutional_mlp, logistic_cg, logistic_sgd, mlp
from nose.plugins.skip import SkipTest
#TODO: dbn, rbm, SdA, SdA_loops, convolutional_mlp
def test_logistic_sgd():
    logistic_sgd.sgd_optimization_mnist(n_iter=10)
def test_logistic_cg():
    logistic_cg.cg_optimization_mnist(n_iter=10)
def test_mlp():
    mlp.sgd_optimization_mnist(n_iter=5)
def test_convolutional_mlp():
    convolutional_mlp.evaluate_lenet5(n_iter=5,nkerns=[5,5])
def test_dbn():
    raise SkipTest('Implementation not finished')
def test_rbm():
    raise SkipTest('Implementation not finished')
def test_SdA():
    raise SkipTest('Implementation not finished')
    SdA.sgd_optimization_mnist(pretraining_epochs = 2, n_iter = 3)