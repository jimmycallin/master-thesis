from tensorflow.train import (GradientDescentOptimizer,
                              AdaGradOptimizer,
                              MomentumOptimizer,
                              RMSPropOptimizer)

def get_optimizers():
    return {'gradient_descent': GradientDescentOptimizer,
            'adagrad': AdaGradOptimizer,
            'momentum': MomentumOptimizer,
            'rmsprop': RMSPropOptimizer}
