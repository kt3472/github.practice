import numpy as np

class SGD:
    '''
    확률적 경사하강법(Stochastic Gradient Descent)
    '''
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]


#class Momentum:
    '''
    모멘텀 SGG(Momentum SGD)
    '''

#class Nesterov:
    '''
    네스테로프 가속 경사(NAG; Nesterov's Accelerated Gradient) (http://arxiv.org/abs/1212.0901)
    '네스테로프 모멘텀 최적화'라고도 한다.
    '''

#class AdaGrad:
    '''
    AdaGrad
    '''

#class RMSprop:
    '''
    RMSprop
    '''

#class Adam:
    '''
    Adam (http://arxiv.org/abs/1412.6980v8)
    '''