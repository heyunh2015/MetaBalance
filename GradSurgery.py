import math
import torch
from torch.optim.optimizer import Optimizer
import time
import numpy as np
#import tensorflow as tf
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



class GradSurgery(Optimizer):
    r"""Implements GradSurgery algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups

    """
    def __init__(self, params):
        super(GradSurgery, self).__init__(params)

    @torch.no_grad()
    def step(self, loss_array):#, closure=None
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        # loss = None
        # if closure is not None:
        #     with torch.enable_grad():
        #         loss = closure()

        self.balance_GradSurgery(loss_array)

        #return loss

    def balance_GradSurgery(self, loss_array):

      for loss_index, loss in enumerate(loss_array):
        loss.backward(retain_graph=True)
        for group in self.param_groups:
          for p in group['params']:

            if p.grad is None:
              print("breaking")
              break

            if p.grad.is_sparse:
              raise RuntimeError('GradSurgery does not support sparse gradients')

            state = self.state[p]

            if loss_index == 0:
              state['main_task_gradient'] = p.grad
              state['sum_gradient'] = torch.zeros_like(p.data)
            else:
              # we modify the core idea of GradSurgery a bit for auxilary learning: 
              # if an auxilary gradient conflict with the main gradient, 
              # we project the auxilary gradient onto the normal vector of the main gradient
              main_task_gradient = state['main_task_gradient']
              dotproduct = torch.dot(torch.reshape(main_task_gradient, (-1,)), torch.reshape(p.grad, (-1,)))
              if dotproduct<0:
                projectScale = dotproduct/(torch.torm(main_task_gradient)*torch.torm(main_task_gradient))
                p.grad = p.grad - projectScale*main_task_gradient

            state['sum_gradient'] += p.grad

            # have to empty p.grad, otherwise the gradient will be accumulated
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

            if loss_index==len(loss_array) - 1:
              
              p.grad = state['sum_gradient']