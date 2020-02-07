'''
This class implements the One Step Spectral Attack as formulated in the paper
"The Adversarial Attack and Detection under the Fisher Information Metric"
'''

'''
Input: input sample x, corresponding labels y, a deep learning model with the output p(y|x) and the loss J(y, x)
'''
import torch
import operator
from torch.autograd import Variable

class OSSA:

    def __init__(self, net, 
                       image, 
                       label):

        self.net =net 
        self.image = image
        self.label = label

        self.criterion = torch.nn.CrossEntropyLoss()
        self.soft_max = torch.nn.Softmax(dim = 1)

        self.soft_max_output, self.image_gradients, self.loss = self.get_outputs(self.net, 
                                                                           self.image,
                                                                           self.label)

        self.attack_perturbation, self.attack_perturbation_show = self.get_attack_elements()
        
    def get_outputs(self, net,
                          image, 
                          label):

        image = Variable(image, requires_grad = True)
        output = net(image)
        soft_max_output = self.soft_max(output)

        loss = self.criterion(output, label.unsqueeze(0))
        loss.backward(retain_graph = True)

        image_gradients = image.grad.data
            
        return soft_max_output, image_gradients, loss.item()

    def get_attack_elements(self):

        # Initialize Attack
        attack_perturbation = torch.rand(self.image.size())
        attack_perturbation_show = attack_perturbation.unsqueeze(0)

        alias_table = [self.soft_max_output]

        '''
        converged = False
        while converged == False:
            
        '''
        
        return attack_perturbation, attack_perturbation_show

    

    
        
