'''
This class implements the One Step Spectral Attack as formulated in the paper
"The Adversarial Attack and Detection under the Fisher Information Metric"
'''

'''
Input: input sample x, corresponding labels y, a deep learning model with the output p(y|x) and the loss J(y, x)
'''
import torch

class OSSA:

    def __init__(self, net, 
                       image, 
                       label):

        self.net =net 
        self.image = image
        self.label = label

        self.criterion = torch.nn.CrossEntropyLoss()

        self.output, self.loss = self.get_outputs(self.net, 
                                                  self.image)

        #self.attack_perturbation, self.max_eigen_value = get_attack_elements()
        print(get_attack_elements())
        
    def get_outputs(self, net,
                          image, 
                          label):

        output = net(image)
        loss = self.criterion(output, label)

        return output, loss

    def get_attack_elements(self):

        attack_perturbation = torch.rand(self.image.size())

        
        return attack_perturbation.size()

    
        
