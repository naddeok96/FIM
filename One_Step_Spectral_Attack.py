'''
This class implements the One Step Spectral Attack as formulated in the paper
"The Adversarial Attack and Detection under the Fisher Information Metric"
'''

'''
Input: input sample x, corresponding labels y, a deep learning model with the output p(y|x) and the loss J(y, x)
'''
import torch
import operator

class OSSA:

    def __init__(self, net, 
                       image, 
                       label):

        self.net =net 
        self.image = image
        self.label = label

        self.criterion = torch.nn.CrossEntropyLoss()
        self.soft_max = torch.nn.Softmax(dim = 1)

        self.soft_max_output, self.gradients, self.loss = self.get_outputs(self.net, 
                                                                           self.image,
                                                                           self.label)

        #self.attack_perturbation, self.max_eigen_value = get_attack_elements()
        print(self.get_attack_elements())
        
    def get_outputs(self, net,
                          image, 
                          label):

        output = net(image)
        soft_max_output = self.soft_max(output)

        loss = self.criterion(output, label.unsqueeze(0))
        loss.backward()

        gradients = {}
        for name, param in net.named_parameters():
            gradients[name] = operator.attrgetter(name + '.grad')(net)
            
        return soft_max_output, gradients, loss.item()

    def get_attack_elements(self):

        attack_perturbation = torch.rand(self.image.size())
        
        return attack_perturbation.size()

    
        
