import os
set_name = "MNIST"
attack_type = ["FGSM"] # ["Gaussian_Noise", "FGSM", "PGD", "CW2", "OSSA"]
files_to_skip = ["UvsNoU", "distill", "weak"]
upper_bound  = 1

sweep_config = {"lr"            : [0.01, 0.001, 0.0001, 0.00001],
                        "optim"         : ["sgd", 'nesterov'],
                        "sched"         : ["One Cycle LR", "Cosine Annealing"],
                        "lsr"           : 0.1,
                        "epochs"        : [25, 50, 100]}

for key in sorted(list(sweep_config.keys()), reverse=True):
    print(key)