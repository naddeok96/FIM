sweep_config = {
                'method': 'bayes',
                'metric': {
                    'name': 'loss',
                    'goal': 'minimize'   
                },
                'early_terminate' : {
                    "type": "hyperband",
                    "s": 2,
                    "eta": 3,
                    "max_iter": 27
                },
                'parameters': {
                    'epochs': {
                        'values': [3e2]
                    },
                    'batch_size': {
                        'values': [124, 256]
                    },
                    'momentum': {
                        'values': [0.9]
                    },
                    'weight_decay': {
                        'values': [5e-4, 1e-5]
                    },
                    'learning_rate': {
                        'values': [1e-1, 1e-2, 1e-3]
                    },
                    'optimizer': {
                        'values': ['adam', 'adadelta', 'sgd', 'nesterov']
                    },
                    'scheduler': {
                        'values': [None, "Cosine Annealing"]
                    },
                    'criterion': {
                        'values': ['cross_entropy']  
                    },
                    'use_SAM':{
                        'values' : [True]
                    },
                    'pretrained':{
                        'values' : [False]
                    },
                    'transformation':{
                        'values' : ["U"]
                    },
                    'model_name':{
                        'values' : ['cifar10_resnet56']
                    }
                }
            }
             