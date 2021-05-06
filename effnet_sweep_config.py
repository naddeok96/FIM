sweep_config = {
                'method': 'bayes',
                'metric': {
                    'name': 'loss',
                    'goal': 'minimize'   
                },
                'parameters': {
                    'epochs': {
                        'values': [5e2, 1e3, 5e3, 1e4, 5e4]
                    },
                    'batch_size': {
                        'values': [32, 64, 124, 256]
                    },
                    'momentum': {
                        'values': [0.9]
                    },
                    'weight_decay': {
                        'values': [0, 1e-3, 5e-3]
                    },
                    'learning_rate': {
                        'values': [1e-1, 1e-2, 1e-3]
                    },
                    'optimizer': {
                        'values': ['sgd', 'adadelta']
                    },
                    'criterion': {
                        'values': ["mse", 'cross_entropy']  
                    },
                    'use_SAM':{
                        'values' : [True, False]
                    },
                    'pretrained':{
                        'values' : [True, False]
                    },
                    'model_name':{
                        'values' : ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
                                    'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7']
                    }
                }
            }
            