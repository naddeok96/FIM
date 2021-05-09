sweep_config = {
                'method': 'bayes',
                'metric': {
                    'name': 'loss',
                    'goal': 'minimize'   
                },
                'parameters': {
                    'epochs': {
                        'values': [1] # [1e2, 1e3]
                    },
                    'batch_size': {
                        'values': [64, 124]
                    },
                    'momentum': {
                        'values': [0.9]
                    },
                    'weight_decay': {
                        'values': [1e-3]
                    },
                    'learning_rate': {
                        'values': [1e-2]
                    },
                    'optimizer': {
                        'values': ['adadelta']
                    },
                    'criterion': {
                        'values': ['mse', 'cross_entropy']  
                    },
                    'use_SAM':{
                        'values' : [False]
                    },
                    'transformation':{
                        'values' : [None]
                    }
                }
            }
             