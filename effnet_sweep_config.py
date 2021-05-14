sweep_config = {
                'method': 'bayes',
                'metric': {
                    'name': 'loss',
                    'goal': 'minimize'   
                },
                'parameters': {
                    'epochs': {
                        'values': [5e2]
                    },
                    'batch_size': {
                        'values': [32]
                    },
                    'momentum': {
                        'values': [0.9]
                    },
                    'weight_decay': {
                        'values': [1e-4, 1e-5]
                    },
                    'learning_rate': {
                        'values': [1e-3, 1e-4]
                    },
                    'optimizer': {
                        'values': ['sgd', 'adam']
                    },
                    'criterion': {
                        'values': ['cross_entropy']  
                    },
                    'use_SAM':{
                        'values' : [True, False]
                    },
                    'pretrained':{
                        'values' : [False]
                    },
                    'transformation':{
                        'values' : ["U"]
                    },
                    'model_name':{
                        'values' : ['efficientnet-b1', 'efficientnet-b0']
                    }
                }
            }
             