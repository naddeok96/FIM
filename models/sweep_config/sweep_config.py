sweep_config = {
                'method': 'bayes',
                'metric': {
                    'name': 'loss',
                    'goal': 'minimize'   
                },
                'parameters': {
                    'epochs': {
                        'values': [1e2, 5e2, 1e3, 5e3]
                    },
                    'batch_size': {
                        'values': [64, 124, 256]
                    },
                    'momentum': {
                        'values': [0.9]
                    },
                    'weight_decay': {
                        'values': [1e-2, 5e-2, 1e-3, 5e-3]
                    },
                    'learning_rate': {
                        'values': [1e-1, 1e-2, 1e-3]
                    },
                    'optimizer': {
                        'values': ['sgd']
                    },
                    'criterion': {
                        'values': ["mse"]
                    }                    
                }
            }