sweep_config = {
                'method': 'grid',
                'metric': {
                'name': 'loss',
                'goal': 'minimize'   
                },
                'parameters': {
                    'epochs': {
                        'values': [1]
                    },
                    'batch_size': {
                        'values': [64]
                    },
                    'momentum': {
                        'values': [0.9]
                    },
                    'weight_decay': {
                        'values': [0.005]
                    },
                    'learning_rate': {
                        'values': [1e-2]
                    },
                    'optimizer': {
                        'values': ['sgd']
                    },
                    'criterion': {
                        'values': ["mse"]
                    }                    
                }
            }