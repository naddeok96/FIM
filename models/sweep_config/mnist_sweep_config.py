sweep_config = {
                'method': 'bayes',
                'metric': {
                    'name': 'loss',
                    'goal': 'minimize' 
                },
                'early_terminate' : {
                    "type": 'hyperband',
                    "s": 2,
                    "eta": 3,
                    "max_iter": 27
                },
  
                'parameters': {
                    'epochs': {
                        'values': [0,2]
                    },
                    'batch_size': {
                        'values': [5]
                        # 'min' : 1,
                        # 'max' : 32
                    },
                    'momentum': {
                        'values': [0.9]
                    },
                    'weight_decay': {
                        'values': [1e-5]
                    },
                    'learning_rate': {
                        'values': [0.07112]
                        # 'min' : 1e-6,
                        # 'max' : 1e-1
                    },
                    'optimizer': {
                        'values': ['nesterov']
                    },
                    'scheduler': {
                        'values': [None] # "Cosine Annealing"]
                    },
                    'criterion': {
                        'values': ['cross_entropy']  
                    },
                    'use_SAM':{
                        'values' : [False]
                    },
                    'pretrained':{
                        'values' : [False]
                    },
                    'transformation':{
                        'values' : [None] # ["models/pretrained/U_w_means_0-005174736492335796_n0-0014449692098423839_n0-0010137659264728427_and_stds_1-130435824394226_1-128873586654663_1-1922636032104492_.pt"]
                    },
                    'model_name':{
                        'values' : ["lenet"]
                    }
                }
            }
             