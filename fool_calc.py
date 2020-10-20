import numpy as np


ossa_accs_on_Unet   = [0.952,  0.8865, 0.7625, 0.409,  0.0353, 0.0629, 0.1092]
fgsm_accs_on_Unet   = [0.952,  0.9044, 0.8227, 0.5617, 0.1134, 0.0223, 0.0077]

def fool_ratio(attack_acc):
    test_acc_vec = attack_acc[0] * np.ones(len(attack_acc))
    print(test_acc_vec)
    return ((test_acc_vec - attack_acc)/attack_acc[0])*100.0

print(fool_ratio(fgsm_accs_on_Unet))