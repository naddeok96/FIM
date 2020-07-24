# # View attack
# num2view = 100
# index = 0
# for _ in range(num2view):
#     image, label, index = data.get_single_image(index = index)
#     _, predicted, adv_predictions = attacker.get_newG_attack(image, label)
#     while ((label != predicted).item()) or (sum(adv_prediction == label for adv_prediction in adv_predictions).item() != 1): # 0 for normal
#         index += 1
#         image, label, index = data.get_single_image(index = index)
#         _, predicted, adv_predictions = attacker.get_newG_attack(image, label)

#     attacker.get_newG_attack(image, label, plot = True)
#     index += 1