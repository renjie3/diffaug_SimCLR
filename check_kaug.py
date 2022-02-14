# import kornia.augmentation as K
# import torch.nn as nn
import numpy as np
# import torchvision
# import torch
# import matplotlib.pyplot as plt
# import kornia

# transform = nn.Sequential(
#     K.RandomAffine(360),
#     K.ColorJitter(0.2, 0.3, 0.2, 0.3)
# )


# x_rgb: torch.tensor = torchvision.io.read_image('./dog_rgb.png')  # CxHxW / torch.uint8

# x_rgb = x_rgb.unsqueeze(0).float() / 255.0  # BxCxHxW
# x_rgb = transform(x_rgb)
# print(x_rgb.shape)

# fig = plt.figure(figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
# plt.axis("off")

# img_rgb: np.array = kornia.tensor_to_image(x_rgb)
# plt.imshow(img_rgb)
# plt.savefig("aug_dog.png")

def get_scheduler(epoch, whole_epoch):
    batch_num = 10
    schedule_batch_num = batch_num * (epoch - 1) // whole_epoch + 1
    return schedule_batch_num

ans = []
for i in range(1, 1001):
    ans.append(get_scheduler(i, 1000))

print(np.sum(ans))