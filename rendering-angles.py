import functools
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchviz import make_dot

radius = 35.
cube_center = torch.tensor([20., 20., 20., 1.])

camera_positions = []
for phi in np.linspace(0, math.pi, 4):
    for theta in np.linspace(0, 2 * math.pi, 5):
        phi += math.pi / 4
        theta += math.pi / 4
        x = radius * math.sin(phi) * math.cos(theta)
        y = radius * math.sin(phi) * math.sin(theta)
        z = radius * math.cos(phi)
        x = 0 if abs(x) < 0.0001 else x
        y = 0 if abs(y) < 0.0001 else y
        z = 0 if abs(z) < 0.0001 else z
        camera_positions.append(torch.tensor([x, y, z, 0]))

# print((torch.stack(camera_positions).unique(dim=0)) + cube_center)
# print(torch.tensor(camera_positions))
# print((torch.tensor(camera_positions) + cube_center).unique(dim=0))

radius = 50
cube_center = torch.tensor([20., 20., 35., 1.])
camera_positions = []
for phi in np.linspace(0, 2 * math.pi, 72):
    x = radius * math.cos(phi)
    y = radius * math.sin(phi)
    z = 8
    x = 0 if abs(x) < 0.0001 else x
    y = 0 if abs(y) < 0.0001 else y
    z = 0 if abs(z) < 0.0001 else z
    camera_positions.append(torch.tensor([x, y, z, 0]))

print((torch.stack(camera_positions)))
print((torch.stack(camera_positions)) + cube_center)
