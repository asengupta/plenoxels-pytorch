import functools
import math
import random
import matplotlib.pyplot as plt
from matplotlib import use as mpl_use
import numpy as np
import torch
import torch.nn as nn
import pathos.multiprocessing as mp
import torch.multiprocessing as tmp
from timeit import default_timer as timer
from torchvision.utils import save_image
from torchvision import datasets, transforms
from torchviz import make_dot
import os
from memory_profiler import profile


# import logging as log


class Log:
    def info(self, msg):
        print(msg)

    def warning(self, msg):
        print(msg)


log = Log()


def white_rgb():
    return torch.tensor([1., 1., 1.])


def black_rgb():
    return torch.tensor([0., 0., 0.])


class Empty:
    # EMPTY_RGB = black_rgb
    # ALL_EMPTY = torch.zeros

    EMPTY_RGB = white_rgb
    ALL_EMPTY = torch.ones


def cube_training_positions():
    return torch.tensor([[-4.7487, 44.7487, 20.0000, 1.0000],
                         [-3.9054, -3.9054, 29.0587, 1.0000],
                         [-1.4330, 41.4330, 2.5000, 1.0000],
                         [-1.4330, 41.4330, 37.5000, 1.0000],
                         [2.5000, 2.5000, -4.7487, 1.0000],
                         [7.6256, 32.3744, -10.3109, 1.0000],
                         [13.5946, 13.5946, 53.8074, 1.0000],
                         [20.0000, 20.0000, -15.0000, 1.0000],
                         [20.0000, 20.0000, 55.0000, 1.0000],
                         [26.4054, 26.4054, -13.8074, 1.0000],
                         [32.3744, 7.6256, -10.3109, 1.0000],
                         [37.5000, 37.5000, 44.7487, 1.0000],
                         [43.9054, 43.9054, 10.9413, 1.0000],
                         [44.7487, -4.7487, 20.0000, 1.0000]])


def table_training_positions():
    return torch.tensor([[70.0000, 20.0000, 43.0000, 1.0000],
                         [69.8043, 24.4190, 43.0000, 1.0000],
                         [69.2189, 28.8034, 43.0000, 1.0000],
                         [68.2482, 33.1189, 43.0000, 1.0000],
                         [66.9000, 37.3318, 43.0000, 1.0000],
                         [65.1847, 41.4090, 43.0000, 1.0000],
                         [63.1157, 45.3186, 43.0000, 1.0000],
                         [60.7093, 49.0301, 43.0000, 1.0000],
                         [57.9844, 52.5144, 43.0000, 1.0000],
                         [54.9621, 55.7443, 43.0000, 1.0000],
                         [51.6662, 58.6943, 43.0000, 1.0000],
                         [48.1225, 61.3416, 43.0000, 1.0000],
                         [44.3586, 63.6653, 43.0000, 1.0000],
                         [40.4042, 65.6472, 43.0000, 1.0000],
                         [36.2900, 67.2719, 43.0000, 1.0000],
                         [32.0484, 68.5267, 43.0000, 1.0000],
                         [27.7124, 69.4016, 43.0000, 1.0000],
                         [23.3161, 69.8899, 43.0000, 1.0000],
                         [18.8939, 69.9878, 43.0000, 1.0000],
                         [14.4803, 69.6944, 43.0000, 1.0000],
                         [10.1099, 69.0121, 43.0000, 1.0000],
                         [5.8169, 67.9462, 43.0000, 1.0000],
                         [1.6349, 66.5051, 43.0000, 1.0000],
                         [-2.4033, 64.7000, 43.0000, 1.0000],
                         [-6.2663, 62.5451, 43.0000, 1.0000],
                         [-9.9236, 60.0572, 43.0000, 1.0000],
                         [-13.3468, 57.2558, 43.0000, 1.0000],
                         [-16.5090, 54.1628, 43.0000, 1.0000],
                         [-19.3854, 50.8024, 43.0000, 1.0000],
                         [-21.9536, 47.2010, 43.0000, 1.0000],
                         [-24.1935, 43.3867, 43.0000, 1.0000],
                         [-26.0874, 39.3894, 43.0000, 1.0000],
                         [-27.6207, 35.2403, 43.0000, 1.0000],
                         [-28.7813, 30.9719, 43.0000, 1.0000],
                         [-29.5601, 26.6177, 43.0000, 1.0000],
                         [-29.9511, 22.2117, 43.0000, 1.0000],
                         [-29.9511, 17.7883, 43.0000, 1.0000],
                         [-29.5601, 13.3823, 43.0000, 1.0000],
                         [-28.7813, 9.0281, 43.0000, 1.0000],
                         [-27.6207, 4.7597, 43.0000, 1.0000],
                         [-26.0874, 0.6106, 43.0000, 1.0000],
                         [-24.1935, -3.3867, 43.0000, 1.0000],
                         [-21.9536, -7.2010, 43.0000, 1.0000],
                         [-19.3854, -10.8024, 43.0000, 1.0000],
                         [-16.5090, -14.1628, 43.0000, 1.0000],
                         [-13.3468, -17.2558, 43.0000, 1.0000],
                         [-9.9236, -20.0572, 43.0000, 1.0000],
                         [-6.2663, -22.5451, 43.0000, 1.0000],
                         [-2.4033, -24.7000, 43.0000, 1.0000],
                         [1.6349, -26.5051, 43.0000, 1.0000],
                         [5.8169, -27.9462, 43.0000, 1.0000],
                         [10.1099, -29.0121, 43.0000, 1.0000],
                         [14.4803, -29.6944, 43.0000, 1.0000],
                         [18.8939, -29.9878, 43.0000, 1.0000],
                         [23.3161, -29.8899, 43.0000, 1.0000],
                         [27.7124, -29.4016, 43.0000, 1.0000],
                         [32.0484, -28.5267, 43.0000, 1.0000],
                         [36.2900, -27.2719, 43.0000, 1.0000],
                         [40.4042, -25.6472, 43.0000, 1.0000],
                         [44.3586, -23.6653, 43.0000, 1.0000],
                         [48.1225, -21.3416, 43.0000, 1.0000],
                         [51.6662, -18.6943, 43.0000, 1.0000],
                         [54.9621, -15.7443, 43.0000, 1.0000],
                         [57.9844, -12.5144, 43.0000, 1.0000],
                         [60.7093, -9.0301, 43.0000, 1.0000],
                         [63.1157, -5.3186, 43.0000, 1.0000],
                         [65.1847, -1.4090, 43.0000, 1.0000],
                         [66.9000, 2.6682, 43.0000, 1.0000],
                         [68.2482, 6.8811, 43.0000, 1.0000],
                         [69.2189, 11.1966, 43.0000, 1.0000],
                         [69.8043, 15.5810, 43.0000, 1.0000],
                         [70.0000, 20.0000, 43.0000, 1.0000]])


log.info(f"Using backend {plt.get_backend()}")

GRID_X = 40
GRID_Y = 40
GRID_Z = 50

INHOMOGENEOUS_ZERO_VECTOR = torch.tensor([0., 0., 0.])
REGULARISATION_FRACTION = 0.01
TV_REGULARISATION_LAMBDA = 0.001
CAUCHY_REGULARISATION_LAMBDA = 0.001
LEARNING_RATE = 0.0005
NUM_STOCHASTIC_RAYS = 1500
ARBITRARY_SCALE = 5

MASTER_RAY_SAMPLE_POSITIONS_STRUCTURE = []
MASTER_VOXELS_STRUCTURE = []
VOXELS_NOT_USED = 0
OUTPUT_FOLDER = "./output"


class Camera:
    def __init__(self, focal_length, center, look_at):
        self.center = center
        self.look_at = look_at
        self.basis = basis_from_depth(look_at, center)
        self.focal_length = focal_length
        camera_center = center.detach().clone()
        transposed_basis = torch.transpose(self.basis, 0, 1)
        camera_center[:3] = camera_center[
                            :3] * -1  # We don't want to multiply the homogenous coordinate component; it needs to remain 1
        camera_origin_translation = torch.eye(4, 4)
        camera_origin_translation[:, 3] = camera_center
        extrinsic_camera_parameters = torch.matmul(torch.inverse(transposed_basis), camera_origin_translation)
        intrinsic_camera_parameters = torch.tensor([[focal_length, 0., 0., 0.],
                                                    [0., focal_length, 0., 0.],
                                                    [0., 0., 1., 0.]])
        self.transform = torch.matmul(intrinsic_camera_parameters, extrinsic_camera_parameters)

    def to_2D(self, point):
        rendered_point = torch.matmul(self.transform, torch.transpose(point, 0, 1))
        point_z = rendered_point[2, 0]
        return rendered_point / point_z

    def viewing_angle(self):
        camera_basis_z = self.basis[2][:3]
        camera_basis_theta = math.atan(camera_basis_z[1] / camera_basis_z[0]) if (
                camera_basis_z[0] != 0) else math.pi / 2
        camera_basis_phi = math.atan((camera_basis_z[0] ** 2 + camera_basis_z[1] ** 2) / camera_basis_z[2]) if (
                camera_basis_z[2] != 0) else math.pi / 2
        return torch.tensor([camera_basis_theta, camera_basis_phi])


def camera_basis_from(camera_depth_z_vector):
    depth_vector = camera_depth_z_vector[:3]  # We just want the inhomogenous parts of the coordinates

    # This calculates the projection of the world z-axis onto the surface defined by the camera direction,
    # since we want to derive the coordinate system of the camera to be orthogonal without having
    # to calculate it manually.
    cartesian_z_vector = torch.tensor([0., 0., 1.])
    cartesian_z_projection_lambda = torch.dot(depth_vector, cartesian_z_vector) / torch.dot(
        depth_vector, depth_vector)
    camera_up_vector = cartesian_z_vector - cartesian_z_projection_lambda * depth_vector

    # This special case is for when the camera is directly pointing up or down, then
    # there is no way to decide which way to orient its up vector in the X-Y plane.
    # We choose to align the up veector with the X-axis in this case.
    if (torch.equal(camera_up_vector, INHOMOGENEOUS_ZERO_VECTOR)):
        camera_up_vector = torch.tensor([1., 0., 0.])
    log.info(f"Up vector is: {camera_up_vector}")
    # The camera coordinate system now has the direction of camera and the up direction of the camera.
    # We need to find the third vector which needs to be orthogonal to both the previous vectors.
    # Taking the cross product of these vectors gives us this third component
    camera_x_vector = torch.linalg.cross(depth_vector, camera_up_vector)
    inhomogeneous_basis = torch.stack([camera_x_vector, camera_up_vector, depth_vector, torch.tensor([0., 0., 0.])])
    homogeneous_basis = torch.hstack((inhomogeneous_basis, torch.tensor([[0.], [0.], [0.], [1.]])))
    homogeneous_basis[0] = unit_vector(homogeneous_basis[0])
    homogeneous_basis[1] = unit_vector(homogeneous_basis[1])
    homogeneous_basis[2] = unit_vector(homogeneous_basis[2])
    return homogeneous_basis


def basis_from_depth(look_at, camera_center):
    log.info(f"Looking at: {look_at}")
    log.info(f"Looking from: {camera_center}")
    depth_vector = torch.sub(look_at, camera_center)
    depth_vector[3] = 1.
    return camera_basis_from(depth_vector)


def unit_vector(camera_basis_vector):
    return camera_basis_vector / math.sqrt(
        pow(camera_basis_vector[0], 2) +
        pow(camera_basis_vector[1], 2) +
        pow(camera_basis_vector[2], 2))


def generate_camera_angles(radius, look_at):
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

    positions = (torch.stack(camera_positions).unique(dim=0)) + look_at
    log.info(positions)
    return positions


HALF_SQRT_3_BY_PI = 0.5 * math.sqrt(3. / math.pi)
HALF_SQRT_15_BY_PI = 0.5 * math.sqrt(15. / math.pi)
QUARTER_SQRT_15_BY_PI = 0.25 * math.sqrt(15. / math.pi)
QUARTER_SQRT_5_BY_PI = 0.25 * math.sqrt(5. / math.pi)

Y_0_0 = 0.5 * math.sqrt(1. / math.pi)
Y_m1_1 = lambda theta, phi: HALF_SQRT_3_BY_PI * math.sin(theta) * math.sin(phi)
Y_0_1 = lambda theta, phi: HALF_SQRT_3_BY_PI * math.cos(theta)
Y_1_1 = lambda theta, phi: HALF_SQRT_3_BY_PI * math.sin(theta) * math.cos(phi)
Y_m2_2 = lambda theta, phi: HALF_SQRT_15_BY_PI * math.sin(theta) * math.cos(phi) * math.sin(
    theta) * math.sin(phi)
Y_m1_2 = lambda theta, phi: HALF_SQRT_15_BY_PI * math.sin(theta) * math.sin(phi) * math.cos(theta)
Y_0_2 = lambda theta, phi: QUARTER_SQRT_5_BY_PI * (3 * math.cos(theta) * math.cos(theta) - 1)
Y_1_2 = lambda theta, phi: HALF_SQRT_15_BY_PI * math.sin(theta) * math.cos(phi) * math.cos(theta)
Y_2_2 = lambda theta, phi: QUARTER_SQRT_15_BY_PI * (
        pow(math.sin(theta) * math.cos(phi), 2) - pow(math.sin(theta) * math.sin(phi), 2))


def harmonic(C_0_0, C_m1_1, C_0_1, C_1_1, C_m2_2, C_m1_2, C_0_2, C_1_2, C_2_2):
    return lambda theta, phi: C_0_0 * Y_0_0 + C_m1_1 * Y_m1_1(theta, phi) + C_0_1 * Y_0_1(theta, phi) + C_1_1 * Y_1_1(
        theta, phi) + C_m2_2 * Y_m2_2(theta, phi) + C_m1_2 * Y_m1_2(theta, phi) + C_0_2 * Y_0_2(theta,
                                                                                                phi) + C_1_2 * Y_1_2(
        theta, phi) + C_2_2 * Y_2_2(theta, phi)


def rgb_harmonics(rgb_harmonic_coefficients):
    red_harmonic = harmonic(*rgb_harmonic_coefficients[:9])
    green_harmonic = harmonic(*rgb_harmonic_coefficients[9:18])
    blue_harmonic = harmonic(*rgb_harmonic_coefficients[18:])
    return (red_harmonic, green_harmonic, blue_harmonic)


class Voxel:
    NUM_INTERPOLATING_VOXEL_NEIGHBOURS = 8
    DEFAULT_OPACITY = 0.05
    VOXEL_PRUNING_OPACITY_THRESHOLD = 0.2
    NUM_VOXEL_NEIGHBOURS = 9 * 3 - 1
    VOXEL_PRUNING_NEIGHBOUR_OPACITY_THRESHOLDS = torch.full([NUM_VOXEL_NEIGHBOURS], VOXEL_PRUNING_OPACITY_THRESHOLD)

    @staticmethod
    def default_voxel(requires_grad=True):
        return lambda: torch.tensor(Voxel.uniform_harmonic(), requires_grad=requires_grad)

    @staticmethod
    def random_coloured_voxel(requires_grad=True):
        return lambda: torch.tensor(
            np.concatenate(([0.7], (np.random.rand(VoxelGrid.VOXEL_DIMENSION - 1) - 0.5))),
            requires_grad=requires_grad)

    @staticmethod
    def uniform_harmonic(density=1.):
        return [density] + ([0.5] + [0.] * (VoxelGrid.PER_CHANNEL_DIMENSION - 1)) * 3

    @staticmethod
    def random_harmonic_coefficient_set():
        return ([random.random()] + [0.] * (VoxelGrid.PER_CHANNEL_DIMENSION - 1))

    @staticmethod
    def uniform_harmonic_random_colour(density=0.04, requires_grad=True):
        return lambda: torch.tensor([
                                        density] + Voxel.random_harmonic_coefficient_set() + Voxel.random_harmonic_coefficient_set() + Voxel.random_harmonic_coefficient_set(),
                                    requires_grad=requires_grad)

    @staticmethod
    def occupied_voxel(density=1., requires_grad=True):
        return lambda: torch.tensor(Voxel.uniform_harmonic(density), requires_grad=requires_grad)

    @staticmethod
    def empty_voxel(requires_grad=True):
        return lambda: torch.zeros([VoxelGrid.VOXEL_DIMENSION], requires_grad=requires_grad)

    @staticmethod
    def like_voxel(prototype_voxel):
        return lambda: prototype_voxel.clone()

    @staticmethod
    def prune(voxel_tensor):
        voxel_tensor.requires_grad = False
        voxel_tensor.fill_(0.)
        voxel_tensor.pruned = True

    @staticmethod
    def is_pruned(voxel_tensor):
        return voxel_tensor.pruned if hasattr(voxel_tensor, "pruned") else False


class Ray:
    def __init__(self, num_samples, view_point, ray_sample_positions, voxel_positions, voxels):
        self.num_samples = num_samples
        self.ray_sample_positions = ray_sample_positions
        self.view_point = view_point
        self.voxels = voxels
        self.voxel_positions = voxel_positions
        if num_samples != len(ray_sample_positions):
            log.warning(f"WARNING: num_samples = {num_samples}, sample_positions = {ray_sample_positions}")

    def at(self, index):
        start = index * Voxel.NUM_INTERPOLATING_VOXEL_NEIGHBOURS
        end = start + Voxel.NUM_INTERPOLATING_VOXEL_NEIGHBOURS
        return self.ray_sample_positions[index], \
               self.voxel_positions[start: end], \
               self.voxels[start: end]


class VoxelAccess:
    def __init__(self, view_points, ray_sample_positions, voxel_pointers, all_voxels, all_voxel_positions):
        self.ray_sample_positions = ray_sample_positions
        self.view_points = view_points
        self.voxel_positions = all_voxel_positions
        self.all_voxels = all_voxels
        self.voxel_pointers = voxel_pointers

    def for_ray(self, ray_index):
        ptr = self.voxel_pointers[ray_index]
        start, end, num_samples = ptr
        return Ray(num_samples, self.view_points[ray_index],
                   self.ray_sample_positions[int(start / Voxel.NUM_INTERPOLATING_VOXEL_NEIGHBOURS): int(
                       end / Voxel.NUM_INTERPOLATING_VOXEL_NEIGHBOURS)],
                   self.voxel_positions[start:end],
                   self.all_voxels[start:end])


def cube_faces(cube_spec):
    x, y, z, dx, dy, dz = cube_spec
    face1 = (x, y, z, dx + 1, dy + 1, 1)
    face2 = (x, y, z + dz, dx + 1, dy + 1, 1)
    face3 = (x, y, z, 1, dy + 1, dz + 1)
    face4 = (x + dx, y, z, 1, dy + 1, dz + 1)
    face5 = (x, y, z, dx + 1, 1, dz + 1)
    face6 = (x, y + dy, z, dx + 1, 1, dz + 1)
    return face1, face2, face3, face4, face5, face6


class VoxelGrid:
    VOXEL_DIMENSION = 28
    PER_CHANNEL_DIMENSION = 9
    DEFAULT_SCALE = torch.tensor([1., 1., 1.])

    def __init__(self, world_tensor, scale=DEFAULT_SCALE):
        self.scale = scale
        self.grid_x, self.grid_y, self.grid_z = torch.tensor(world_tensor.shape) * self.scale
        self.voxel_grid_x, self.voxel_grid_y, self.voxel_grid_z = world_tensor.shape
        self.voxel_grid = world_tensor

    def voxel_dimensions(self):
        return torch.tensor([self.voxel_grid_x, self.voxel_grid_y, self.voxel_grid_z]).int()

    def world_x(self):
        return self.grid_x

    def world_y(self):
        return self.grid_y

    def world_z(self):
        return self.grid_z

    @staticmethod
    def build_empty_world(x, y, z, scale=DEFAULT_SCALE):
        return VoxelGrid.new(x, y, z, Voxel.empty_voxel(), scale)

    @staticmethod
    def build_random_world(x, y, z, scale=DEFAULT_SCALE):
        return VoxelGrid.new(x, y, z, Voxel.random_coloured_voxel(), scale)

    @staticmethod
    def build_with_voxel(x, y, z, prototype_voxel, scale=DEFAULT_SCALE):
        return VoxelGrid.new(x, y, z, Voxel.like_voxel(prototype_voxel), scale)

    @staticmethod
    def copy_from(world, scale=DEFAULT_SCALE):
        return VoxelGrid(world.voxel_grid.copy(), scale=world.scale)

    @classmethod
    def from_tensor(cls, world_tensor, scale=DEFAULT_SCALE):
        return cls(world_tensor, scale)

    @classmethod
    def as_parameter(cls, world, model):
        world_tensor = world.voxel_grid
        voxel_x, voxel_y, voxel_z = world.voxel_dimensions()
        new_world = VoxelGrid.build_empty_world(voxel_x, voxel_y, voxel_z, scale=world.scale)
        for i, j, k, v in world.all_voxels():
            parameter = nn.Parameter(v)
            if Voxel.is_pruned(v):
                Voxel.prune(parameter)
            new_world.set((i, j, k), parameter)
            model.register_parameter(f"{(i, j, k)}", parameter)
        return new_world

    @classmethod
    def new(cls, voxel_x, voxel_y, voxel_z, make_voxel, scale=DEFAULT_SCALE):
        log.info(f"Initialising world with dimensions ({voxel_x, voxel_y, voxel_z})")
        voxel_grid = np.ndarray((voxel_x, voxel_y, voxel_z), dtype=list)
        for i in range(voxel_x):
            for j in range(voxel_y):
                for k in range(voxel_z):
                    voxel_grid[i, j, k] = make_voxel()
        return cls(voxel_grid, scale)

    def at(self, world_x, world_y, world_z):
        if self.is_outside(world_x, world_y, world_z):
            return Voxel.empty_voxel(requires_grad=False)()
        else:
            voxel_x, voxel_y, voxel_z = self.to_voxel_coordinates(torch.tensor([world_x, world_y, world_z]))
            return self.voxel_grid[voxel_x, voxel_y, voxel_z]

    # TODO: Remember to test scale_up() again
    def scale_up(self):
        new_dimensions = self.voxel_dimensions() * 2
        new_scale = self.scale / 2
        x_scale, y_scale, z_scale = new_scale
        log.info(f"New scaled up dimensions={new_dimensions}")
        scaled_up_world = VoxelGrid.build_empty_world(new_dimensions[0], new_dimensions[1], new_dimensions[2],
                                                      scale=new_scale)
        for i, j, k, original_voxel in self.voxels_in_world(
                torch.tensor([0, 0, 0, self.grid_x, self.grid_y, self.grid_z])):
            x2 = (i * 2 + 1 / x_scale).int()
            y2 = (j * 2 + 1 / y_scale).int()
            z2 = (k * 2 + 1 / z_scale).int()
            scaled_up_world.voxel_grid[i * 2: x2, j * 2: y2, k * 2: z2].fill(original_voxel.detach().clone())
            if hasattr(original_voxel, "pruned"):
                for x, y, z, v in scaled_up_world.voxel_by_position(
                        torch.tensor([i * 2, j * 2, k * 2, 1 / x_scale, 1 / y_scale, 1 / z_scale])):
                    Voxel.prune(v)
        return scaled_up_world

    def to_voxel_coordinates(self, world_coordinates):
        return torch.divide(world_coordinates, self.scale).int()

    def set(self, voxel_position, voxel):
        voxel_x, voxel_y, voxel_z = voxel_position
        if self.is_outside_grid(voxel_x, voxel_y, voxel_z):
            log.warning(f"[WARNING]: set() attempted to set a value at {(voxel_position)} outside grid")
            return
        else:
            self.voxel_grid[voxel_x, voxel_y, voxel_z] = voxel

    def is_inside_grid(self, voxel_x, voxel_y, voxel_z):
        return (0 <= voxel_x < self.voxel_grid_x and
                0 <= voxel_y < self.voxel_grid_y and
                0 <= voxel_z < self.voxel_grid_z)

    def is_outside_grid(self, voxel_x, voxel_y, voxel_z):
        return not self.is_inside_grid(voxel_x, voxel_y, voxel_z)

    def is_inside(self, world_x, world_y, world_z):
        return (0 <= world_x < self.grid_x and
                0 <= world_y < self.grid_y and
                0 <= world_z < self.grid_z)

    def is_outside(self, world_x, world_y, world_z):
        return not self.is_inside(world_x, world_y, world_z)

    def neighbour_opacities(self, voxel_x, voxel_y, voxel_z):
        opacities = []
        for i in range(voxel_x - 1, voxel_x + 2):
            for j in range(voxel_y - 1, voxel_y + 2):
                for k in range(voxel_z - 1, voxel_z + 2):
                    if voxel_x == i and voxel_y == j and voxel_z == k:
                        continue
                    opacities.append(self.voxel_by_position(i, j, k)[0])
        # print(f"Opacities are {torch.stack(opacities)}")
        return torch.stack(opacities)

    def prune(self, voxel_position):
        voxel = self.voxel_by_position(*voxel_position)
        if (voxel[0] > Voxel.VOXEL_PRUNING_OPACITY_THRESHOLD):
            return False
        surrounding_opacities = self.neighbour_opacities(*voxel_position)
        log.info(f"Scanning neighbours...{surrounding_opacities}")
        if (surrounding_opacities.less_equal(
                Voxel.VOXEL_PRUNING_NEIGHBOUR_OPACITY_THRESHOLDS).all()):
            voxel.requires_grad = False
            voxel.mul_(0)
            Voxel.prune(voxel)
            return True

    def channel_opacity(self, distance_density_color_tensors, viewing_angle):
        number_of_samples = len(distance_density_color_tensors)
        density_distance_products = distance_density_color_tensors[:, 0] * distance_density_color_tensors[:, 1]
        summing_matrix = torch.tensor(list(
            functools.reduce(lambda acc, n: acc + [[1.] * n + [0.] * (number_of_samples - n)],
                             range(1, number_of_samples + 1),
                             [])))
        # print(f"Sigma-D={density_distance_products.type()}, summing matrix = {summing_matrix.t().type()}")
        transmittances = torch.matmul(density_distance_products.double(), summing_matrix.t().double())
        transmittances = torch.exp(-transmittances)

        red_channel, green_channel, blue_channel = [], [], []
        for index, distance_density_color_tensor in enumerate(distance_density_color_tensors):
            red_harmonic, green_harmonic, blue_harmonic = rgb_harmonics(
                distance_density_color_tensor[2:])
            r = red_harmonic(viewing_angle[0], viewing_angle[1])
            g = green_harmonic(viewing_angle[0], viewing_angle[1])
            b = blue_harmonic(viewing_angle[0], viewing_angle[1])
            red_channel.append(r)
            green_channel.append(g)
            blue_channel.append(b)

        red_channel, green_channel, blue_channel = torch.stack(red_channel), torch.stack(green_channel), torch.stack(
            blue_channel)
        base_transmittance_factors = transmittances * (1 - torch.exp(- density_distance_products))
        red = (base_transmittance_factors * red_channel).sum()
        green = (base_transmittance_factors * green_channel).sum()
        blue = (base_transmittance_factors * blue_channel).sum()

        color_densities = torch.stack([red, green, blue])
        return color_densities

    def to_voxel_cube_spec(self, world_cube_spec):
        x1, y1, z1, dx, dy, dz = world_cube_spec
        x2, y2, z2 = x1 + dx, y1 + dy, z1 + dz
        return self.to_voxel_coordinates(torch.stack([x1, y1, z1])), self.to_voxel_coordinates(
            torch.stack([x2, y2, z2]))

    def voxels_in_world(self, world_cube_spec):
        return self.__voxels(self.to_voxel_cube_spec(world_cube_spec))

    def voxels_by_position(self, voxel_cube_spec):
        x1, y1, z1, dx, dy, dz = voxel_cube_spec
        x2, y2, z2 = x1 + dx, y1 + dy, z1 + dz
        return self.__voxels([[x1, y1, z2], [x2, y2, z2]])

    def __voxels(self, from_to_voxel_positions):
        from_voxel, to_voxel = from_to_voxel_positions
        voxel_x1, voxel_y1, voxel_z1 = from_voxel
        voxel_x2, voxel_y2, voxel_z2 = to_voxel
        for i in torch.arange(voxel_x1, voxel_x2):
            for j in torch.arange(voxel_y1, voxel_y2):
                for k in torch.arange(voxel_z1, voxel_z2):
                    yield int(i), int(j), int(k), self.voxel_by_position(int(i), int(j), int(k))

    def all_voxels(self):
        return self.__voxels(torch.tensor([[0., 0., 0.], [self.voxel_grid_x, self.voxel_grid_y, self.voxel_grid_z]]))

    def build_solid_cube(self, cube_spec):
        for i, j, k, _ in self.voxels_in_world(cube_spec):
            self.voxel_grid[i, j, k] = Voxel.occupied_voxel(0.2)()

    def build_monochrome_hollow_cube(self, cube_spec):
        self.build_hollow_cube_with_randomly_coloured_sides(Voxel.default_voxel(), cube_spec)

    def build_hollow_cube_with_randomly_coloured_sides(self, make_voxel, cube_spec):
        voxel_1, voxel_2, voxel_3, voxel_4, voxel_5, voxel_6 = make_voxel(), make_voxel(), make_voxel(), make_voxel(), make_voxel(), make_voxel()
        face1, face2, face3, face4, face5, face6 = cube_faces(cube_spec)

        for i, j, k, _ in self.voxels_in_world(face1):
            self.voxel_grid[i, j, k] = voxel_1
        for i, j, k, _ in self.voxels_in_world(face2):
            self.voxel_grid[i, j, k] = voxel_2
        for i, j, k, _ in self.voxels_in_world(face3):
            self.voxel_grid[i, j, k] = voxel_3
        for i, j, k, _ in self.voxels_in_world(face4):
            self.voxel_grid[i, j, k] = voxel_4
        for i, j, k, _ in self.voxels_in_world(face5):
            self.voxel_grid[i, j, k] = voxel_5
        for i, j, k, _ in self.voxels_in_world(face6):
            self.voxel_grid[i, j, k] = voxel_6

    def density(self, ray_samples_with_positions_distances, viewing_angle):
        global MASTER_VOXELS_STRUCTURE
        collected_intensities = []
        for ray_sample in ray_samples_with_positions_distances:
            ray_sample_world_position = ray_sample[:3]
            collected_intensities.append(
                self.intensities(ray_sample_world_position, self.interpolating_neighbours(ray_sample_world_position)))
        return self.channel_opacity(
            torch.cat([ray_samples_with_positions_distances[:, 3:], torch.stack(collected_intensities)], 1),
            viewing_angle)

    def density_split(self, ray_sample_distances, ray, viewing_angle):
        collected_intensities = []
        for index, distance in enumerate(ray_sample_distances):
            ray_sample_world_position, voxel_positions, voxels = ray.at(index)
            if len(voxels) == 0:
                return Empty.EMPTY_RGB()
            collected_intensities.append(self.intensities(ray_sample_world_position, voxels))

        return self.channel_opacity(torch.cat([ray_sample_distances, torch.stack(collected_intensities)], 1),
                                    viewing_angle)

    def interpolating_neighbours(self, ray_sample_world_position):
        x_0, x_1, y_0, y_1, z_0, z_1, _1, _2, _3 = self.interpolating_neighbour_endpoints(ray_sample_world_position)
        c_000 = self.voxel_by_position(x_0, y_0, z_0)
        c_001 = self.voxel_by_position(x_0, y_0, z_1)
        c_010 = self.voxel_by_position(x_0, y_1, z_0)
        c_011 = self.voxel_by_position(x_0, y_1, z_1)
        c_100 = self.voxel_by_position(x_1, y_0, z_0)
        c_101 = self.voxel_by_position(x_1, y_0, z_1)
        c_110 = self.voxel_by_position(x_1, y_1, z_0)
        c_111 = self.voxel_by_position(x_1, y_1, z_1)

        return (c_000, c_001, c_010, c_011, c_100, c_101, c_110, c_111)

    def interpolating_neighbour_endpoints(self, ray_sample_world_coords):
        # print(f"Scale is {self.scale}")
        x, y, z = ray_sample_world_coords
        voxel_x, voxel_y, voxel_z = self.to_voxel_coordinates(ray_sample_world_coords)
        x_0, x_1 = voxel_x, voxel_x + 1
        y_0, y_1 = voxel_y, voxel_y + 1
        z_0, z_1 = voxel_z, voxel_z + 1
        x_d = (x / self.scale[0] - x_0) / (x_1 - x_0)
        y_d = (y / self.scale[1] - y_0) / (y_1 - y_0)
        z_d = (z / self.scale[2] - z_0) / (z_1 - z_0)
        return x_0, x_1, y_0, y_1, z_0, z_1, x_d, y_d, z_d

    def intensities(self, ray_sample_world_position, interpolating_neighbours):
        global MASTER_VOXELS_STRUCTURE
        _, _, _, _, _, _, x_d, y_d, z_d = self.interpolating_neighbour_endpoints(ray_sample_world_position)
        c_000, c_001, c_010, c_011, c_100, c_101, c_110, c_111 = interpolating_neighbours

        c_00 = c_000 * (1 - x_d) + c_100 * x_d
        c_01 = c_001 * (1 - x_d) + c_101 * x_d
        c_10 = c_010 * (1 - x_d) + c_110 * x_d
        c_11 = c_011 * (1 - x_d) + c_111 * x_d
        c_0 = c_00 * (1 - y_d) + c_10 * y_d
        c_1 = c_01 * (1 - y_d) + c_11 * y_d
        c = c_0 * (1 - z_d) + c_1 * z_d
        MASTER_VOXELS_STRUCTURE += [c_000, c_001, c_010, c_011, c_100, c_101, c_110, c_111]
        if (c[0].abs() > 900):
            log.warning(f"WARNING: Bad neighbouring tensor at {(ray_sample_world_position)}")
            log.warning(f"WARNING: {(x_d, y_d, z_d)}")
            log.warning(interpolating_neighbours)
        return c

    def neighbours(self, world_x, world_y, world_z):
        x_0, x_1, y_0, y_1, z_0, z_1, _1, _2, _3 = self.interpolating_neighbour_endpoints(
            torch.tensor([world_x, world_y, world_z]))
        c_000 = self.voxel_by_position(x_0, y_0, z_0)
        c_001 = self.voxel_by_position(x_0, y_0, z_1)
        c_010 = self.voxel_by_position(x_0, y_1, z_0)
        c_011 = self.voxel_by_position(x_0, y_1, z_1)
        c_100 = self.voxel_by_position(x_1, y_0, z_0)
        c_101 = self.voxel_by_position(x_1, y_0, z_1)
        c_110 = self.voxel_by_position(x_1, y_1, z_0)
        c_111 = self.voxel_by_position(x_1, y_1, z_1)

        return ([c_000, c_001, c_010, c_011, c_100, c_101, c_110, c_111], torch.tensor([[x_0, y_0, z_0],
                                                                                        [x_0, y_0, z_1],
                                                                                        [x_0, y_1, z_0],
                                                                                        [x_0, y_1, z_1],
                                                                                        [x_1, y_0, z_0],
                                                                                        [x_1, y_0, z_1],
                                                                                        [x_1, y_1, z_0],
                                                                                        [x_1, y_1, z_1]]))

    def voxel_by_position(self, voxel_x, voxel_y, voxel_z):
        if (voxel_x < 0 or voxel_x >= self.voxel_grid_x or
                voxel_y < 0 or voxel_y >= self.voxel_grid_y or
                voxel_z < 0 or voxel_z >= self.voxel_grid_z):
            return torch.zeros(VoxelGrid.VOXEL_DIMENSION)
        return self.voxel_grid[voxel_x, voxel_y, voxel_z]


class ClampingFunctions:
    SIGMOID = nn.Sigmoid()
    CLAMP = lambda t: torch.clamp(t, min=0, max=1)
    DEFAULT = CLAMP


class Renderer:
    def __init__(self, world, camera, view_spec, ray_spec):
        self.world = world
        self.camera = camera
        self.ray_length = ray_spec[0]
        self.num_ray_samples = ray_spec[1]
        self.x_1, self.x_2 = view_spec[0], view_spec[1]
        self.y_1, self.y_2 = view_spec[2], view_spec[3]
        self.num_view_samples_x = view_spec[4]
        self.num_view_samples_y = view_spec[5]

    def render_from_ray(self, ray, viewing_angle, clamping_function):
        # print(f"Wall clock in render_from_ray() is {timer()}")
        ray_sample_positions = ray.ray_sample_positions
        unique_ray_samples = ray_sample_positions
        view_x, view_y = ray.view_point

        if (len(unique_ray_samples) <= 1):
            return torch.tensor([view_x, view_y, 0., 0., 0.])

        t1 = unique_ray_samples[:-1]
        t2 = unique_ray_samples[1:]
        consecutive_sample_distances = (t1 - t2).pow(2).sum(1).sqrt()

        # Make 1D tensor into 2D tensor
        # List of tensors, each entry is distance from i-th sample to the next sample
        ray_sample_distances = torch.reshape(consecutive_sample_distances, (-1, 1))
        color_densities = self.world.density_split(ray_sample_distances, ray, viewing_angle)
        color_tensor = clamping_function(color_densities * ARBITRARY_SCALE)

        if (view_x < self.x_1 or view_x > self.x_2
                or view_y < self.y_1 or view_y > self.y_2):
            log.warning(f"[WARNING]: bad generation: {view_x}, {view_y}")

        # print(color_tensor)
        return torch.cat([torch.tensor([view_x, view_y]), color_tensor])

    def render_from_rays(self, voxel_access, clamping_function=ClampingFunctions.DEFAULT):
        X, Y = 0, 1
        RED_CHANNEL, GREEN_CHANNEL, BLUE_CHANNEL = 2, 3, 4
        camera = self.camera
        # composite_colour_tensors = self.render_parallel(voxel_access, camera)
        composite_colour_tensors = self.render_serial(voxel_access, camera, clamping_function)
        red_channel = composite_colour_tensors[:, [X, Y, RED_CHANNEL]]
        green_channel = composite_colour_tensors[:, [X, Y, GREEN_CHANNEL]]
        blue_channel = composite_colour_tensors[:, [X, Y, BLUE_CHANNEL]]
        log.info("Done volumetric calculations from rays!!")
        return (red_channel, green_channel, blue_channel)

    def render_serial(self, voxel_access, camera, clamping_function):
        viewing_angle = camera.viewing_angle()
        num_view_points = len(voxel_access.view_points)
        composite_colour_tensors = torch.stack(list(
            map(lambda index: self.render_from_ray(voxel_access.for_ray(index), viewing_angle, clamping_function),
                range(num_view_points))))
        return composite_colour_tensors

    def render_from_angle(self, ray):
        return self.render_from_ray(ray, self.camera.viewing_angle(), clamping_function=ClampingFunctions.DEFAULT)

    def render_parallel(self, voxel_access, camera):
        viewing_angle = camera.viewing_angle()
        num_view_points = len(voxel_access.view_points)
        workers = os.cpu_count()
        p = tmp.Pool(workers)
        start_copy_rays = timer()
        rays = list(map(lambda i: voxel_access.for_ray(i), range(num_view_points)))
        end_copy_rays = timer()
        log.info(f"Copying rays took {end_copy_rays - start_copy_rays}")
        log.info(f"Wall clock is {timer()}")
        start_render_rays = timer()
        # responses = p.map(lambda ray: self.render_from_ray(ray, viewing_angle), rays)
        responses = p.map(self.render_from_angle, rays)
        p.close()
        p.join()
        end_render_rays = timer()
        log.info(f"Actual rendering took {end_render_rays - start_render_rays}")
        composite_colour_tensors = torch.stack(list(responses))
        return composite_colour_tensors

    @staticmethod
    def initialise_plt(plt):
        plt.rcParams['axes.xmargin'] = 0
        plt.rcParams['axes.ymargin'] = 0
        figure = plt.figure(f"{random.random()}", frameon=False)
        plt.rcParams['axes.facecolor'] = 'black'
        plt.axis("equal")
        plt.style.use("dark_background")
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.axis("off")
        return figure

    def build_rays(self, ray_intersection_weights):
        camera = self.camera
        camera_basis_x = camera.basis[0][:3]
        camera_basis_y = camera.basis[1][:3]
        camera_basis_z = camera.basis[2][:3]
        camera_center_inhomogenous = camera.center[:3]
        all_voxel_positions = []
        view_points = []
        voxel_pointers = []
        all_voxels = []
        ray_sample_positions = []
        view_screen_origin = camera_basis_z * camera.focal_length + camera_center_inhomogenous
        counter = 0
        for ray_intersection_weight in ray_intersection_weights:
            ray_screen_intersection = camera_basis_x * ray_intersection_weight[0] + \
                                      camera_basis_y * ray_intersection_weight[1] + view_screen_origin
            unit_ray = unit_vector(ray_screen_intersection - camera_center_inhomogenous)
            view_x, view_y = ray_intersection_weight[0], ray_intersection_weight[1]
            num_intersecting_voxels = 0

            all_voxels_per_ray = []
            all_voxel_positions_per_ray = []
            ray_sample_positions_per_ray = []
            for k in np.linspace(0, self.ray_length, int(self.num_ray_samples)):
                ray_endpoint = camera_center_inhomogenous + unit_ray * k
                ray_x, ray_y, ray_z = ray_endpoint
                if (self.world.is_outside(ray_x, ray_y, ray_z)):
                    continue
                # We are in the box
                interpolating_voxels, interpolating_voxel_positions = self.world.neighbours(ray_x, ray_y, ray_z)
                num_intersecting_voxels += 1
                all_voxels_per_ray += interpolating_voxels
                all_voxel_positions_per_ray += interpolating_voxel_positions
                ray_sample_positions_per_ray.append(torch.tensor([ray_x, ray_y, ray_z]))
            if (num_intersecting_voxels <= 1):
                continue
            all_voxels += all_voxels_per_ray
            all_voxel_positions += all_voxel_positions_per_ray
            ray_sample_positions += ray_sample_positions_per_ray

            view_points.append((view_x, view_y))
            voxel_pointers.append(
                (counter, counter + Voxel.NUM_INTERPOLATING_VOXEL_NEIGHBOURS * num_intersecting_voxels,
                 num_intersecting_voxels))
            counter += Voxel.NUM_INTERPOLATING_VOXEL_NEIGHBOURS * num_intersecting_voxels

            if (view_x < self.x_1 or view_x > self.x_2
                    or view_y < self.y_1 or view_y > self.y_2):
                log.warning(f"[WARNING]: bad generation: {view_x}, {view_y}")
        log.info("Done building candidate rays!!")

        return VoxelAccess(view_points, torch.stack(ray_sample_positions), voxel_pointers, all_voxels,
                           all_voxel_positions)

    def render(self, plt, clamping_function=ClampingFunctions.DEFAULT, text=None):
        RED_CHANNEL, GREEN_CHANNEL, BLUE_CHANNEL = 0, 1, 2
        global VOXELS_NOT_USED
        global MASTER_RAY_SAMPLE_POSITIONS_STRUCTURE
        red_image = []
        green_image = []
        blue_image = []
        camera = self.camera
        camera_basis_x = camera.basis[0][:3]
        camera_basis_y = camera.basis[1][:3]
        camera_basis_z = camera.basis[2][:3]
        viewing_angle = camera.viewing_angle()
        camera_center_inhomogenous = camera.center[:3]

        Renderer.initialise_plt(plt)
        log.info(f"Camera basis={camera.basis}")
        view_screen_origin = camera_basis_z * camera.focal_length + camera_center_inhomogenous
        log.info(f"View screen origin={view_screen_origin}")
        for i in np.linspace(self.x_1, self.x_2, int(self.num_view_samples_x)):
            red_column = []
            green_column = []
            blue_column = []
            for j in np.linspace(self.y_1, self.y_2, int(self.num_view_samples_y)):
                ray_screen_intersection = camera_basis_x * i + camera_basis_y * j + view_screen_origin
                unit_ray = unit_vector(ray_screen_intersection - camera_center_inhomogenous)
                # print(f"Camera basis is {camera.basis}, Camera center is {camera_center_inhomogenous}, intersection is {ray_screen_intersection}, Unit ray is [{unit_ray}]")
                ray_samples = []
                # To remove artifacts, set ray step samples to be higher, like 200
                for k in np.linspace(0, self.ray_length, int(self.num_ray_samples)):
                    ray_endpoint = camera_center_inhomogenous + unit_ray * k
                    ray_x, ray_y, ray_z = ray_endpoint
                    if (self.world.is_outside(ray_x, ray_y, ray_z)):
                        # print(
                        # f"Skipping [{ray_x},{ray_y},{ray_z}], k={k}, unit ray={unit_ray}, camera is {camera_center_inhomogenous}")
                        continue
                    # We are in the box
                    ray_samples.append([ray_x, ray_y, ray_z])
                    # print(
                    #     f"Sample at ({[ray_x, ray_y, ray_z]}), voxel value here is {self.world.at(ray_x, ray_y, ray_z)}")

                # unique_ray_samples = torch.unique(torch.tensor(ray_samples), dim=0)
                unique_ray_samples = torch.tensor(ray_samples)
                if (len(unique_ray_samples) <= 1):
                    red_column.append(torch.tensor(0))
                    green_column.append(torch.tensor(0))
                    blue_column.append(torch.tensor(0))
                    plt.plot(i, j, marker="o", color=Empty.EMPTY_RGB().detach().numpy())
                    # print("Too few")
                    continue

                MASTER_RAY_SAMPLE_POSITIONS_STRUCTURE += unique_ray_samples
                VOXELS_NOT_USED += 8
                t1 = unique_ray_samples[:-1]
                t2 = unique_ray_samples[1:]
                consecutive_sample_distances = (t1 - t2).pow(2).sum(1).sqrt()
                # print(consecutive_sample_distances)

                # Make 1D tensor into 2D tensor
                ray_samples_with_distances = torch.cat([t1, torch.reshape(consecutive_sample_distances, (-1, 1))], 1)
                # print(ray_samples_with_distances)
                color_densities = self.world.density(ray_samples_with_distances, viewing_angle)

                # color_tensor = torch.clamp(color_densities, min=0, max=1)
                color_tensor = clamping_function(color_densities * ARBITRARY_SCALE)
                # print(color_tensor)
                plt.plot(i, j, marker="o", color=color_tensor.detach().numpy())
                red_column.append(color_tensor[RED_CHANNEL])
                green_column.append(color_tensor[GREEN_CHANNEL])
                blue_column.append(color_tensor[BLUE_CHANNEL])
            red_image.append(torch.tensor(red_column))
            green_image.append(torch.tensor(green_column))
            blue_image.append(torch.tensor(blue_column))

        # Flip to prevent image being rendered upside down when saved to a file
        red_image_tensor = torch.flip(torch.stack(red_image).t(), [0])
        green_image_tensor = torch.flip(torch.stack(green_image).t(), [0])
        blue_image_tensor = torch.flip(torch.stack(blue_image).t(), [0])

        if (text is not None):
            plt.text(0.5, 0.5, text, fontsize=14, backgroundcolor="white", alpha=0.5, color="black")
        plt.show()
        log.info("Done rendering in full!!")
        return (red_image_tensor, green_image_tensor, blue_image_tensor)

    def plot_from_image(self, image_data, plt, text=None):
        Renderer.initialise_plt(plt)
        red_render_channel, green_render_channel, blue_render_channel = image_data.detach().numpy()
        width, height = red_render_channel.shape
        for i in range(width):
            for j in range(height):
                plt.plot(i, height - 1 - j, marker="o",
                         color=[red_render_channel[j, i], green_render_channel[j, i], blue_render_channel[j, i]])
        if (text is not None):
            plt.text(0.5, 0.5, text, fontsize=14, backgroundcolor="white", alpha=0.5, color="black")
        plt.show()


def stochastic_samples(num_stochastic_samples, view_spec):
    x_1, x_2 = view_spec[0], view_spec[1]
    y_1, y_2 = view_spec[2], view_spec[3]
    view_length = x_2 - x_1
    view_height = y_2 - y_1

    # Need to convert the range [Random(0,1), Random(0,1)] into bounds of [[x1, x2], [y1, y2]]
    ray_intersection_weights = list(
        map(lambda x: torch.mul(torch.rand(2), torch.tensor([view_length, view_height])) + torch.tensor(
            [x_1, y_1]), list(range(0, num_stochastic_samples))))
    return ray_intersection_weights


def fullscreen_samples(view_spec):
    x_1, x_2 = view_spec[0], view_spec[1]
    y_1, y_2 = view_spec[2], view_spec[3]
    num_view_samples_x = view_spec[4]
    num_view_samples_y = view_spec[5]

    ray_intersection_weights = []
    for i in np.linspace(x_1, x_2, int(num_view_samples_x)):
        for j in np.linspace(y_1, y_2, int(num_view_samples_y)):
            ray_intersection_weights.append(torch.tensor([i, j]))

    log.info(f"Number of weights={len(ray_intersection_weights)}")
    return ray_intersection_weights


def camera_to_image(x, y, view_spec):
    view_x1, view_x2, view_y1, view_y2, num_rays_x, num_rays_y = view_spec
    step_x = (view_x2 - view_x1) / num_rays_x
    step_y = (view_y2 - view_y1) / num_rays_y

    # (view_y2 - y) implies we are flipping the Y-axis
    image_x = int((x - view_x1) / step_x)
    image_y = int((view_y2 - y) / step_y)

    # In the above calculation, [-1,1] maps to [0, num_rays]. Only +1 maps to num_rays.
    # We need to handle that isolated case and decrement by 1 to bring into the range
    #  of valid indices
    image_x = image_x if image_x < num_rays_x else image_x - 1
    image_y = image_y if image_y < num_rays_y else image_x - 1
    return (image_x, image_y)


def samples_to_image(red_samples, green_samples, blue_samples, view_spec, generate_background_pixel=torch.zeros):
    X, Y, INTENSITY = 0, 1, 2
    view_x1, view_x2, view_y1, view_y2, num_rays_x, num_rays_y = view_spec
    num_rays_x, num_rays_y = int(num_rays_x), int(num_rays_y)
    red_render_channel = generate_background_pixel([num_rays_y, num_rays_x])
    green_render_channel = generate_background_pixel([num_rays_y, num_rays_x])
    blue_render_channel = generate_background_pixel([num_rays_y, num_rays_x])
    for index, pixel in enumerate(red_samples):
        # print(
        #     f"({view_y2 - pixel[1]}, {pixel[0] - view_x1}) -> ({int((view_y2 - pixel[1]) / step_y)}, {int((pixel[0] - view_x1) / step_x)}), {pixel[2]}")
        x, y = camera_to_image(pixel[X], pixel[Y], view_spec)
        red_render_channel[y - 1, x - 1] = red_samples[index][INTENSITY]
        green_render_channel[y - 1, x - 1] = green_samples[index][INTENSITY]
        blue_render_channel[y - 1, x - 1] = blue_samples[index][INTENSITY]
    image_data = torch.stack([red_render_channel, green_render_channel, blue_render_channel])
    return image_data


def mse(rendered_channel, true_channel, view_spec):
    # true_channel = torch.ones([2, 2]) * 10
    small_diffs = 0
    medium_diffs = 0
    large_diffs = 0
    channel_total_error = torch.tensor(0.)
    for point in rendered_channel:
        x, y, intensity = point
        intensity = intensity
        image_x, image_y = camera_to_image(x, y, view_spec)
        pixel_error = (true_channel[image_y, image_x] - intensity).pow(2)
        # print(pixel_error)
        if (pixel_error <= 0.001):
            small_diffs += 1
        elif (pixel_error > 0.001 and pixel_error <= 0.01):
            medium_diffs += 1
        else:
            large_diffs += 1
        channel_total_error += pixel_error

    # print(f"Small diffs = {small_diffs}")
    # print(f"Medium diffs = {medium_diffs}")
    # print(f"Large diffs = {large_diffs}")
    return channel_total_error / len(rendered_channel)


def tv_for_voxel(voxel_accessor, world):
    voxel_max_x, voxel_max_y, voxel_max_z = world.voxel_dimensions()
    all_voxels = voxel_accessor.all_voxels
    voxel_positions = voxel_accessor.voxel_positions
    index = int(random.random() * len(all_voxels))
    position = voxel_positions[index]
    voxel = all_voxels[index]
    x_plus_1 = position + torch.tensor([1, 0, 0])
    y_plus_1 = position + torch.tensor([0, 1, 0])
    z_plus_1 = position + torch.tensor([0, 0, 1])
    # print(f"Position is {position}, TV coords are: {(x_plus_1, y_plus_1, z_plus_1)}")
    voxel_x1 = world.voxel_by_position(*x_plus_1)
    voxel_y1 = world.voxel_by_position(*y_plus_1)
    voxel_z1 = world.voxel_by_position(*z_plus_1)
    delta_x = ((voxel - voxel_x1) / (256 / voxel_max_x)).pow(2)
    delta_y = ((voxel - voxel_y1) / (256 / voxel_max_y)).pow(2)
    delta_z = ((voxel - voxel_z1) / (256 / voxel_max_z)).pow(2)

    tv_regularisation_term = (delta_x + delta_y + delta_z + 0.0001).sqrt().sum()
    if (math.isnan(tv_regularisation_term)):
        log.warning("[WARNING] NaN in TV regularisation term")
        log.warning(
            f"Sqrt sum={tv_regularisation_term}, Source voxel={voxel}, Positions are: {(x_plus_1, y_plus_1, z_plus_1)}, Voxels = {(voxel_x1, voxel_y1, voxel_z1)}, Deltas={(delta_x, delta_y, delta_z)}")
    return tv_regularisation_term


def tv_term(voxel_accessor, world):
    num_voxels_to_include = int(REGULARISATION_FRACTION * len(voxel_accessor.all_voxels))
    return torch.stack(
        list(map(lambda i: tv_for_voxel(voxel_accessor, world), list(range(num_voxels_to_include))))).mean()


def modify_grad(parameter_world, voxel_access):
    vx, vy, vz = parameter_world.voxel_dimensions()
    for i, j, k, v in parameter_world.all_voxels():
        v.requires_grad = False

    activated_parameters = []
    for ray_index, view_point in enumerate(voxel_access.view_points):
        ray = voxel_access.for_ray(ray_index)
        for i in range(ray.num_samples):
            sample_position, voxel_positions, voxels = ray.at(i)
            for voxel_position in voxel_positions:
                x, y, z = voxel_position
                if (x < 0 or x > vx - 1 or
                        y < 0 or y > vy - 1 or
                        z < 0 or z > vz - 1):
                    continue
                candidate_voxel = parameter_world.voxel_by_position(x, y, z)
                if (Voxel.is_pruned(candidate_voxel)):
                    continue
                candidate_voxel.requires_grad = True
                activated_parameters.append(torch.tensor([x, y, z]))

    if activated_parameters:
        log.info(f"Activated {len(torch.stack(activated_parameters).unique(dim=0))} parameters...")
    else:
        log.warning(f"[WARNING] No parameters were activated!!")


class PlenoxelModel(nn.Module):
    def __init__(self, world):
        super().__init__()
        self.parameter_world = VoxelGrid.as_parameter(world, self)

    def world(self):
        return self.parameter_world

    # @profile
    def forward(self, input):
        camera, view_spec, ray_spec = input
        # Use self.parameter_world as the weights, take camera as input
        renderer = Renderer(self.parameter_world, camera, view_spec, ray_spec)
        num_stochastic_rays = NUM_STOCHASTIC_RAYS
        # voxel_access = renderer.build_rays(stochastic_samples(num_stochastic_rays, view_spec))
        voxel_access = renderer.build_rays(fullscreen_samples(view_spec))
        r, g, b = renderer.render_from_rays(voxel_access)
        modify_grad(self.parameter_world, voxel_access)
        return r, g, b, renderer, voxel_access


def cauchy_term(voxel_access, world):
    all_unique_voxels = torch.stack(voxel_access.all_voxels)
    return torch.log(1 + 2 * all_unique_voxels[:, 0].pow(2)).sum()


# @profile
def train_minibatch(model, optimizer, camera, view_spec, ray_spec, image_channels, batch_index, epoch_index):
    log.info(f"Shape = {image_channels.shape}")
    optimizer.zero_grad()

    r, g, b, renderer, voxel_access = model([camera, view_spec, ray_spec])
    image = samples_to_image(r, g, b, view_spec, generate_background_pixel=Empty.ALL_EMPTY)

    log.info("Calculating loss...")
    log.info(f"TV Regularising using {int(len(voxel_access.all_voxels) * REGULARISATION_FRACTION)} voxels...")
    log.info(f"Cauchy Regularising using {len(voxel_access.all_voxels)} voxels...")
    red_mse = mse(r, image_channels[0], view_spec)
    green_mse = mse(g, image_channels[1], view_spec)
    blue_mse = mse(b, image_channels[2], view_spec)
    total_loss = red_mse + green_mse + blue_mse + \
                 TV_REGULARISATION_LAMBDA * tv_term(voxel_access, model.parameter_world) + \
                 CAUCHY_REGULARISATION_LAMBDA * cauchy_term(voxel_access, model.parameter_world)
    log.info(f"Loss={total_loss}, RGB MSE={(red_mse, green_mse, blue_mse)}")
    total_loss.backward()
    log.info(f"Model parameters: {len(list(model.parameters()))}")
    num_activated_parameters = 0
    for param in model.parameters():
        if (param.grad is None):
            continue
        num_activated_parameters += 1
        # print(f"Param grad after={param.grad}")
    log.info(f"Activated parameters tally with non-null gradient={num_activated_parameters}")
    #     print(f"Gradients={torch.max(param.grad)}")
    # make_dot(total_mse, params=dict(list(model.named_parameters()))).render("mse", format="png")
    # make_dot(r, params=dict(list(model.named_parameters()))).render("channel", format="png")
    optimizer.step()
    return total_loss.detach(), renderer, image, voxel_access


def render_training_images(camera_positions, focal_length, camera_look_at, world, view_spec, ray_spec, plt):
    for index, p in enumerate(camera_positions):
        c = Camera(focal_length, p, camera_look_at)
        r = Renderer(world, c, view_spec, ray_spec)
        red, green, blue = r.render(plt)
        save_image(torch.stack([red, green, blue]), f"./images/cube/training/rotating-cube-{index:02}.png")

    plt.show()
    log.info("Completed rendering images")


def prune_voxels(world, voxel_accessors):
    pruned_voxels = []
    for voxel_accessor in voxel_accessors:
        num_view_points = len(voxel_accessor.view_points)
        for ray_index in range(num_view_points):
            ray = voxel_accessor.for_ray(ray_index)
            for voxel_position in ray.voxel_positions:
                if world.prune(voxel_position):
                    pruned_voxels.append(voxel_position)
    return pruned_voxels


def prune_voxels2(world):
    pruned_voxels = []
    for i, j, k, v in world.all_voxels():
        voxel_position = torch.tensor([i, j, k])
        if world.prune(voxel_position):
            pruned_voxels.append(voxel_position)
    return pruned_voxels


def train(world, camera_look_at, focal_length, view_spec, ray_spec, training_positions, final_camera, num_epochs):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    CUBE_TRAINING_FOLDER = "./images/cube"
    TABLE_TRAINING_FOLDER = "./images/table/small-png"
    dataset = datasets.ImageFolder(TABLE_TRAINING_FOLDER, transform=to_tensor)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False)
    training_images = list(data_loader)[0][0]

    log.info(
        f"{training_images.shape[0]} images, {training_images.shape[1]} channels per image, resolution is {training_images.shape[2:]}")

    model = PlenoxelModel(world)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    epoch_losses = []
    voxel_accessors = []
    for epoch in range(num_epochs):
        batch_losses = []
        log.info(f"In epoch {epoch}")
        for batch, position in enumerate(training_positions[:1]):
            log.info(f"Before Training for camera position #{batch}={position}")
            test_camera = Camera(focal_length, position, camera_look_at)
            minibatch_loss, renderer, image, voxel_access = train_minibatch(model, optimizer, test_camera, view_spec,
                                                                            ray_spec,
                                                                            training_images[batch], batch, epoch)
            batch_losses.append(minibatch_loss)
            # voxel_accessors.append(voxel_access)
            log.info(f"After Training for camera position #{batch}={position}")
            renderer.plot_from_image(image, plt, f"Epoch: {epoch} Image: {batch}")
            save_image(image, f"{OUTPUT_FOLDER}/reconstruction/reconstruction-{epoch:02}-{batch:02}.png")

        # torch.save(model.parameter_world, f"{OUTPUT_FOLDER}/models/table-{epoch}.pt")
        epoch_losses.append(batch_losses)

    # pruned_voxels = prune_voxels(model.parameter_world, voxel_accessors)
    # model.parameter_world.prune(pruned_voxels)
    # print(f"Pruned {len(pruned_voxels)} voxels!!")
    final_renderer = Renderer(model.world(), final_camera, view_spec, ray_spec)
    red, green, blue = final_renderer.render(plt)
    transforms.ToPILImage()(torch.stack([red, green, blue])).show()
    log.info("Rendered final result")
    plt.show()
    return model.parameter_world, epoch_losses


# Reconstructs the world from disk
def reconstruct_flyby_from_file(filename, camera_positions, focal_length, look_at, view_spec, ray_spec):
    log.info(f"Camera position: {camera_positions}")
    voxel_grid = torch.load(filename)
    reconstructed_world = VoxelGrid.from_tensor(voxel_grid)
    log.info("Constructing flyby...")
    reconstruct_flyby_from_world(reconstructed_world, camera_positions, focal_length, look_at, view_spec, ray_spec)


def reconstruct_flyby_from_world(world, camera_positions, focal_length, look_at, view_spec, ray_spec):
    log.info("Constructing flyby...")
    for index, view_point in enumerate(camera_positions):
        c = Camera(focal_length, view_point, look_at)
        r1 = Renderer(world, c, view_spec, ray_spec)
        red, green, blue = r1.render(plt, text=f"Frame {index}")
        save_image(torch.stack([red, green, blue]), f"{OUTPUT_FOLDER}/frames/animated-cube-{index:02}.png")
    log.info("Finished constructing flyby!!")


def model_stats(filename, plt):
    model_tensor = torch.load(filename)
    world = VoxelGrid(model_tensor)
    all_opacities = []
    for i, j, k, v in world.all_voxels():
        all_opacities.append(v[0].detach())
    plt.figure()
    plt.hist(all_opacities, bins=50)
    pruned_voxels = prune_voxels2(world)
    log.info(f"Pruned voxels = {len(pruned_voxels)}")
    plt.show()


def test_upscale_rendering(world, original_renderer, camera, view_spec, ray_spec):
    upscaled_world = world.scale_up()
    model = PlenoxelModel(upscaled_world)
    upscale_renderer = Renderer(model.parameter_world, camera, view_spec, ray_spec)
    test_rendering(upscale_renderer, view_spec)
    test_rendering(original_renderer, view_spec)


def run_training(world, camera, view_spec, ray_spec):
    focal_length = camera.focal_length
    camera_look_at = camera.look_at

    RECONSTRUCTED_WORLD_FILENAME = f"{OUTPUT_FOLDER}/reconstructed.pt"
    # Trains on multiple training images
    test_positions = torch.tensor([[-20., -10., 40., 1.]])
    # training_positions = cube_training_positions()
    training_positions = table_training_positions()
    num_epochs = 30
    reconstructed_world, epoch_losses = train(world, camera_look_at, focal_length, view_spec, ray_spec,
                                              training_positions, camera, num_epochs)
    log.info(f"Epoch losses = {epoch_losses}")
    torch.save(reconstructed_world.voxel_grid, RECONSTRUCTED_WORLD_FILENAME)
    log.info(f"Saved world to {RECONSTRUCTED_WORLD_FILENAME}!")
    reconstruct_flyby_from_file(RECONSTRUCTED_WORLD_FILENAME, training_positions, focal_length, camera_look_at,
                                view_spec,
                                ray_spec)
    # camera_positions = generate_camera_angles(camera_radius, camera_look_at)

    log.info("Everything done!!")


def test_rendering(renderer, view_spec):
    # This draws stochastic rays and returns a set of samples with colours
    # However, it separates out the determining the intersecting voxels and the transmittance
    # calculations, so that it can be put through a Plenoxel model optimisation
    start_build_rays = timer()
    voxel_access = renderer.build_rays(fullscreen_samples(view_spec))
    # voxel_access = renderer.build_rays(stochastic_samples(2000, view_spec))
    end_build_rays = timer()
    log.info(f"Building rays took {end_build_rays - start_build_rays}")
    start_render_rays = timer()
    r, g, b = renderer.render_from_rays(voxel_access, clamping_function=ClampingFunctions.CLAMP)
    end_render_rays = timer()
    log.info(f"Rendering rays took {end_render_rays - start_render_rays}")
    image_data = samples_to_image(r, g, b, view_spec, generate_background_pixel=Empty.ALL_EMPTY)
    renderer.plot_from_image(image_data, plt)
    transforms.ToPILImage()(image_data).show()
    start_render_full = timer()
    renderer.render(plt, clamping_function=ClampingFunctions.CLAMP)
    end_render_full = timer()
    log.info(f"Rendering rays in full took {end_render_full - start_render_full}")
    log.info("Finished rendering!!")


def main():
    random_world = VoxelGrid.build_random_world(GRID_X, GRID_Y, GRID_Z)
    mono_world = VoxelGrid.build_with_voxel(GRID_X, GRID_Y, GRID_Z, torch.cat(
        [torch.tensor([0.0002, random.random() * 100.]), torch.zeros(VoxelGrid.VOXEL_DIMENSION - 2)]))
    empty_world = VoxelGrid.build_empty_world(GRID_X, GRID_Y, GRID_Z)
    empty_world.build_hollow_cube_with_randomly_coloured_sides(
        Voxel.uniform_harmonic_random_colour(density=0.4, requires_grad=True),
        torch.tensor([10, 10, 10, 20, 20, 20]))
    world = random_world
    # empty_world.build_solid_cube(torch.tensor([10, 10, 10, 20, 20, 20]))
    # world.build_monochrome_hollow_cube(torch.tensor([10, 10, 10, 20, 20, 20]))
    cube_center = torch.tensor([20., 20., 20., 1.])
    table_center = torch.tensor([20., 20., 35., 1.])
    # camera_look_at = torch.tensor([0., 0., 0., 1])
    # camera_look_at = cube_center
    camera_look_at = table_center

    camera_center = torch.tensor([-20., -10., 45., 1.])
    # camera_center = torch.tensor([-4.7487, 44.7487, 20.0000, 1.0000])
    camera_radius = 35.
    focal_length = 2
    camera = Camera(focal_length, camera_center, camera_look_at)
    num_rays_x, num_rays_y = 50, 50
    view_x1, view_x2 = -1, 1
    view_y1, view_y2 = -1.5, 0.5
    view_spec = torch.tensor([view_x1, view_x2, view_y1, view_y2, num_rays_x, num_rays_y])
    ray_length = 100
    num_ray_samples = 100
    ray_spec = torch.tensor([ray_length, num_ray_samples])

    renderer = Renderer(world, camera, view_spec, ray_spec)
    # test_rendering(renderer, view_spec)

    # Generates training images
    # camera_positions = generate_camera_angles(camera_radius, cube_center)
    # render_training_images(camera_positions, focal_length, cube_center, world, view_spec, ray_spec, plt, camera_radius)

    # upscaled_world = world.scale_up()
    run_training(world, camera, view_spec, ray_spec)
    # test_rendering(renderer, view_spec)
    # test_upscale_rendering(world, renderer, camera, view_spec, ray_spec)


if __name__ == '__main__':
    main()
