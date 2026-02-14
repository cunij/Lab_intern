import numpy as np
import matplotlib.pyplot as plt

class OccupancyGridMap:
    def __init__(self, data_array, cell_size, occupancy_threshold=1.0):
        self.data = [np.copy(data_array), np.copy(data_array)]  # Layer 0과 Layer 1
        self.dim_cells = data_array.shape
        self.dim_meters = (self.dim_cells[0] * cell_size, self.dim_cells[1] * cell_size)
        self.cell_size = cell_size
        self.static_obstacles = np.copy(data_array)
        self.occupancy_threshold = occupancy_threshold
        self.visited_layers = np.zeros((2, *self.dim_cells), dtype=np.float32)  # 두 개의 visited layer 생성
    
    def set_visited_empty(self, layer=0):
        if layer not in [0, 1]:
            raise ValueError("Invalid layer index. Must be 0 or 1.")
        self.visited_layers[layer] = np.zeros(self.dim_cells, dtype=np.float32)

    def erase_visited_idx(self, point_idx, layer=0):  # start point를 위해
        if layer not in [0, 1]:
            raise ValueError("Invalid layer index. Must be 0 or 1.")
        x_index, y_index = point_idx
        if not self.is_inside_idx(point_idx):
            raise Exception('Point is outside map boundary')
        self.visited_layers[layer][y_index][x_index] = 0.0

    def mark_visited_idx(self, point_idx, layer=0):
        if layer not in [0, 1]:
            raise ValueError("Invalid layer index. Must be 0 or 1.")
        x_index, y_index = point_idx
        if not self.is_inside_idx(point_idx):
            raise Exception('Point is outside map boundary')
        self.visited_layers[layer][y_index][x_index] = 1.0

    def mark_visited(self, point, layer=0):
        x, y = point
        x_index, y_index = self.get_index_from_coordinates(x, y)
        return self.mark_visited_idx((x_index, y_index), layer=layer)

    def is_visited_idx(self, point_idx, layer=0):
        if layer not in [0, 1]:
            raise ValueError("Invalid layer index. Must be 0 or 1.")
        x_index, y_index = point_idx
        if not self.is_inside_idx(point_idx):
            raise Exception('Point is outside map boundary')
        return self.visited_layers[layer][y_index][x_index] == 1.0

    def is_visited(self, point, layer=0):
        x, y = point
        x_index, y_index = self.get_index_from_coordinates(x, y)
        return self.is_visited_idx((x_index, y_index), layer=layer)

    def get_data_idx(self, point_idx, layer):
        x_index, y_index = point_idx
        if not self.is_inside_idx(point_idx):
            raise Exception('Point is outside map boundary')
        return self.data[layer][y_index][x_index]

    def get_data(self, point):
        x, y = point
        x_index, y_index = self.get_index_from_coordinates(x, y)
        return self.get_data_idx((x_index, y_index))

    def set_data_idx(self, point_idx, new_value, layer):
        x_index, y_index = point_idx
        if not self.is_inside_idx(point_idx):
            raise Exception('Point is outside map boundary')
        self.data[layer][y_index][x_index] = new_value

    def set_data(self, point, new_value, layer):
        x, y = point
        x_index, y_index = self.get_index_from_coordinates(x, y)
        self.set_data_idx((x_index, y_index), new_value, layer)

    def is_inside_idx(self, point_idx):
        x_index, y_index = point_idx
        return 0 <= x_index < self.dim_cells[1] and 0 <= y_index < self.dim_cells[0]

    def is_inside(self, point):
        x, y = point
        x_index, y_index = self.get_index_from_coordinates(x, y)
        return self.is_inside_idx((x_index, y_index))

    def is_occupied_idx(self, point_idx, layer):
        return self.get_data_idx(point_idx, layer) >= self.occupancy_threshold
    
    def get_index_from_coordinates(self, x, y):
        x_index = int(round(x / self.cell_size))
        y_index = int(round(y / self.cell_size))
        return x_index, y_index

    def get_coordinates_from_index(self, x_index, y_index):
        x = x_index * self.cell_size
        y = y_index * self.cell_size
        return x, y
    
    def plot(self, alpha=1, min_val=0, origin='lower'):
        plt.imshow(self.static_obstacles, vmin=min_val, vmax=1, origin=origin, interpolation='none', alpha=alpha)
        plt.draw()


