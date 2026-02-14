import numpy as np
from randomnet import generate_random_nets, save_nets_py
from gridmap import OccupancyGridMap
from a_star_ import plot_congestion_from_csv
from utils import *
import time
from itertools import chain

plot_congestion_from_csv("paths.csv", grid_step=5, output_file="congestion.png", vmax=28)
