import argparse
import time
from itertools import chain

import numpy as np
from a_star_ import plot_congestion_from_csv
from gridmap import OccupancyGridMap
from randomnet import generate_random_nets, save_nets_py
from utils import *

  # term_RL 기준 상위 폴더



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x-size", type=int, default=200)
    parser.add_argument("--y-size", type=int, default=200)
    parser.add_argument("--cell-size", type=float, default=1.0)
    parser.add_argument("--scalar", type=float, default=1)
    parser.add_argument("--use-random-nets", type=int, default=0)
    parser.add_argument("--random-nets-count", type=int, default=30)
    parser.add_argument("--min-pins-per-net", type=int, default=2)
    parser.add_argument("--max-pins-per-net", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--random-margin", type=int, default=1)
    parser.add_argument("--congestion-grid-size", type=int, default=5)
    parser.add_argument("--congestion-reweight", type=int, default=0)
    parser.add_argument("--congestion-source-scalar", type=float, default=-1)
    parser.add_argument("--congestion-cost-alpha", type=float, default=0.3)
    parser.add_argument("--congestion-cost-cap", type=float, default=0.95)
    parser.add_argument("--congestion-top-n", type=int, default=7)
    parser.add_argument("--congestion-plot-max", type=float, default=28)
    parser.add_argument("--steiner-point", type=int, default=1)
    parser.add_argument("--image_number", type=float, default=1)
    parser.add_argument("--save-files", type=int, default=1)
    parser.add_argument("--plot", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    # Start measuring time
    start_time = time.time()

    # Define grid size
    x_size = args.x_size
    y_size = args.y_size
    cell_size = args.cell_size
    scalar = args.scalar
    USE_RANDOM_NETS = args.use_random_nets != 0
    RANDOM_NETS_COUNT = args.random_nets_count
    MIN_PINS_PER_NET = args.min_pins_per_net
    MAX_PINS_PER_NET = args.max_pins_per_net
    RANDOM_SEED = args.random_seed
    RANDOM_MARGIN = args.random_margin
    CONGESTION_GRID_SIZE = args.congestion_grid_size
    CONGESTION_REWEIGHT = args.congestion_reweight != 0
    CONGESTION_SOURCE_SCALAR = args.congestion_source_scalar
    CONGESTION_COST_ALPHA = args.congestion_cost_alpha
    CONGESTION_COST_CAP = args.congestion_cost_cap
    CONGESTION_TOP_N = args.congestion_top_n
    CONGESTION_PLOT_MAX = args.congestion_plot_max if args.congestion_plot_max >= 0 else None
    STEINER_POINT = args.steiner_point
    IMAGE_NUMBER = args.image_number
    SAVE_FILES = args.save_files != 0
    PLOT = args.plot != 0


    # Define start and end points in meters
    Net = [((161, 46), (95, 84), (135, 60), (161, 82), (133, 82), (105, 62)), # A[3]
           ((91, 62), (91, 46), (97, 62)), # A[2]
           ((93, 96), (129, 134), (145, 82), (113, 82), (113, 82), (137, 118), (103, 118), (139, 82)), # A[1]
           ((99, 134), (161, 120), (139, 118), (97, 118), (165, 134)), # A[0]
           ((165, 118), (159, 82), (167, 118), (111, 82), (143, 118)), # B[3]
           ((165, 116), (159, 94), (159, 80), (143, 116), (143, 94), (111, 94), (111, 82)), # B[2]
           ((103, 132), (103, 64), (133, 132), (87, 60), (131, 62), (135, 118), (105, 118), (135, 82)), # B[1]
           ((97, 98), (91, 82), (101, 118), (87, 82), (101, 62)), # B[0]
           ((149, 62), (151, 86)), # n1
           ((145, 62), (137, 58)), # n2
           ((77, 82), (85, 58)), # n4
           ((73, 82), (97, 86)), # n5
           ((37, 82), (105, 86)), # n7
           ((53, 82), (67, 84)), # n8
           ((147, 134), (159, 122)), # n9
           ((143, 134), (135, 130)), # n10
           ((59, 130), (47, 96)), # n11
           ((25, 98), (141, 134), (147, 122)), # n12
           ((9, 98), (11, 84)), # n13
           ((31, 24), (71, 82), (93, 58)), # n16
           ((69, 26), (139, 60)), # n18
           ((15, 60), (57, 94)), # n19
           ((37, 64), (5, 78)), # n20
           ((53, 62), (53, 40), (25, 40), (25, 28)), # n21
           ((155, 28), (153, 76)), # n23
           ((171, 28), (143, 76), (143, 64)), # n24
           ((55, 12), (5, 58)), # n25
           ((33, 10), (21, 22)), # n26
           ((127, 24), (17, 10)), # n27
           ((95, 24), (95, 10)), # n28
           ((167, 44), (161, 10), (105, 26), (99, 8)), # n29
           ((143, 10), (123, 22), (107, 22), (105, 10), (101, 26)), # n30
           ((137, 10), (101, 10), (99, 28), (65, 10)), # n31
           ((157, 8), (141, 8)), # n32
           ((137, 138), (11, 136)), # n35
           ((93, 122), (81, 118), (49, 132)), # n36
           ((91, 116), (91, 100)), # n37
           ((105, 130), (83, 116)) # n38
           ]
    if USE_RANDOM_NETS:
        Net = generate_random_nets(
            RANDOM_NETS_COUNT,
            MIN_PINS_PER_NET,
            MAX_PINS_PER_NET,
            int(x_size),
            int(y_size),
            seed=RANDOM_SEED,
            margin=RANDOM_MARGIN,
        )
        save_nets_py(Net, "random_nets_generated.py")
    print("===", scalar, "===")
    scaled_Net_points = [
        [(int(x * scalar), int(y * scalar)) for x, y in segment]
        for segment in Net
    ]
    all_points = list(chain.from_iterable(scaled_Net_points))

    # Create an empty grid with all cells free (0.0)
    data_array = np.zeros((int(y_size * scalar), int(x_size * scalar)))
    # Define obstacles
    obstacles = []
    obstacles.extend(all_points)
    data_array = set_obstacle(data_array, obstacles)

    # Create an occupancy grid map
    gmap = OccupancyGridMap(data_array, cell_size)

    # Initialize variables
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']

    if CONGESTION_REWEIGHT:
        source_scalar = CONGESTION_SOURCE_SCALAR if CONGESTION_SOURCE_SCALAR > 0 else scalar
        if source_scalar != scalar:
            print("=== congestion source scalar ===", source_scalar)
            scaled_source_points = [
                [(int(x * source_scalar), int(y * source_scalar)) for x, y in segment]
                for segment in Net
            ]
            source_all_points = list(chain.from_iterable(scaled_source_points))
            source_data_array = np.zeros((int(y_size * source_scalar), int(x_size * source_scalar)))
            source_obstacles = []
            source_obstacles.extend(source_all_points)
            source_data_array = set_obstacle(source_data_array, source_obstacles)
            source_gmap = OccupancyGridMap(source_data_array, cell_size)

            pass1_start = time.time()
            paths_first, success_first, total_costs_first = a_star_loop_with_steiner(
                scaled_source_points,
                source_gmap,
                '4N',
                STEINER_POINT,
            )
            pass1_end = time.time()
            congestion = compute_congestion_bins_from_paths(paths_first, CONGESTION_GRID_SIZE)
            print_max_congestion(congestion, CONGESTION_GRID_SIZE)
            if SAVE_FILES:
                plot_congestion_from_paths(
                    congestion,
                    grid_step=CONGESTION_GRID_SIZE,
                    output_file="congestion.png",
                    vmax=CONGESTION_PLOT_MAX,
                )
            gmap = OccupancyGridMap(data_array, cell_size)
            scale_ratio = source_scalar / scalar
            apply_congestion_cost_to_gmap_topn(
                gmap,
                congestion,
                CONGESTION_GRID_SIZE,
                alpha=CONGESTION_COST_ALPHA,
                cap=CONGESTION_COST_CAP,
                scale_ratio=scale_ratio,
                top_n=CONGESTION_TOP_N,
            )
            pass2_start = time.time()
            paths, success, total_costs = a_star_loop_with_steiner(
                scaled_Net_points,
                gmap,
                '4N',
                STEINER_POINT,
            )
            pass2_end = time.time()
            congestion_final = compute_congestion_bins_from_paths(paths, CONGESTION_GRID_SIZE)
            print_max_congestion(congestion_final, CONGESTION_GRID_SIZE)
            print("Pass1 time: {:.4f} ms".format((pass1_end - pass1_start) * 1000))
            print("Pass2 time: {:.4f} ms".format((pass2_end - pass2_start) * 1000))
        else:
            pass1_start = time.time()
            paths_first, success_first, total_costs_first = a_star_loop_with_steiner(
                scaled_Net_points,
                gmap,
                '4N',
                STEINER_POINT,
            )
            pass1_end = time.time()
            congestion = compute_congestion_bins_from_paths(paths_first, CONGESTION_GRID_SIZE)
            print_max_congestion(congestion, CONGESTION_GRID_SIZE)
            if SAVE_FILES:
                plot_congestion_from_paths(
                    congestion,
                    grid_step=CONGESTION_GRID_SIZE,
                    output_file="congestion.png",
                    vmax=CONGESTION_PLOT_MAX,
                )
            gmap = OccupancyGridMap(data_array, cell_size)
            apply_congestion_cost_to_gmap(
                gmap,
                congestion,
                CONGESTION_GRID_SIZE,
                alpha=CONGESTION_COST_ALPHA,
                cap=CONGESTION_COST_CAP,
            )
            pass2_start = time.time()
            paths, success, total_costs = a_star_loop_with_steiner(
                scaled_Net_points,
                gmap,
                '4N',
                STEINER_POINT,
            )
            pass2_end = time.time()
            congestion_final = compute_congestion_bins_from_paths(paths, CONGESTION_GRID_SIZE)
            print_max_congestion(congestion_final, CONGESTION_GRID_SIZE)
            print("Pass1 time: {:.4f} ms".format((pass1_end - pass1_start) * 1000))
            print("Pass2 time: {:.4f} ms".format((pass2_end - pass2_start) * 1000))
    else:
        # Find paths for each net with 1-steiner candidate search
        paths, success, total_costs = a_star_loop_with_steiner(
            scaled_Net_points,
            gmap,
            '4N',
            STEINER_POINT,
        )

# End measuring time
    end_time = time.time()
    cost_sum = sum(total_costs[key] for key in total_costs)
    print("Sum of total HPWL:", cost_sum/scalar)
    elapsed_time = (end_time - start_time) * 1000
    print("Elapsed time: {:.4f} ms".format(elapsed_time))
    if SAVE_FILES:
        output_file = "./paths.csv"
        save_paths_to_csv(paths, output_file)
        if not CONGESTION_REWEIGHT:
            plot_congestion_from_csv(
                "paths.csv",
                grid_step=CONGESTION_GRID_SIZE,
                output_file="congestion.png",
                vmax=CONGESTION_PLOT_MAX,
            )
        # print(f"Paths saved to {output_file}")
    # # Visualize path with gds format
    # create_gds_paths_with_ports_and_squares(paths, scaled_start_points, scaled_end_points, width=0.3)

    # Visualize the grid, obstacles, and the paths
    if PLOT:
        plot_routing(gmap, paths, colors, str(IMAGE_NUMBER) + ".png")


if __name__ == "__main__":
    main()
