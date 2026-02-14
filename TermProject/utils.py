import csv
from itertools import combinations

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from a_star_ import (
    _BIAS_STATS,
    _DEBUG_BIAS_STATS,
    _NET_PATH_TO_INDEX,
    _collect_pin_indices,
    _evaluate_points,
    _hanan_candidates,
    _merge_net_paths,
    _mst_edges,
    _route_edges,
)

def set_obstacle(data_array, obstacles):
    for x, y in obstacles:
        data_array[y, x] = 1.0  # Set cell as occupied (1.0)
    return data_array

def a_star_loop_with_steiner(node_points, gmap, movement='8N', max_steiner_points=2):
    paths = {}
    success = 0
    iteration = 0
    total_costs = {}
    pin_indices = _collect_pin_indices(node_points, gmap)
    candidate_counts = []

    _NET_PATH_TO_INDEX.clear()
    for net_index, segment in enumerate(node_points):
        base_points = list(segment)
        candidates = _hanan_candidates(base_points, gmap)
        candidate_counts.append(len(candidates))
        # print(f"[steiner] net={net_index} candidates={len(candidates)}")
        best_points = base_points
        best_length = None

        edges, total_length = _evaluate_points(base_points, gmap, movement, pin_indices)
        if total_length is not None:
            best_length = total_length
            best_points = base_points

        for candidate in candidates:
            points = base_points + [candidate]
            edges, total_length = _evaluate_points(points, gmap, movement, pin_indices)
            if total_length is None:
                continue
            if best_length is None or total_length < best_length:
                best_length = total_length
                best_points = points

        if max_steiner_points >= 2:
            for candidate_a, candidate_b in combinations(candidates, 2):
                points = base_points + [candidate_a, candidate_b]
                edges, total_length = _evaluate_points(points, gmap, movement, pin_indices)
                if total_length is None:
                    continue
                if best_length is None or total_length < best_length:
                    best_length = total_length
                    best_points = points

        edges = _mst_edges(best_points)
        seg_paths, success, seg_costs, _ = _route_edges(
            edges,
            gmap,
            success,
            movement=movement,
            net_index=net_index,
            pin_indices=pin_indices,
        )
        iteration += len(edges)
        paths.update(seg_paths)
        total_costs.update(seg_costs)

    print(success, '/', iteration)
    if candidate_counts:
        max_c = max(candidate_counts)
        avg_c = sum(candidate_counts) / len(candidate_counts)
        # print(f"[steiner] candidates summary: nets={len(candidate_counts)} min={min(candidate_counts)} max={max_c} avg={avg_c:.2f}")
        if max_steiner_points >= 2:
            est_evals = 1 + max_c + (max_c * (max_c - 1)) // 2
            # print(f"[steiner] worst-case evals per net (<=2 steiners): {est_evals}")
    # if _DEBUG_BIAS_STATS:
    #     print(
    #         "Bias stats:",
    #         "bias_steps_triggered=" + str(_BIAS_STATS["bias_steps_triggered"]),
    #         "\nturn_penalty_applied=" + str(_BIAS_STATS["turn_penalty_applied"]),
    #         "\nnodes_expanded=" + str(_BIAS_STATS["nodes_expanded"]),
    #     )
    return paths, success, total_costs

def save_paths_to_csv(paths, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Start", "End", "Path"])  # Header
        if _NET_PATH_TO_INDEX and len(_NET_PATH_TO_INDEX) == len(paths):
            net_paths = {}
            for (start, end), path in paths.items():
                net_index = _NET_PATH_TO_INDEX.get((start, end))
                net_paths.setdefault(net_index, []).append((start, end, path))
            for net_index in sorted(net_paths):
                merged_paths = _merge_net_paths(net_paths[net_index])
                for start, end, path in merged_paths:
                    writer.writerow([str(start), str(end), str(path)])
        else:
            for (start, end), path in paths.items():
                writer.writerow([str(start), str(end), str(path)])

def plot_routing(gmap, paths, colors, fig_name):
    # Visualize the grid, obstacles, and the paths
    plt.figure(figsize=(10, 10))
    gmap.plot(origin='upper', alpha=0.5, min_val=0)

    plt.gca().set_facecolor('black')

    # Draw start and end points, paths
    for i, (key, path) in enumerate(paths.items()):
        start_m, end_m = key
        net_index = _NET_PATH_TO_INDEX.get(key)
        if net_index is None:
            color_index = i % len(colors)
        else:
            color_index = net_index % len(colors)
        if path:
            color = colors[color_index]
            path_idx = [gmap.get_index_from_coordinates(x, y) for x, y in path]

            # Draw path
            plt.plot([p[0] for p in path], [p[1] for p in path], color=color, linewidth=1)

            # Draw start and end points
            plt.scatter([start_m[0]], [start_m[1]], color=color, edgecolors='black', s=20, label=f'Start {start_m}')
            plt.scatter([end_m[0]], [end_m[1]], color=color, edgecolors='black', s=20, label=f'End {end_m}')
        # else:
        #     print(f"No path found between start point {start_m} and end point {end_m}.")

    plt.title("A* Pathfinding")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.gca().invert_yaxis()
    # Set grid intervals
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.grid(False)
    plt.savefig(fig_name, transparent=True)

def compute_congestion_bins_from_paths(paths, grid_step):
    max_x = 0
    max_y = 0
    for path in paths.values():
        if not path:
            continue
        for x, y in path:
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y

    cols = int(max_x // grid_step) + 1
    rows = int(max_y // grid_step) + 1
    congestion = np.zeros((rows, cols), dtype=np.int32)

    for path in paths.values():
        if not path:
            continue
        for x, y in path:
            col = int(x // grid_step)
            row = int(y // grid_step)
            if 0 <= row < rows and 0 <= col < cols:
                congestion[row, col] += 1

    return congestion

def plot_congestion_from_paths(
    congestion,
    grid_step=5,
    output_file="congestion.png",
    threshold=0,
    vmax=None,
):
    if congestion.size == 0:
        print("No valid paths found for congestion plot.")
        return

    max_value = int(congestion.max())
    if max_value > 0:
        max_indices = np.argwhere(congestion == max_value)
        ranges = []
        for row, col in max_indices.tolist():
            x_start = col * grid_step
            x_end = x_start + grid_step - 1
            y_start = row * grid_step
            y_end = y_start + grid_step - 1
            ranges.append((x_start, x_end, y_start, y_end))
        print(f"Max congestion: {max_value} at ranges {ranges}")
    else:
        print("Max congestion: 0 (no path points found)")

    rows, cols = congestion.shape
    x_max = (cols * grid_step) - 1
    y_max = (rows * grid_step) - 1
    plt.figure(figsize=(10, 10))
    masked = np.ma.masked_less(congestion, threshold)
    cmap = plt.cm.inferno.copy()
    cmap.set_bad(color="black")
    plt.imshow(
        masked,
        origin="lower",
        cmap=cmap,
        extent=[0, x_max, 0, y_max],
        aspect="equal",
        vmin=0,
        vmax=vmax,
    )
    plt.colorbar(label="Path Points Count")
    plt.title("Routing Congestion")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.tight_layout()
    plt.savefig(output_file)

def print_max_congestion(congestion, grid_step=5):
    if congestion.size == 0:
        print("Max congestion: 0 (no path points found)")
        return
    max_value = int(congestion.max())
    if max_value > 0:
        max_indices = np.argwhere(congestion == max_value)
        ranges = []
        for row, col in max_indices.tolist():
            x_start = col * grid_step
            x_end = x_start + grid_step - 1
            y_start = row * grid_step
            y_end = y_start + grid_step - 1
            ranges.append((x_start, x_end, y_start, y_end))
        print(f"Max congestion: {max_value} at ranges {ranges}")
    else:
        print("Max congestion: 0 (no path points found)")

def apply_congestion_cost_to_gmap(
    gmap,
    congestion,
    grid_step,
    alpha=0.3,
    cap=0.95,
    scale_ratio=1.0,
):
    if congestion.size == 0:
        return
    max_c = int(congestion.max())
    if max_c <= 0:
        return
    rows, cols = congestion.shape
    for y in range(gmap.dim_cells[0]):
        y_s = int(y * scale_ratio)
        row = int(y_s // grid_step)
        if row >= rows:
            continue
        for x in range(gmap.dim_cells[1]):
            x_s = int(x * scale_ratio)
            col = int(x_s // grid_step)
            if col >= cols:
                continue
            if gmap.get_data_idx((x, y), 0) >= gmap.occupancy_threshold:
                continue
            c = congestion[row, col]
            if c <= 0:
                continue
            delta = alpha * (c / max_c)
            new_val = min(cap, gmap.get_data_idx((x, y), 0) + delta)
            gmap.set_data_idx((x, y), new_val, 0)
            gmap.set_data_idx((x, y), new_val, 1)

def apply_congestion_cost_to_gmap_maxonly(
    gmap,
    congestion,
    grid_step,
    alpha=0.3,
    cap=0.95,
    scale_ratio=1.0,
):
    if congestion.size == 0:
        return
    max_c = int(congestion.max())
    if max_c <= 0:
        return
    max_mask = congestion == max_c
    rows, cols = congestion.shape
    for y in range(gmap.dim_cells[0]):
        y_s = int(y * scale_ratio)
        row = int(y_s // grid_step)
        if row >= rows:
            continue
        for x in range(gmap.dim_cells[1]):
            x_s = int(x * scale_ratio)
            col = int(x_s // grid_step)
            if col >= cols:
                continue
            if not max_mask[row, col]:
                continue
            if gmap.get_data_idx((x, y), 0) >= gmap.occupancy_threshold:
                continue
            delta = alpha
            new_val = min(cap, gmap.get_data_idx((x, y), 0) + delta)
            gmap.set_data_idx((x, y), new_val, 0)
            gmap.set_data_idx((x, y), new_val, 1)

def apply_congestion_cost_to_gmap_topn(
    gmap,
    congestion,
    grid_step,
    alpha=0.3,
    cap=0.95,
    scale_ratio=1.0,
    top_n=1,
):
    if congestion.size == 0:
        return
    if top_n <= 0:
        return
    flat = congestion.ravel()
    nonzero = np.count_nonzero(flat)
    if nonzero == 0:
        return
    top_n = min(top_n, nonzero)
    top_indices = np.argpartition(flat, -top_n)[-top_n:]
    mask = np.zeros_like(flat, dtype=bool)
    mask[top_indices] = True
    top_mask = mask.reshape(congestion.shape)
    rows, cols = congestion.shape
    for y in range(gmap.dim_cells[0]):
        y_s = int(y * scale_ratio)
        row = int(y_s // grid_step)
        if row >= rows:
            continue
        for x in range(gmap.dim_cells[1]):
            x_s = int(x * scale_ratio)
            col = int(x_s // grid_step)
            if col >= cols:
                continue
            if not top_mask[row, col]:
                continue
            if gmap.get_data_idx((x, y), 0) >= gmap.occupancy_threshold:
                continue
            delta = alpha
            new_val = min(cap, gmap.get_data_idx((x, y), 0) + delta)
            gmap.set_data_idx((x, y), new_val, 0)
            gmap.set_data_idx((x, y), new_val, 1)
