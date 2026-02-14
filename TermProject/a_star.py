import math
import csv
import ast
from heapq import heappush, heappop
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from gridmap import OccupancyGridMap

_NET_PATH_TO_INDEX = {}

def dist2d(point1, point2):
    x1, y1 = point1[0:2]
    x2, y2 = point2[0:2]
    dist2 = abs(x1 - x2) + abs(y1 - y2)
    return dist2*1.001

def _get_movements_4n():
    return [(1, 0, 1.0), (0, 1, 1.0), (-1, 0, 1.0), (0, -1, 1.0)]

def _get_movements_8n():
    s2 = math.sqrt(2)
    return [(1, 0, 1.0), (0, 1, 1.0), (-1, 0, 1.0), (0, -1, 1.0), (1, 1, s2), (-1, 1, s2), (-1, -1, s2), (1, -1, s2)]

def _a_star_once(
    start_m,
    goal_m,
    gmap,
    success,
    movement='8N',
    occupancy_cost_factor=3,
    lateral_penalty=5.0,
    lateral_penalty_steps=5,
    forward_penalty=10.0,
    lookahead_steps=3,
    lookahead_gamma=0.7,
    lookahead_grid_step=3,
    lookahead_cost_factor=1.0,
    # lateral_bonus=7.0,
    heuristic_weight=1.0,
):
    gmap.set_visited_empty(layer=0)  # Layer 0 초기화
    gmap.set_visited_empty(layer=1)  # Layer 1 초기화

    start = gmap.get_index_from_coordinates(start_m[0], start_m[1])
    goal = gmap.get_index_from_coordinates(goal_m[0], goal_m[1])

    if gmap.is_occupied_idx(start,0):
        gmap.set_data(start, 0, 0)
        gmap.set_data(start, 0, 1)

    if gmap.is_occupied_idx(goal,0):
        gmap.set_data(goal, 0, 0)
        gmap.set_data(goal, 0, 1)

    if gmap.is_visited_idx(start, layer=0):
        gmap.erase_visited_idx(start, layer=0)

    if gmap.is_visited_idx(goal, layer=0):
        gmap.erase_visited_idx(goal, layer=0)

    start_node_cost = 0
    start_node_estimated_cost_to_goal = (dist2d(start, goal) * heuristic_weight) + start_node_cost
    front = [(start_node_estimated_cost_to_goal, start_node_cost, start, None, 0, 0, 0)]  # 마지막 숫자는 bias steps
    came_from = {0: {}, 1: {}}  # Layer 0과 1의 경로 추적을 분리

    if movement == '4N':
        movements = _get_movements_4n()
    elif movement == '8N':
        movements = _get_movements_8n()
    else:
        raise ValueError('Unknown movement')

    congestion_bins = None
    if lookahead_steps > 0:
        occupied_points = _occupied_points_from_gmap(gmap)
        congestion_bins = _compute_congestion_bins(occupied_points, lookahead_grid_step)

    while front:
        element = heappop(front)
        total_cost, cost, pos, previous, current_layer, previous_layer, bias_steps = element

        # 이미 방문했는지 확인
        if gmap.is_visited_idx(pos, layer=current_layer):
            continue

        # 현재 레이어에 방문 표시
        gmap.mark_visited_idx(pos, layer=current_layer)

        # 방향 전환 시 다른 레이어에도 방문 표시
        if previous:
            dx = pos[0] - previous[0]
            dy = pos[1] - previous[1]
            if dx != 0 and dy != 0:  # 방향이 바뀌는 경우
                gmap.mark_visited_idx(pos, layer=1 - current_layer)
            forward_pos = (pos[0] + dx, pos[1] + dy)
            forward_layer = 0 if dy == 0 else 1
            if gmap.is_inside_idx(forward_pos) and gmap.is_occupied_idx(forward_pos, forward_layer):
                bias_steps = max(bias_steps, lateral_penalty_steps)

        # 경로 추적
        came_from[current_layer][pos] = (previous, previous_layer)

        if pos == goal:
            success += 1
            break

        for dx, dy, deltacost in movements:
            new_x = pos[0] + dx
            new_y = pos[1] + dy
            new_pos = (new_x, new_y)

            if not gmap.is_inside_idx(new_pos):
                continue

            new_layer = 0 if dy == 0 else 1  # 방향에 따라 새로운 레이어 결정

            if not gmap.is_visited_idx(new_pos, layer=new_layer) and not gmap.is_occupied_idx(new_pos, new_layer):
                potential_function_cost = gmap.get_data_idx(new_pos, new_layer) * occupancy_cost_factor
                turn_penalty = 0.0
                forward_bias = 0.0
                lateral_bias = 0.0
                lookahead_cost = 0.0
                if previous:
                    prev_dx = pos[0] - previous[0]
                    prev_dy = pos[1] - previous[1]
                    is_straight = (dx, dy) == (prev_dx, prev_dy)
                    if is_straight and congestion_bins is not None:
                        lookahead_cost = _lookahead_congestion_cost(
                            pos,
                            (dx, dy),
                            congestion_bins,
                            lookahead_steps,
                            lookahead_gamma,
                            lookahead_grid_step,
                        )
                    if bias_steps > 0:
                        if not is_straight:
                            turn_penalty = lateral_penalty
                            #lateral_bias = -lateral_bonus
                        else:
                            forward_bias = forward_penalty
                new_cost = (
                    cost
                    + deltacost
                    + potential_function_cost
                    + turn_penalty
                    + forward_bias
                    + (lookahead_cost * lookahead_cost_factor)
                    
                )
                new_total_cost_to_goal = new_cost + (dist2d(new_pos, goal) * heuristic_weight) + potential_function_cost
                next_bias_steps = max(bias_steps - 1, 0)
                heappush(front, (new_total_cost_to_goal, new_cost, new_pos, pos, new_layer, current_layer, next_bias_steps))

    # 경로 재구성
    path = []
    path_idx = []
    if pos == goal:
        while pos:
            path_idx.append(pos)
            pos_m_x, pos_m_y = gmap.get_coordinates_from_index(pos[0], pos[1])
            path.append((pos_m_x, pos_m_y))

            # 이전 노드와 레이어로 이동
            pos, current_layer = came_from[current_layer].get(pos, (None, None))

        path.reverse()
        path_idx.reverse()
    
    return path, path_idx, success, cost

def a_star(
    start_m,
    goal_m,
    gmap,
    success,
    movement='8N',
    occupancy_cost_factor=3,
    lateral_penalty=10.0,
    lateral_penalty_steps=0,
    forward_penalty=10.0,
    lookahead_steps=3,
    lookahead_gamma=0.2,
    lookahead_grid_step=3,
    lookahead_cost_factor=1.0,
    
):
    return _a_star_once(
        start_m,
        goal_m,
        gmap,
        success,
        movement=movement,
        occupancy_cost_factor=occupancy_cost_factor,
        lateral_penalty=lateral_penalty,
        lateral_penalty_steps=lateral_penalty_steps,
        forward_penalty=forward_penalty,
        lookahead_steps=lookahead_steps,
        lookahead_gamma=lookahead_gamma,
        lookahead_grid_step=lookahead_grid_step,
        lookahead_cost_factor=lookahead_cost_factor,
        
        heuristic_weight=1.0,
    )

def _clone_gmap(gmap):
    new_gmap = OccupancyGridMap(np.copy(gmap.data[0]), gmap.cell_size, gmap.occupancy_threshold)
    new_gmap.data[1] = np.copy(gmap.data[1])
    new_gmap.static_obstacles = np.copy(gmap.static_obstacles)
    return new_gmap

def _path_length_4n(path_idx):
    if not path_idx:
        return None
    return max(len(path_idx) - 1, 0)

def _occupied_points_from_gmap(gmap):
    points = set()
    rows, cols = gmap.dim_cells
    for y in range(rows):
        for x in range(cols):
            if gmap.data[0][y][x] >= gmap.occupancy_threshold or gmap.data[1][y][x] >= gmap.occupancy_threshold:
                points.add((x, y))
    return list(points)

def _compute_congestion_bins(points, grid_step):
    if not points:
        return np.zeros((1, 1), dtype=np.int32)
    max_x = max(x for x, _ in points)
    max_y = max(y for _, y in points)
    cols = int(max_x // grid_step) + 1
    rows = int(max_y // grid_step) + 1
    congestion = np.zeros((rows, cols), dtype=np.int32)
    for x, y in points:
        col = int(x // grid_step)
        row = int(y // grid_step)
        if 0 <= row < rows and 0 <= col < cols:
            congestion[row, col] += 1
    return congestion

def _lookahead_congestion_cost(
    pos,
    direction,
    congestion_bins,
    lookahead_steps,
    lookahead_gamma,
    grid_step,
):
    dx, dy = direction
    cost = 0.0
    rows, cols = congestion_bins.shape
    for k in range(1, lookahead_steps + 1):
        x = pos[0] + dx * k
        y = pos[1] + dy * k
        col = int(x // grid_step)
        row = int(y // grid_step)
        if row < 0 or col < 0 or row >= rows or col >= cols:
            break
        cost += (lookahead_gamma ** k) * congestion_bins[row, col]
    return cost

def _mark_path_occupancy(path_idx, gmap):
    prev_pos = None
    prev_dx = None
    prev_dy = None
    for pos in path_idx:
        if prev_pos is not None:
            dx = pos[0] - prev_pos[0]
            dy = pos[1] - prev_pos[1]
            if dx != 0 and dx == prev_dx:  # Horizontal movement
                gmap.set_data_idx(pos, 1, layer=0)
            elif dy != 0 and dy == prev_dy:  # Vertical movement
                gmap.set_data_idx(pos, 1, layer=1)
            else:  # Diagonal or bending
                gmap.set_data_idx(pos, 1, layer=0)
                gmap.set_data_idx(pos, 1, layer=1)
            prev_dx = dx
            prev_dy = dy
        prev_pos = pos

def _mst_edges(points):
    if len(points) < 2:
        return []
    remaining = set(points)
    current = remaining.pop()
    tree = {current}
    edges = []
    while remaining:
        best = None
        best_dist = None
        for u in tree:
            for v in remaining:
                dist = abs(u[0] - v[0]) + abs(u[1] - v[1])
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best = (u, v)
        edges.append(best)
        tree.add(best[1])
        remaining.remove(best[1])
    return edges

def _hanan_candidates(points, gmap):
    xs = sorted({p[0] for p in points})
    ys = sorted({p[1] for p in points})
    candidates = []
    for x in xs:
        for y in ys:
            if (x, y) in points:
                continue
            if not gmap.is_inside((x, y)):
                continue
            idx = gmap.get_index_from_coordinates(x, y)
            if gmap.is_occupied_idx(idx, 0) or gmap.is_occupied_idx(idx, 1):
                continue
            candidates.append((x, y))
    return candidates

def _collect_pin_indices(node_points, gmap):
    pin_indices = set()
    for segment in node_points:
        for x, y in segment:
            pin_indices.add(gmap.get_index_from_coordinates(x, y))
    return pin_indices

def _apply_pin_blockers(gmap, pin_indices, open_indices):
    for idx in pin_indices:
        gmap.set_data_idx(idx, 1, layer=0)
        gmap.set_data_idx(idx, 1, layer=1)
    for idx in open_indices:
        gmap.set_data_idx(idx, 0, layer=0)
        gmap.set_data_idx(idx, 0, layer=1)

def _route_edges(edges, gmap, success, movement='8N', net_index=None, pin_indices=None):
    paths = {}
    total_costs = {}
    total_lengths = {}
    for start_m, end_m in edges:
        if pin_indices is not None:
            start_idx = gmap.get_index_from_coordinates(start_m[0], start_m[1])
            end_idx = gmap.get_index_from_coordinates(end_m[0], end_m[1])
            _apply_pin_blockers(gmap, pin_indices, {start_idx, end_idx})
        path, path_idx, success, fin_cost = a_star(
            start_m,
            end_m,
            gmap,
            success,
            movement=movement,
        )
        key = (start_m, end_m)
        paths[key] = path
        total_costs[key] = fin_cost
        path_length = _path_length_4n(path_idx)
        if path_length is not None:
            total_lengths[key] = path_length
        if net_index is not None:
            _NET_PATH_TO_INDEX[key] = net_index
        if path:
            _mark_path_occupancy(path_idx, gmap)
        else:
            print(f"No path found between {start_m} and {end_m}")
    return paths, success, total_costs, total_lengths

def _evaluate_points(points, gmap, movement, pin_indices):
    edges = _mst_edges(points)
    test_gmap = _clone_gmap(gmap)
    _, _, _, total_lengths = _route_edges(
        edges,
        test_gmap,
        0,
        movement=movement,
        pin_indices=pin_indices,
    )
    if len(total_lengths) != len(edges):
        return None, None
    total_length = sum(total_lengths.values())
    return edges, total_length

def a_star_loop(node_points, gmap, movement='8N'):
    return a_star_loop_with_steiner(node_points, gmap, movement=movement)

def a_star_loop_with_steiner(node_points, gmap, movement='8N', max_steiner_points=2):
    paths = {}
    success = 0
    iteration = 0
    total_costs = {}
    pin_indices = _collect_pin_indices(node_points, gmap)

    _NET_PATH_TO_INDEX.clear()
    for net_index, segment in enumerate(node_points):
        base_points = list(segment)
        candidates = _hanan_candidates(base_points, gmap)
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
    return paths, success, total_costs

def _merge_net_paths(net_paths):
    degree = {}
    polylines = []
    for start, end, path in net_paths:
        if not path:
            continue
        degree[start] = degree.get(start, 0) + 1
        degree[end] = degree.get(end, 0) + 1
        polylines.append(list(path))

    changed = True
    while changed:
        changed = False
        for i in range(len(polylines)):
            if changed:
                break
            a = polylines[i]
            a_start = a[0]
            a_end = a[-1]
            for j in range(i + 1, len(polylines)):
                b = polylines[j]
                b_start = b[0]
                b_end = b[-1]
                if a_end == b_start and degree.get(a_end, 0) == 2:
                    merged = a + b[1:]
                elif a_start == b_end and degree.get(a_start, 0) == 2:
                    merged = b + a[1:]
                elif a_start == b_start and degree.get(a_start, 0) == 2:
                    merged = list(reversed(a)) + b[1:]
                elif a_end == b_end and degree.get(a_end, 0) == 2:
                    merged = a + list(reversed(b))[1:]
                else:
                    continue
                polylines[i] = merged
                polylines.pop(j)
                changed = True
                break

    merged_paths = []
    for path in polylines:
        if not path:
            continue
        merged_paths.append((path[0], path[-1], path))
    return merged_paths

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

def plot_routing(gmap, paths, colors):
    # Visualize the grid, obstacles, and the paths
    plt.figure(figsize=(10, 10))
    gmap.plot(origin='upper', alpha=0.5, min_val=0)

    plt.gca().set_facecolor('black')  # Set background color to light grey

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

    plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
    plt.savefig('paths_mainn.png')

def plot_congestion_from_csv(input_file, grid_step=5, output_file="congestion.png", threshold=0):
    with open(input_file, mode="r") as file:
        reader = csv.DictReader(file)
        paths = []
        for row in reader:
            try:
                path = ast.literal_eval(row["Path"])
            except (ValueError, SyntaxError):
                continue
            if path:
                paths.append(path)

    if not paths:
        print("No valid paths found for congestion plot.")
        return

    max_x = 0
    max_y = 0
    for path in paths:
        for x, y in path:
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y

    cols = int(max_x // grid_step) + 1
    rows = int(max_y // grid_step) + 1
    congestion = np.zeros((rows, cols), dtype=np.int32)

    for path in paths:
        for x, y in path:
            col = int(x // grid_step)
            row = int(y // grid_step)
            if 0 <= row < rows and 0 <= col < cols:
                congestion[row, col] += 1

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
    )
    plt.colorbar(label="Path Points Count")
    plt.title("Routing Congestion")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.tight_layout()
    plt.savefig(output_file)
