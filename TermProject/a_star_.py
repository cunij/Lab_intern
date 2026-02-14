import math
import csv
import ast
from heapq import heappush, heappop
import numpy as np
import matplotlib.pyplot as plt
from gridmap import OccupancyGridMap

_NET_PATH_TO_INDEX = {}
_DEBUG_BIAS_STATS = True
_BIAS_STATS = {
    "bias_steps_triggered": 0,
    "turn_penalty_applied": 0,
    "nodes_expanded": 0,
}

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

def a_star(
    start_m,
    goal_m,
    gmap,
    success,
    movement='8N',
    occupancy_cost_factor=3,
    lateral_penalty=0.3,
    lateral_penalty_steps=1,
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

    while front:
        element = heappop(front)
        total_cost, cost, pos, previous, current_layer, previous_layer, bias_steps = element

        # 이미 방문했는지 확인
        if gmap.is_visited_idx(pos, layer=current_layer):
            continue

        _BIAS_STATS["nodes_expanded"] += 1

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
                _BIAS_STATS["bias_steps_triggered"] += 1
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
                lateral_bias = 0.0
                if previous and bias_steps > 0:
                    prev_dx = pos[0] - previous[0]
                    prev_dy = pos[1] - previous[1]
                    if (dx, dy) != (prev_dx, prev_dy):
                        turn_penalty = lateral_penalty
                        _BIAS_STATS["turn_penalty_applied"] += 1
                new_cost = (
                    cost
                    + deltacost
                    + potential_function_cost
                    + turn_penalty
                    
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


def _clone_gmap(gmap):
    new_gmap = OccupancyGridMap(np.copy(gmap.data[0]), gmap.cell_size, gmap.occupancy_threshold)
    new_gmap.data[1] = np.copy(gmap.data[1])
    new_gmap.static_obstacles = np.copy(gmap.static_obstacles)
    return new_gmap

def _path_length_4n(path_idx):
    if not path_idx:
        return None
    return max(len(path_idx) - 1, 0)

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
        # else:
            # print(f"No path found between {start_m} and {end_m}")
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

def plot_congestion_from_csv(input_file, grid_step=5, output_file="congestion.png", threshold=0, vmax=None):
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
        vmin=0,
        vmax=vmax,
    )
    plt.colorbar(label="Path Points Count")
    plt.title("Routing Congestion")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.tight_layout()
    plt.savefig(output_file)
