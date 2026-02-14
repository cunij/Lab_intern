import csv
import ast
import matplotlib.pyplot as plt

def read_paths_from_csv(input_file):
    paths = {}
    total_rows = 0
    success_count = 0
    with open(input_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            total_rows += 1
            start = ast.literal_eval(row["Start"])
            end = ast.literal_eval(row["End"])
            try:
                path = ast.literal_eval(row["Path"])  # Convert string to list
                if not path:
                    print(f"Fail: Path from {start} to {end} is empty.")
                    paths[(start, end)] = None
                else:
                    paths[(start, end)] = path
                    success_count += 1
            except ValueError:
                print(f"Fail: Path from {start} to {end} is invalid.")
                paths[(start, end)] = None
    return paths, total_rows, success_count

def calculate_wire_length_and_bending(paths):
    total_length = 0
    total_bending = 0
    for path in paths.values():
        if path is None or len(path) < 2:
            continue
        
        total_length += len(path) - 1  # Path segments count
        # Calculate bending
        for i in range(1, len(path) - 1):
            dx1 = path[i][0] - path[i - 1][0]
            dy1 = path[i][1] - path[i - 1][1]
            dx2 = path[i + 1][0] - path[i][0]
            dy2 = path[i + 1][1] - path[i][1]
            if (dx1, dy1) != (dx2, dy2):  # Change in direction
                total_bending += 1
    return total_length, total_bending

def plot_paths(paths):
    plt.figure(figsize=(10, 10))
    for (start, end), path in paths.items():
        if path is None:
            continue
        x_coords = [p[0] for p in path]
        y_coords = [p[1] for p in path]
        plt.plot(
            x_coords, y_coords, marker='o', markersize=1, linewidth=2, label=f"{start} -> {end}"
        )
    
    plt.title('Routing Paths')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid()
    plt.axis('equal')
    plt.savefig('evaluation.png')

# 사용 예제
input_file = "paths.csv"

# CSV에서 경로 읽기
paths, total_rows, success_count = read_paths_from_csv(input_file)

# Wire Length 및 Bending 계산
wl, bending = calculate_wire_length_and_bending(paths)
print(f"WL = {wl}")
print(f"Bending = {bending}")

# 성공률 계산 및 출력
success_rate = success_count / total_rows if total_rows > 0 else 0
print(f"Success Rate = {success_count}/{total_rows} ({success_rate * 100:.2f}%)")

# 경로 플롯
plot_paths(paths)