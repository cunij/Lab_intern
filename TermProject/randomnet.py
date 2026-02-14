import random

def generate_random_nets(
    num_nets,
    min_pins_per_net,
    max_pins_per_net,
    x_max,
    y_max,
    seed=None,
    margin=0,
):
    if seed is not None:
        random.seed(seed)
    if min_pins_per_net < 2 or max_pins_per_net < 2:
        raise ValueError("min_pins_per_net and max_pins_per_net must be >= 2")
    if min_pins_per_net > max_pins_per_net:
        raise ValueError("min_pins_per_net must be <= max_pins_per_net")
    xs = range(margin, x_max - margin)
    ys = range(margin, y_max - margin)
    all_cells = [(x, y) for x in xs for y in ys]
    max_needed = num_nets * max_pins_per_net
    if max_needed > len(all_cells):
        raise ValueError("Not enough cells for requested random nets")
    pool = all_cells[:]
    random.shuffle(pool)
    idx = 0
    nets = []
    for _ in range(num_nets):
        pins_per_net = random.randint(min_pins_per_net, max_pins_per_net)
        if idx + pins_per_net > len(pool):
            raise ValueError("Not enough cells for requested random nets")
        nets.append(tuple(pool[idx:idx + pins_per_net]))
        idx += pins_per_net
    return nets

def save_nets_py(nets, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("NETS = [\n")
        for net in nets:
            f.write("    " + repr(net) + ",\n")
        f.write("]\n")
