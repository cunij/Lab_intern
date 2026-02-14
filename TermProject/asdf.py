def dist2d(point1, point2):
    x1, y1 = point1[0:2]
    x2, y2 = point2[0:2]
    dist2 = (x1 - x2) ** 2 + (y1 - y2) ** 2
    return math.sqrt(dist2)

def dist2d(point1, point2):
    x1, y1 = point1[0:2]
    x2, y2 = point2[0:2]
    dist2 = abs(x1 - x2) + abs(y1 - y2)
    return dist2*1.001