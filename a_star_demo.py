import numpy as np
import matplotlib.pyplot as plt
import heapq
import time

class Node:
    def __init__(self, position, g=0, h=0):
        self.position = position
        self.g = g
        self.h = h
        self.f = g + h

    def __lt__(self, other):
        return self.f < other.f


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_star(start, goal, grid):
    open_list = []
    closed_set = set()
    path = []

    start_node = Node(start, 0, heuristic(start, goal))
    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)
        closed_set.add(current_node.position)

        if current_node.position == goal:
            path = reconstruct_path(current_node)
            break

        neighbors = get_neighbors(current_node.position, grid)
        for next_position in neighbors:
            if next_position in closed_set:
                continue

            g_cost = current_node.g + 1
            h_cost = heuristic(next_position, goal)
            neighbor_node = Node(next_position, g_cost, h_cost)
            neighbor_node.parent = current_node

            if any(neighbor.position == next_position and neighbor.g <= g_cost for neighbor in open_list):
                continue

            heapq.heappush(open_list, neighbor_node)

    return path


def get_neighbors(position, grid):
    neighbors = []
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for direction in directions:
        neighbor = (position[0] + direction[0], position[1] + direction[1])
        if is_within_bounds(neighbor, grid) and grid[neighbor[0]][neighbor[1]] == 0:
            neighbors.append(neighbor)
    return neighbors


def is_within_bounds(position, grid):
    return 0 <= position[0] < grid.shape[0] and 0 <= position[1] < grid.shape[1]


def reconstruct_path(node):
    path = []
    current = node
    while current:
        path.append(current.position)
        current = current.parent if hasattr(current, 'parent') else None
    return path[::-1]


def animate_path(grid, path, start, goal):
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap='Greys', origin='upper')
    plt.scatter(start[1], start[0], c='green', label='Start', s=100)
    plt.scatter(goal[1], goal[0], c='red', label='Goal', s=100)

    robot_position = start
    for step in path:
        plt.scatter(robot_position[1], robot_position[0], c='blue', s=100)
        plt.pause(1)
        robot_position = step

    plt.scatter(robot_position[1], robot_position[0], c='blue', s=100)
    plt.legend()
    plt.title('A* Pathfinding Animation')
    plt.gca().invert_yaxis()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    grid = np.array([[0, 0, 0, 0, 0],
                     [0, 1, 1, 1, 0],
                     [0, 0, 0, 0, 0],
                     [0, 1, 1, 0, 0],
                     [0, 0, 0, 0, 0]])

    start = (0, 0)
    goal = (4, 4)

    path = a_star(start, goal, grid)

    print("Path from start to goal:", path)
    animate_path(grid, path, start, goal)
