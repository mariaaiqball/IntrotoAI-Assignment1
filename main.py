import numpy as np
import random
import os
import matplotlib.pyplot as plt

# Constants
GRID_SIZE = 101
NUM_GRIDS = 50
BLOCK_PROB = 0.3  # 30% probability of being blocked
SAVE_DIR = "gridworlds"

# Directions for movement (N, S, E, W)
DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

def generate_gridworld():
    """Generates a single 101x101 gridworld using DFS with random tie-breaking."""
    grid = np.full((GRID_SIZE, GRID_SIZE), -1)  # -1 indicates unvisited
    
    # Start from a random cell
    start_x, start_y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
    stack = [(start_x, start_y)]
    grid[start_x, start_y] = 0  # Mark as unblocked
    
    while stack:
        x, y = stack[-1]
        neighbors = []
        
        # Find unvisited neighbors
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[nx, ny] == -1:
                neighbors.append((nx, ny))
        
        if neighbors:
            # Randomly pick a neighbor
            nx, ny = random.choice(neighbors)
            
            # Mark as blocked/unblocked with probabilities
            if random.random() < BLOCK_PROB:
                grid[nx, ny] = 1  # Blocked
            else:
                grid[nx, ny] = 0  # Unblocked
                stack.append((nx, ny))  # Continue DFS
        else:
            stack.pop()  # Backtrack
    
    # Convert -1 to 0 (default unblocked if not visited)
    grid[grid == -1] = 0
    return grid

def save_gridworld(grid, index):
    """Saves the gridworld to a file."""
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    np.savetxt(f"{SAVE_DIR}/grid_{index}.txt", grid, fmt="%d")

def load_gridworld(index):
    """Loads a gridworld from a file."""
    return np.loadtxt(f"{SAVE_DIR}/grid_{index}.txt", dtype=int)

def visualize_grid(grid):
    """Visualizes the gridworld using Matplotlib."""
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap="gray_r")  # Black for blocked, white for unblocked
    plt.title("Gridworld Visualization")
    plt.show()

# Generate, save, and visualize 50 gridworlds
def main():
    print("Generating 50 gridworld environments...")
    for i in range(NUM_GRIDS):
        grid = generate_gridworld()
        save_gridworld(grid, i)
        print(f"Grid {i+1} saved.")
    print("All gridworlds generated and saved.")
    
    # Load and visualize a random gridworld as a sample
    sample_grid = load_gridworld(random.randint(0, NUM_GRIDS - 1))
    visualize_grid(sample_grid)

if __name__ == "__main__":
    main()
