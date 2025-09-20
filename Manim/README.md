
# What is Manim?

Manim is a powerful animation engine created by mathematician Grant Sanderson (3Blue1Brown) for generating explanatory math videos. It's a Python library that allows you to create precise, programmatic animations for explaining mathematical and algorithmic concepts through beautiful visualizations.

Key features of Manim include:
- Mathematical equation rendering using LaTeX
- Geometric shapes and transformations
- Graph plotting and visualization
- Animation of algorithms and data structures
- Camera movements and scene transitions
- Timeline-based animation control

Manim is particularly well-suited for creating educational content about algorithms because it allows you to:
1. Visualize step-by-step execution
2. Highlight key operations
3. Show data transformations
4. Create consistent, professional-looking animations
5. Script complex sequences for repeatability

## How to Install Manim

There are currently two main versions of Manim:
1. **ManimCE (Community Edition)** - The most actively maintained version
2. **ManimGL** - Grant's original version with OpenGL rendering

For algorithm visualizations, ManimCE is generally recommended due to its active community and better documentation. Here's how to install it:

### Prerequisites

Before installing Manim, you'll need:
- Python 3.7 or newer
- FFmpeg (for video rendering)
- LaTeX (for rendering mathematical expressions)
- Various system dependencies

### Installation Steps by Operating System

#### Windows:

1. **Install Python**:
   - Download and install from [python.org](https://www.python.org/downloads/)
   - Make sure to check "Add Python to PATH" during installation

2. **Install FFmpeg**:
   - Download from [ffmpeg.org](https://ffmpeg.org/download.html) or use Chocolatey:
   ```
   choco install ffmpeg
   ```

3. **Install LaTeX** (MiKTeX is recommended):
   - Download from [miktex.org](https://miktex.org/download)
   - During installation, select "Install missing packages on the fly"

4. **Install Manim**:
   ```
   pip install manim
   ```

#### macOS:

1. **Install Homebrew** (if not already installed):
   ```
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install dependencies**:
   ```
   brew install python ffmpeg cairo pango scipy
   ```

3. **Install LaTeX**:
   ```
   brew install --cask mactex
   ```

4. **Install Manim**:
   ```
   pip install manim
   ```

#### Linux (Ubuntu/Debian):

1. **Install dependencies**:
   ```
   sudo apt update
   sudo apt install python3-pip python3-venv python3-dev libcairo2-dev libpango1.0-dev ffmpeg
   ```

2. **Install LaTeX**:
   ```
   sudo apt install texlive texlive-latex-extra texlive-fonts-extra texlive-latex-recommended texlive-science texlive-xetex
   ```

3. **Install Manim**:
   ```
   pip3 install manim
   ```

### Verifying the Installation

After installation, verify that Manim is working correctly:

```python
from manim import *

class SquareToCircle(Scene):
    def construct(self):
        square = Square()
        circle = Circle()
        self.play(Create(square))
        self.play(Transform(square, circle))
        self.wait()

# Save this as test_manim.py
```

Run it with:
```
manim -pql test_manim.py SquareToCircle
```

The `-pql` flags mean:
- `-p`: Preview (play the video after rendering)
- `-q`: Quality (l=low, m=medium, h=high, k=4K)
- `-l`: Low quality (faster rendering for testing)

## Using a Virtual Environment (Recommended)

It's good practice to use a virtual environment:

```
python -m venv manim-env
# On Windows
manim-env\Scripts\activate
# On macOS/Linux
source manim-env/bin/activate

pip install manim
```

## Alternative: Using Docker

If you're having trouble with installation, Manim also provides Docker images:

```
docker pull manimcommunity/manim
docker run --rm -it -v "/path/to/your/project:/manim" manimcommunity/manim
```

## Learning Resources

- [Official Manim Documentation](https://docs.manim.community/)
- [Manim Community Examples](https://github.com/ManimCommunity/manim/tree/main/example_scenes)
- [Manim Tutorials on YouTube](https://www.youtube.com/playlist?list=PLsMrDyoG1sZm6-jIUQCgN3BVyEVOZz3LQ)
- [Manim Discord Community](https://www.manim.community/discord/)


# Algorithm Visualization Prompts (16:9 Optimized)

## 1. Stable Matching (Gale-Shapley Algorithm)
```
Create a Manim animation that renders directly in the current directory, not in a nested media folder.
The animation should visualize the Gale-Shapley algorithm for stable matching, showing how participants propose, accept, and reject until a stable matching is reached. Include a small example with 3-4 participants on each side, displaying their preference lists and tracking the matching process step by step.

For optimal visualization:
- Configure a 16:9 aspect ratio (config.frame_width = 16, config.frame_height = 9)
- Use dedicated screen regions: title bar at top, status bar at bottom, clean center area for visualization
- Use color coding (e.g., blue for proposers, red for receivers, green for matches)
- Implement curved connecting lines to avoid overlap when showing matches
- Include self-contained rendering code at the end of the file:

```python
if __name__ == "__main__":
    from manim.cli.render.commands import render_animation
    from pathlib import Path
    
    # Configure rendering
    config.media_dir = "."
    config.video_dir = "."
    config.images_dir = "."
    config.preview = True
    config.format = "mp4"
    config.quality = "l"  # low quality for faster rendering
    
    # Configure 16:9 aspect ratio
    config.frame_height = 9
    config.frame_width = 16
    
    # Render the scene
    render_animation([YourSceneClassName], Path("your_filename.py"))
```
```

## 2. Algorithm Analysis (Big-Oh Notation)
```
Create a Manim animation that renders directly in the current directory, not in a nested media folder.
The animation should illustrate Big-Oh notation by comparing growth rates of common functions (constant, logarithmic, linear, n log n, quadratic, exponential). Show these functions on a coordinate system with animations of how they grow as n increases, and include visual representations of their practical implications for algorithm performance.

For optimal visualization:
- Configure a 16:9 aspect ratio (config.frame_width = 16, config.frame_height = 9)
- Use a dedicated title bar at the top and information panel at the bottom
- Create a large central coordinate system for function visualization
- Use distinct colors for each function with a consistent color legend
- Implement zooming animations to show both small and large input scales
- Include practical time comparisons (e.g., "10⁹ operations = 1 second on modern hardware")
- Add self-contained rendering code:

```python
if __name__ == "__main__":
    from manim.cli.render.commands import render_animation
    from pathlib import Path
    
    # Configure rendering
    config.media_dir = "."
    config.video_dir = "."
    config.images_dir = "."
    config.preview = True
    config.format = "mp4"
    config.quality = "l"  # low quality for faster rendering
    
    # Configure 16:9 aspect ratio
    config.frame_height = 9
    config.frame_width = 16
    
    # Render the scene
    render_animation([BigOhVisualization], Path("big_oh_visualization.py"))
```
```

## 3. Graph Search Algorithms (BFS/DFS)
```
Create a Manim animation that renders directly in the current directory, not in a nested media folder.
The animation should demonstrate both Breadth-First Search (BFS) and Depth-First Search (DFS) on the same graph. Display a medium-sized graph (8-10 nodes) and animate how each algorithm traverses the nodes, highlighting the frontier, visited nodes, and the queue/stack data structures used by each algorithm.

For optimal visualization:
- Configure a 16:9 aspect ratio (config.frame_width = 16, config.frame_height = 9)
- Position the graph centrally with ample spacing between nodes
- Create dedicated areas for visualizing the queue/stack data structures
- Use a consistent color scheme (e.g., blue for unvisited, green for frontier, red for visited)
- Implement a step counter and status display to track algorithm progress
- Show the algorithms sequentially with clear transition between BFS and DFS
- Use curved edges for better visibility in dense graphs
- Add self-contained rendering code:

```python
if __name__ == "__main__":
    from manim.cli.render.commands import render_animation
    from pathlib import Path
    
    # Configure rendering
    config.media_dir = "."
    config.video_dir = "."
    config.images_dir = "."
    config.preview = True
    config.format = "mp4"
    config.quality = "l"  # low quality for faster rendering
    
    # Configure 16:9 aspect ratio
    config.frame_height = 9
    config.frame_width = 16
    
    # Render the scene
    render_animation([GraphSearchVisualization], Path("graph_search_visualization.py"))
```
```

## 4. Greedy Algorithms (Interval Scheduling)
```
Create a Manim animation that renders directly in the current directory, not in a nested media folder.
The animation should visualize the greedy algorithm for interval scheduling. Show a set of intervals on a timeline, and animate the process of selecting intervals that finish earliest and don't overlap with previously selected intervals. Include a comparison with a non-optimal approach to demonstrate why the greedy strategy works.

For optimal visualization:
- Configure a 16:9 aspect ratio (config.frame_width = 16, config.frame_height = 9) 
- Create a wide horizontal timeline as the central element
- Represent intervals as rectangles with distinct colors and labels
- Include a step counter and algorithm explanation in a dedicated area
- Split the screen to show side-by-side comparison between optimal greedy approach and non-optimal alternative
- Use animation to highlight conflicts and selection decisions
- Create a final summary showing efficiency metrics for both approaches
- Add self-contained rendering code:

```python
if __name__ == "__main__":
    from manim.cli.render.commands import render_animation
    from pathlib import Path
    
    # Configure rendering
    config.media_dir = "."
    config.video_dir = "."
    config.images_dir = "."
    config.preview = True
    config.format = "mp4"
    config.quality = "l"  # low quality for faster rendering
    
    # Configure 16:9 aspect ratio
    config.frame_height = 9
    config.frame_width = 16
    
    # Render the scene
    render_animation([IntervalSchedulingVisualization], Path("interval_scheduling_visualization.py"))
```
```

## 5. Divide and Conquer (Merge Sort)
```
Create a Manim animation that renders directly in the current directory, not in a nested media folder.
The animation should illustrate the merge sort algorithm using divide and conquer. Show an array being recursively divided into smaller subarrays, and then demonstrate the merging process where the subarrays are combined in sorted order. Use color coding and clear transitions to show each step of the algorithm.

For optimal visualization:
- Configure a 16:9 aspect ratio (config.frame_width = 16, config.frame_height = 9)
- Represent array elements as squares with numbers inside them
- Create a tree-like visualization showing the recursive division and merging
- Use animation paths that clearly show elements moving during the merge process
- Implement consistent color coding for different stages (dividing, comparing, merging)
- Add a step counter and algorithm stage indicator
- Include a time/space complexity display
- Create small explanatory text for key concepts
- Add self-contained rendering code:

```python
if __name__ == "__main__":
    from manim.cli.render.commands import render_animation
    from pathlib import Path
    
    # Configure rendering
    config.media_dir = "."
    config.video_dir = "."
    config.images_dir = "."
    config.preview = True
    config.format = "mp4"
    config.quality = "l"  # low quality for faster rendering
    
    # Configure 16:9 aspect ratio
    config.frame_height = 9
    config.frame_width = 16
    
    # Render the scene
    render_animation([MergeSortVisualization], Path("merge_sort_visualization.py"))
```
```

## 6. Dynamic Programming (Knapsack Problem)
```
Create a Manim animation that renders directly in the current directory, not in a nested media folder.
The animation should demonstrate solving the 0/1 Knapsack Problem using dynamic programming. Show items with their weights and values, visualize the construction of the dynamic programming table cell by cell, and highlight the optimal solution backtracking through the completed table.

For optimal visualization:
- Configure a 16:9 aspect ratio (config.frame_width = 16, config.frame_height = 9)
- Create a visual representation of items (e.g., icons with weight/value labels)
- Position the DP table in the center of the screen with ample size
- Use color highlighting to show the currently calculated cell and its dependent cells
- Implement a capacity indicator showing remaining knapsack space
- Create a step-by-step formula breakdown for calculating each cell value
- Include visual backtracking through the table to show optimal solution construction
- Add self-contained rendering code:

```python
if __name__ == "__main__":
    from manim.cli.render.commands import render_animation
    from pathlib import Path
    
    # Configure rendering
    config.media_dir = "."
    config.video_dir = "."
    config.images_dir = "."
    config.preview = True
    config.format = "mp4"
    config.quality = "l"  # low quality for faster rendering
    
    # Configure 16:9 aspect ratio
    config.frame_height = 9
    config.frame_width = 16
    
    # Render the scene
    render_animation([KnapsackVisualization], Path("knapsack_visualization.py"))
```
```

## 7. Network Flow (Ford-Fulkerson Algorithm)
```
Create a Manim animation that renders directly in the current directory, not in a nested media folder.
The animation should visualize the Ford-Fulkerson algorithm for finding maximum flow in a network. Use a simple directed graph with capacities on edges, and animate the process of finding augmenting paths and updating residual capacities until no more augmenting paths exist.

For optimal visualization:
- Configure a 16:9 aspect ratio (config.frame_width = 16, config.frame_height = 9)
- Create a large central graph with clearly labeled nodes and edges
- Show edge capacity and current flow using a fraction display (flow/capacity)
- Visualize flow using animated particles or varying edge thickness
- Highlight augmenting paths with distinct colors
- Create a residual graph visualization that updates with each iteration
- Include a running total of current maximum flow
- Add a step counter and algorithm phase indicator
- Add self-contained rendering code:

```python
if __name__ == "__main__":
    from manim.cli.render.commands import render_animation
    from pathlib import Path
    
    # Configure rendering
    config.media_dir = "."
    config.video_dir = "."
    config.images_dir = "."
    config.preview = True
    config.format = "mp4"
    config.quality = "l"  # low quality for faster rendering
    
    # Configure 16:9 aspect ratio
    config.frame_height = 9
    config.frame_width = 16
    
    # Render the scene
    render_animation([FordFulkersonVisualization], Path("ford_fulkerson_visualization.py"))
```
```

## 8. Intractability and NP-Completeness
```
Create a Manim animation that renders directly in the current directory, not in a nested media folder.
The animation should explain the concept of NP-Completeness by visualizing the relationship between complexity classes (P, NP, NP-Complete, NP-Hard). Include an illustration of polynomial-time reductions between problems, specifically showing how a problem like 3-SAT can be reduced to another NP-Complete problem.

For optimal visualization:
- Configure a 16:9 aspect ratio (config.frame_width = 16, config.frame_height = 9)
- Create a dedicated title bar with clear section indicators
- Visualize complexity classes using Venn diagrams or nested regions
- Use animations to show the concept of polynomial-time reductions
- Include small visual examples of common NP-Complete problems
- Demonstrate a specific reduction (e.g., 3-SAT to 3-Coloring) with step-by-step mapping
- Create graphical representations of time complexity functions
- Add a "big picture" summary at the end
- Add self-contained rendering code:

```python
if __name__ == "__main__":
    from manim.cli.render.commands import render_animation
    from pathlib import Path
    
    # Configure rendering
    config.media_dir = "."
    config.video_dir = "."
    config.images_dir = "."
    config.preview = True
    config.format = "mp4"
    config.quality = "l"  # low quality for faster rendering
    
    # Configure 16:9 aspect ratio
    config.frame_height = 9
    config.frame_width = 16
    
    # Render the scene
    render_animation([NPCompletenessVisualization], Path("np_completeness_visualization.py"))
```
```

## 9. Approximation Algorithms (Set Cover)
```
Create a Manim animation that renders directly in the current directory, not in a nested media folder.
The animation should demonstrate a greedy approximation algorithm for the Set Cover problem. Show a universe of elements and available sets, then animate the process of repeatedly selecting the set that covers the most uncovered elements until all elements are covered. Compare the approximation result with an optimal solution.

For optimal visualization:
- Configure a 16:9 aspect ratio (config.frame_width = 16, config.frame_height = 9)
- Create a universe display with elements represented as small circles
- Show available sets as colored regions or outlines around groups of elements
- Split the screen to show greedy algorithm on one side and optimal solution on the other
- Implement a coverage counter and efficiency metric
- Use animations to highlight newly covered elements with each set selection
- Include a cost comparison between the approximation and optimal solution
- Add a theoretical bound visualization
- Add self-contained rendering code:

```python
if __name__ == "__main__":
    from manim.cli.render.commands import render_animation
    from pathlib import Path
    
    # Configure rendering
    config.media_dir = "."
    config.video_dir = "."
    config.images_dir = "."
    config.preview = True
    config.format = "mp4"
    config.quality = "l"  # low quality for faster rendering
    
    # Configure 16:9 aspect ratio
    config.frame_height = 9
    config.frame_width = 16
    
    # Render the scene
    render_animation([SetCoverVisualization], Path("set_cover_visualization.py"))
```
```

## 10. Randomized Algorithms (QuickSort)
```
Create a Manim animation that renders directly in the current directory, not in a nested media folder.
The animation should visualize randomized QuickSort, demonstrating how random pivot selection helps avoid worst-case scenarios. Show the partitioning process around pivots and compare performance with deterministic pivot selection on different input arrays (sorted, reverse sorted, random).

For optimal visualization:
- Configure a 16:9 aspect ratio (config.frame_width = 16, config.frame_height = 9)
- Create a split-screen comparison between randomized and deterministic pivot selection
- Represent array elements as rectangles with heights proportional to values
- Highlight pivots with distinct colors and markers
- Animate the partitioning process with clear movement of elements
- Include counters for comparisons and swaps for each algorithm variant
- Create input array selection options (sorted, reverse sorted, random)
- Show a running time comparison visualization
- Add self-contained rendering code:

```python
if __name__ == "__main__":
    from manim.cli.render.commands import render_animation
    from pathlib import Path
    
    # Configure rendering
    config.media_dir = "."
    config.video_dir = "."
    config.images_dir = "."
    config.preview = True
    config.format = "mp4"
    config.quality = "l"  # low quality for faster rendering
    
    # Configure 16:9 aspect ratio
    config.frame_height = 9
    config.frame_width = 16
    
    # Render the scene
    render_animation([QuickSortVisualization], Path("quick_sort_visualization.py"))
```
```

## 11. Heaps and Priority Queues
```
Create a Manim animation that renders directly in the current directory, not in a nested media folder.
The animation should visualize heap operations (insertion, extract-min, heapify) on a binary min-heap. Show both the tree representation and the array representation of the heap, and demonstrate how these operations maintain the heap property through animations of element swaps and comparisons.

For optimal visualization:
- Configure a 16:9 aspect ratio (config.frame_width = 16, config.frame_height = 9)
- Position the tree representation in the upper half of the screen
- Show the array representation in the lower half, with indices clearly marked
- Create synchronized animations between tree and array views
- Use color highlighting to show parent-child relationships and comparisons
- Implement a step counter and operation type indicator
- Include visual verification of the heap property at each stage
- Add animated arrows to show element movements during operations
- Add self-contained rendering code:

```python
if __name__ == "__main__":
    from manim.cli.render.commands import render_animation
    from pathlib import Path
    
    # Configure rendering
    config.media_dir = "."
    config.video_dir = "."
    config.images_dir = "."
    config.preview = True
    config.format = "mp4"
    config.quality = "l"  # low quality for faster rendering
    
    # Configure 16:9 aspect ratio
    config.frame_height = 9
    config.frame_width = 16
    
    # Render the scene
    render_animation([HeapVisualization], Path("heap_visualization.py"))
```
```

## 12. Union-Find Data Structure
```
Create a Manim animation that renders directly in the current directory, not in a nested media folder.
The animation should demonstrate the Union-Find data structure with path compression and union by rank. Visualize the forest of trees representing the sets, and animate operations like MakeSet, Find, and Union. Show how these optimizations improve the performance of operations over time.

For optimal visualization:
- Configure a 16:9 aspect ratio (config.frame_width = 16, config.frame_height = 9)
- Create a large central area for the forest visualization
- Represent sets as trees with parent pointers as arrows
- Include a dedicated operation panel showing the current operation
- Create a split-screen comparison between optimized and unoptimized versions
- Implement an operation counter and tree height/rank display
- Use color coding to highlight roots, current paths, and merged sets
- Create before/after states for path compression visualization
- Add self-contained rendering code:

```python
if __name__ == "__main__":
    from manim.cli.render.commands import render_animation
    from pathlib import Path
    
    # Configure rendering
    config.media_dir = "."
    config.video_dir = "."
    config.images_dir = "."
    config.preview = True
    config.format = "mp4"
    config.quality = "l"  # low quality for faster rendering
    
    # Configure 16:9 aspect ratio
    config.frame_height = 9
    config.frame_width = 16
    
    # Render the scene
    render_animation([UnionFindVisualization], Path("union_find_visualization.py"))
```
```

## 13. Shortest Path Algorithms (Dijkstra)
```
Create a Manim animation that renders directly in the current directory, not in a nested media folder.
The animation should illustrate Dijkstra's algorithm for finding the shortest path in a weighted graph. Visualize the process of growing the tree of shortest paths from the source vertex, updating distance estimates for neighboring vertices, and selecting the vertex with the minimum distance at each step.

For optimal visualization:
- Configure a 16:9 aspect ratio (config.frame_width = 16, config.frame_height = 9)
- Create a large central graph with clearly labeled nodes and weighted edges
- Include a priority queue visualization showing distance estimates
- Implement a distance table that updates with each iteration
- Use color coding to indicate vertex states (unvisited, in queue, finalized)
- Show the growing shortest path tree with distinct edge styling
- Create animated path traceback from destination to source
- Add step counter and algorithm phase indicator
- Add self-contained rendering code:

```python
if __name__ == "__main__":
    from manim.cli.render.commands import render_animation
    from pathlib import Path
    
    # Configure rendering
    config.media_dir = "."
    config.video_dir = "."
    config.images_dir = "."
    config.preview = True
    config.format = "mp4"
    config.quality = "l"  # low quality for faster rendering
    
    # Configure 16:9 aspect ratio
    config.frame_height = 9
    config.frame_width = 16
    
    # Render the scene
    render_animation([DijkstraVisualization], Path("dijkstra_visualization.py"))
```
```

## 14. Sorting Algorithms Comparison
```
Create a Manim animation that renders directly in the current directory, not in a nested media folder.
The animation should compare multiple sorting algorithms (insertion sort, merge sort, quick sort) running on the same input array. Display the algorithms side by side, highlighting key operations and showing a running count of comparisons and swaps to visualize the different time complexities.

For optimal visualization:
- Configure a 16:9 aspect ratio (config.frame_width = 16, config.frame_height = 9)
- Create a three-way split screen for side-by-side algorithm comparison
- Represent array elements as rectangles with heights proportional to values
- Implement synchronized starting but independent progression for each algorithm
- Include operation counters (comparisons, swaps) for each algorithm
- Use consistent color coding across algorithms (e.g., for comparisons, swaps, sorted portions)
- Add a running time indicator to show relative speeds
- Create a final summary comparing actual performance vs. theoretical complexity
- Add self-contained rendering code:

```python
if __name__ == "__main__":
    from manim.cli.render.commands import render_animation
    from pathlib import Path
    
    # Configure rendering
    config.media_dir = "."
    config.video_dir = "."
    config.images_dir = "."
    config.preview = True
    config.format = "mp4"
    config.quality = "l"  # low quality for faster rendering
    
    # Configure 16:9 aspect ratio
    config.frame_height = 9
    config.frame_width = 16
    
    # Render the scene
    render_animation([SortingComparisonVisualization], Path("sorting_comparison_visualization.py"))
```
```

## 15. Graph Coloring Problem
```
Create a Manim animation that renders directly in the current directory, not in a nested media folder.
The animation should demonstrate a greedy approach to the graph coloring problem. Show a graph and animate the process of assigning colors to vertices one by one, ensuring that no adjacent vertices have the same color. Illustrate cases where the greedy approach does and doesn't produce an optimal coloring.

For optimal visualization:
- Configure a 16:9 aspect ratio (config.frame_width = 16, config.frame_height = 9)
- Create a split-screen comparison between two graph examples or algorithms
- Implement a color palette display showing available colors
- Use animation to highlight vertices being considered and their neighbors
- Include a step counter and coloring order explanation
- Create a chromatic number indicator showing optimal vs. achieved results
- Add a constraint checking visualization when assigning colors
- Show special graph types with known optimal colorings for comparison
- Add self-contained rendering code:

```python
if __name__ == "__main__":
    from manim.cli.render.commands import render_animation
    from pathlib import Path
    
    # Configure rendering
    config.media_dir = "."
    config.video_dir = "."
    config.images_dir = "."
    config.preview = True
    config.format = "mp4"
    config.quality = "l"  # low quality for faster rendering
    
    # Configure 16:9 aspect ratio
    config.frame_height = 9
    config.frame_width = 16
    
    # Render the scene
    render_animation([GraphColoringVisualization], Path("graph_coloring_visualization.py"))
```
```
These prompts should give you a solid foundation for creating Manim visualizations for each of the core concepts in the INFO 6205 course.
