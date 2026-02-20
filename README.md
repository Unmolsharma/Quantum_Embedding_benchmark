# Minor Embedding Benchmarking Framework

A comprehensive, extensible benchmarking framework for comparing quantum annealing minor embedding algorithms on D-Wave hardware topologies.

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start (minorminer only)](#quick-start-minorminer-only)
- [Full Setup (All 4 Methods)](#full-setup-all-4-methods)
- [Understanding the Test Graphs](#understanding-the-test-graphs)
- [Where to Find Your Results](#where-to-find-your-results)
- [Adding Your Own Algorithm](#adding-your-own-algorithm)
- [How It Works](#how-it-works)
- [Metrics Explained](#metrics-explained)
- [Troubleshooting](#troubleshooting)

---

## Overview

This framework benchmarks minor embedding algorithms across multiple dimensions:

**Algorithms Supported:**
- ✅ **minorminer** - D-Wave's heuristic algorithm (ready to use)
- ⚙️ **ATOM** - Adaptive topology embedding (requires compilation)
- ⚙️ **CHARME** - RL-based embedding (requires compilation)
- ⚙️ **OCT-Based** - Virtual embedding via odd cycle transversal (requires compilation)

**What It Measures:**
- ⏱️ **Speed** - Embedding time in seconds
- ✅ **Reliability** - Success rate (% of problems solved)
- 📊 **Quality** - Average and maximum chain lengths
- 💾 **Efficiency** - Total qubits and couplers used
- 📈 **Scalability** - Performance vs problem size

**Hardware Topologies:**
- Chimera (D-Wave 2000Q) - default
- Pegasus (D-Wave Advantage) - extensible

---

## Quick Start (minorminer only)

**No C++ compiler needed! Works immediately on any system.**

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/embedding-benchmark.git
cd embedding-benchmark
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

**What gets installed:**
- `networkx` - Graph operations
- `minorminer` - D-Wave's embedding algorithm
- `dwave-networkx` - Hardware graph topologies
- `pandas`, `matplotlib`, `seaborn` - Analysis and visualization
- `numpy`, `scipy` - Numerical operations

### Step 3: Run the Benchmark

```bash
python quick_start.py
```

**What happens:**
1. Creates a Chimera 4×4 target graph (128 qubits)
2. Generates ~50 test problems (random + structured graphs)
3. Runs minorminer on each problem
4. Saves results to `./quick_start_results/`
5. Generates plots and statistics

**Time:** ~5-10 minutes

### Step 4: Check Your Results

```bash
cd quick_start_results
ls
```

You'll see:
- `results.csv` - All embedding attempts with metrics
- `summary_statistics.csv` - Aggregated performance by method
- `success_rates.png` - Bar chart of success rates
- `embedding_times.png` - Box plot of execution times
- `chain_lengths.png` - Chain quality distributions
- `scalability.png` - Time vs problem size

**Open the PNG files to see your results!**

---

## Full Setup (All 4 Methods)

**Requires:** C++ compiler (g++, MinGW, or Visual Studio)

### Prerequisites

**Linux/Mac:**
```bash
# Already have g++ usually
g++ --version
```

**Windows:**
Install MinGW or MSYS2:
- Download MSYS2: https://www.msys2.org/
- Install g++: `pacman -S mingw-w64-x86_64-gcc`

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Clone the C++ Repositories

```bash
python setup_benchmark.py
```

This clones:
- ATOM → `implementations/atom/`
- CHARME → `implementations/charme/`
- OCT-Based → `implementations/oct_based/`

### Step 3: Compile Each Method

```bash
# ATOM
cd implementations/atom
make
cd ../..

# CHARME
cd implementations/charme
make
cd ../..

# OCT-Based
cd implementations/oct_based
make
cd ../..
```

### Step 4: Test the Integrations

```bash
python embedding_integrations.py
```

Expected output:
```
Testing ATOM...
  Status: ✓ Success!
  Time: 0.234s
  Chains: 4
  Validation: ✓ Valid

Testing CHARME...
  Status: ✓ Success!
  ...

Testing OCT-Based...
  Status: ✓ Success!
  ...
```

### Step 5: Run Full Benchmark

```bash
python complete_examples.py
```

Choose option 3: "Benchmark all methods"

Results saved to `./example_3_results/`

---

## Understanding the Test Graphs

The benchmark generates two types of test problems:

### Random Graphs

**Generated via:** `nx.gnp_random_graph(n, p)`

- **Purpose:** Test general performance across varying problem characteristics
- **Parameters:**
  - `n` = number of nodes (sizes: 4, 6, 8, 10, 12)
  - `p` = edge probability (densities: 0.3, 0.5, 0.7)
  - Multiple random instances per configuration for statistical validity

**Example:** `random_n8_d0.50_i2`
- 8 nodes
- 50% edge density
- Instance #2

### Structured Graphs

**1. Complete Graphs (K_n)**
- Every node connects to every other node
- **Why:** Hardest to embed - tests worst-case performance
- **Examples:** K4, K6, K8, K10

**2. Grid Graphs**
- 2D lattice structure
- **Why:** Maps naturally to Chimera topology - tests topology awareness
- **Examples:** 3×3, 4×4, 5×5

**3. Cycle Graphs**
- Nodes in a ring
- **Why:** Tests chain formation on simple structures
- **Examples:** 5-node, 10-node, 15-node, 20-node cycles

**4. Tree Graphs**
- Balanced binary trees
- **Why:** Tests hierarchical problem structures
- **Examples:** depth-3, depth-4

### Why This Mix?

- **Random graphs** → General algorithm behavior
- **Complete graphs** → Worst-case stress testing
- **Grids** → Real-world QUBO problems often have grid-like structure
- **Cycles/Trees** → Specific topology characteristics

### Customizing Test Problems

```python
from embedding_benchmark import EmbeddingBenchmark, create_chimera_graph
import networkx as nx

target = create_chimera_graph(4, 4, 4)
benchmark = EmbeddingBenchmark(target)

# Use built-in generator
problems = benchmark.generate_test_problems(
    sizes=[6, 8, 10],
    densities=[0.4, 0.6],
    instances_per_config=3
)

# Or create your own
custom_problems = [
    ("my_graph", nx.karate_club_graph()),
    ("my_grid", nx.grid_2d_graph(5, 5)),
]

benchmark.run_full_benchmark(custom_problems, timeout=60.0)
```

---

## Where to Find Your Results

### Results Directory Structure

```
./quick_start_results/           # or ./example_3_results/, etc.
├── results.json                 # Raw data (JSON format)
├── results.csv                  # Raw data (spreadsheet format)
├── summary_statistics.csv       # Aggregated metrics
├── success_rates.png            # Success rate comparison
├── embedding_times.png          # Speed comparison
├── chain_lengths.png            # Quality comparison
└── scalability.png              # Scaling analysis
```

### Understanding the CSV Files

#### `results.csv`

Every row = one embedding attempt

**Key Columns:**
- `method_name` - Which algorithm (minorminer, atom, charme, oct_based)
- `problem_name` - Which test graph
- `problem_size` - Number of nodes
- `problem_density` - Edge density
- `success` - True/False (did it find an embedding?)
- `embedding_time` - Seconds to complete
- `avg_chain_length` - Average chain length (lower = better)
- `max_chain_length` - Longest chain (lower = better)
- `total_qubits_used` - Physical qubits needed
- `total_couplers_used` - Physical couplers needed

**Example row:**
```csv
minorminer,complete_K4,4,1.0,True,0.023,1.25,2,5,8
```
→ minorminer solved K4 in 0.023s with avg chain length 1.25

#### `summary_statistics.csv`

One row per method with aggregated stats:

**Columns:**
- `Method` - Algorithm name
- `Total Runs` - Number of problems attempted
- `Successful` - Number solved
- `Success Rate (%)` - Percentage solved
- `Avg Time (s)` - Mean embedding time (successful only)
- `Std Time (s)` - Standard deviation
- `Avg Chain Length` - Mean chain length
- `Avg Max Chain` - Mean of maximum chains
- `Avg Qubits Used` - Mean physical qubits

**Use this for:** Quick comparison between methods

### Interpreting the Plots

#### `success_rates.png`
- **Y-axis:** Success rate (0-100%)
- **X-axis:** Method name
- **Higher is better** - more reliable algorithm

#### `embedding_times.png`
- **Y-axis:** Time in seconds (log scale often)
- **Box plot** showing distribution
- **Lower is better** - faster algorithm
- **Tighter boxes** = more consistent

#### `chain_lengths.png`
- **Two subplots:** Average and Maximum chain lengths
- **Lower is better** - less noise on real quantum hardware
- **Key metric** for actual quantum annealing performance

#### `scalability.png`
- **X-axis:** Problem size (nodes)
- **Y-axis:** Time (seconds)
- **Flat line** = excellent scaling
- **Steep curve** = poor scaling

### Opening and Analyzing Results

**In Excel/Google Sheets:**
```bash
# Open CSV
start results.csv  # Windows
open results.csv   # Mac
xdg-open results.csv  # Linux
```

**In Python:**
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('quick_start_results/results.csv')

# Filter to successful embeddings
success_df = df[df['success'] == True]

# Compare methods
print(success_df.groupby('method_name')['avg_chain_length'].mean())

# Custom plot
success_df.boxplot(column='embedding_time', by='method_name')
plt.show()
```

**In R:**
```r
library(ggplot2)
df <- read.csv('quick_start_results/results.csv')
ggplot(df, aes(x=method_name, y=avg_chain_length)) + geom_boxplot()
```

---

## Adding Your Own Algorithm

Want to benchmark your own embedding algorithm? Here's the complete guide.

### Overview

To add your algorithm, you need to:
1. Write an embedding function
2. Add it to `embedding_integrations.py`
3. Register it in the integration system
4. Add a placeholder in `embedding_benchmark.py`
5. Test and run

### Step-by-Step: Python Algorithm

#### Step 1: Write Your Embedding Function

Create `my_algorithm.py`:

```python
import time
import networkx as nx
from typing import Dict, Optional

def my_embedding_algorithm(source_graph: nx.Graph, 
                           target_graph: nx.Graph, 
                           timeout: float = 60.0) -> Optional[Dict]:
    """
    Your embedding algorithm
    
    Args:
        source_graph: NetworkX graph to embed (logical graph)
        target_graph: NetworkX hardware graph (Chimera/Pegasus)
        timeout: Maximum time in seconds
        
    Returns:
        dict: {
            'embedding': {logical_node: [physical_node1, physical_node2, ...]},
            'time': elapsed_seconds
        }
        or None if failed
    """
    start_time = time.time()
    
    # YOUR ALGORITHM HERE
    # Example: Simple 1-to-1 mapping (won't work for most cases)
    embedding = {}
    target_nodes = list(target_graph.nodes())
    
    for i, node in enumerate(source_graph.nodes()):
        if i < len(target_nodes):
            # Map each source node to a chain (list of target nodes)
            embedding[node] = [target_nodes[i]]  # Chain of length 1
        else:
            # Not enough target nodes
            return None
    
    # Check timeout
    if time.time() - start_time > timeout:
        return None
    
    elapsed_time = time.time() - start_time
    
    return {
        'embedding': embedding,
        'time': elapsed_time
    }
```

#### Step 2: Add Integration to `embedding_integrations.py`

Open `embedding_integrations.py` and add these parts:

**At the top (around line 10):**
```python
# Add import
from my_algorithm import my_embedding_algorithm
```

**Add new function (around line 250, after `run_oct_embedding`):**
```python
def run_my_algorithm(source_graph: nx.Graph, target_graph: nx.Graph,
                     timeout: float = 60.0) -> Optional[Dict]:
    """Run my custom embedding algorithm"""
    try:
        result = my_embedding_algorithm(source_graph, target_graph, timeout)
        
        if result is None or 'embedding' not in result:
            return None
        
        # Validate the embedding
        if not validate_embedding(result['embedding'], source_graph, target_graph):
            print("My algorithm: Invalid embedding")
            return None
        
        return result
        
    except Exception as e:
        print(f"My algorithm error: {e}")
        return None
```

**Update `integrate_all_methods()` function (around line 350):**
```python
def integrate_all_methods(benchmark_instance):
    """Integrate all methods into a benchmark instance"""
    
    def atom_wrapper(source_graph, timeout=60.0):
        return run_atom_embedding(source_graph, benchmark_instance.target_graph, timeout)
    
    def charme_wrapper(source_graph, timeout=60.0):
        return run_charme_embedding(source_graph, benchmark_instance.target_graph, timeout)
    
    def oct_wrapper(source_graph, timeout=60.0):
        return run_oct_embedding(source_graph, benchmark_instance.target_graph, timeout)
    
    # ADD THIS:
    def my_algorithm_wrapper(source_graph, timeout=60.0):
        return run_my_algorithm(source_graph, benchmark_instance.target_graph, timeout)
    
    benchmark_instance.run_atom = atom_wrapper
    benchmark_instance.run_charme = charme_wrapper
    benchmark_instance.run_oct_based = oct_wrapper
    benchmark_instance.run_my_algorithm = my_algorithm_wrapper  # ADD THIS
    
    print("✓ Integration wrappers installed")
```

**Update `test_integration()` function (around line 370):**
```python
def test_integration(method_name: str = "all"):
    """Test integration of one or all methods"""
    # ... existing code ...
    
    methods = {
        'atom': run_atom_embedding,
        'charme': run_charme_embedding,
        'oct_based': run_oct_embedding,
        'my_algorithm': run_my_algorithm  # ADD THIS
    }
    
    # ... rest of function ...
```

#### Step 3: Add Placeholder to `embedding_benchmark.py`

Open `embedding_benchmark.py` and add (around line 150, after `run_oct_based`):

```python
def run_my_algorithm(self, source_graph: nx.Graph, 
                     timeout: float = 60.0) -> Optional[Dict]:
    """Run my custom algorithm (placeholder)"""
    return None
```

**Update `run_full_benchmark()` method (around line 360):**
```python
def run_full_benchmark(self, problems, timeout=60.0, methods=None):
    """Run complete benchmark suite"""
    
    if methods is None:
        # ADD 'my_algorithm' to this list:
        methods = ['minorminer', 'atom', 'charme', 'oct_based', 'my_algorithm']
    
    method_map = {
        'minorminer': self.run_minorminer,
        'atom': self.run_atom,
        'charme': self.run_charme,
        'oct_based': self.run_oct_based,
        'my_algorithm': self.run_my_algorithm  # ADD THIS
    }
    
    # ... rest of function ...
```

#### Step 4: Test Your Integration

```bash
python embedding_integrations.py --method my_algorithm
```

Expected output:
```
Testing MY_ALGORITHM...
  Status: ✓ Success!
  Time: 0.012s
  Chains: 4
  Validation: ✓ Valid
```

#### Step 5: Run Benchmark Comparison

Create `test_my_algorithm.py`:

```python
from embedding_benchmark import EmbeddingBenchmark, create_chimera_graph
from embedding_integrations import integrate_all_methods

# Setup
target = create_chimera_graph(4, 4, 4)
benchmark = EmbeddingBenchmark(target, results_dir="./my_algorithm_results")

# Integrate (includes your algorithm)
integrate_all_methods(benchmark)

# Generate test problems
problems = benchmark.generate_test_problems(
    sizes=[4, 6, 8],
    densities=[0.3, 0.5],
    instances_per_config=3
)

# Compare your algorithm with minorminer
benchmark.run_full_benchmark(
    problems,
    timeout=60.0,
    methods=['minorminer', 'my_algorithm']  # Just these two
)

# Generate plots
benchmark.generate_report()

print("\n✓ Results in ./my_algorithm_results/")
```

Run it:
```bash
python test_my_algorithm.py
```

Results will show minorminer vs your algorithm!

### Step-by-Step: C++ Algorithm

If your algorithm is C++, follow the ATOM pattern:

#### Step 1: Your C++ Program Interface

Your program should accept:
```bash
./my_embedder source.txt target.txt output.txt
```

**Input format (`source.txt` and `target.txt`):**
```
num_nodes num_edges
node1 node2
node3 node4
...
```

**Output format (`output.txt`):**
```
logical_node physical_node1 physical_node2 ...
0 5 6
1 7 8 9
2 10
```
Each line = one logical node mapped to its chain of physical nodes

#### Step 2: Integration Function

In `embedding_integrations.py`, add:

```python
def run_my_cpp_algorithm(source_graph: nx.Graph, target_graph: nx.Graph,
                         timeout: float = 60.0) -> Optional[Dict]:
    """Run my C++ algorithm via subprocess"""
    try:
        import subprocess
        import tempfile
        import os
        
        # Find your executable
        my_dir = Path("./my_algorithm")
        my_exe = my_dir / "my_embedder"
        if not my_exe.exists():
            my_exe = my_dir / "my_embedder.exe"
        
        if not my_exe.exists():
            print("⚠️ My C++ algorithm not compiled")
            return None
        
        # Write graphs to temp files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(f"{source_graph.number_of_nodes()} {source_graph.number_of_edges()}\n")
            for u, v in source_graph.edges():
                f.write(f"{u} {v}\n")
            source_file = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(f"{target_graph.number_of_nodes()} {target_graph.number_of_edges()}\n")
            for u, v in target_graph.edges():
                f.write(f"{u} {v}\n")
            target_file = f.name
        
        output_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False).name
        
        start_time = time.time()
        
        try:
            # Call your C++ program
            result = subprocess.run(
                [str(my_exe), source_file, target_file, output_file],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=my_dir
            )
            
            elapsed_time = time.time() - start_time
            
            # Parse output
            if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                return None
            
            embedding = {}
            with open(output_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        logical_node = int(parts[0])
                        physical_chain = [int(x) for x in parts[1:]]
                        embedding[logical_node] = physical_chain
            
            # Cleanup temp files
            os.unlink(source_file)
            os.unlink(target_file)
            os.unlink(output_file)
            
            # Validate
            if not embedding or not validate_embedding(embedding, source_graph, target_graph):
                return None
            
            return {'embedding': embedding, 'time': elapsed_time}
            
        except subprocess.TimeoutExpired:
            os.unlink(source_file)
            os.unlink(target_file)
            if os.path.exists(output_file):
                os.unlink(output_file)
            return None
        
    except Exception as e:
        print(f"My C++ algorithm error: {e}")
        return None
```

Then follow Steps 2-5 from the Python algorithm section.

### Quick Reference: Files to Edit

| File | What to Add | Where |
|------|-------------|-------|
| `my_algorithm.py` | Your algorithm implementation | New file |
| `embedding_integrations.py` | `run_my_algorithm()` function | After line 250 |
| `embedding_integrations.py` | Wrapper in `integrate_all_methods()` | Around line 350 |
| `embedding_integrations.py` | Method in `test_integration()` | Around line 370 |
| `embedding_benchmark.py` | Placeholder `run_my_algorithm()` | After line 150 |
| `embedding_benchmark.py` | Add to `methods` list | Line 360 |
| `embedding_benchmark.py` | Add to `method_map` | Line 365 |

### Testing Checklist

- [ ] `python embedding_integrations.py --method my_algorithm` shows success
- [ ] Creates valid embedding (validation passes)
- [ ] Returns timing information
- [ ] Works on test graphs (K4, grids, etc.)
- [ ] Integrated into benchmark framework
- [ ] Generates comparison plots

---

## How It Works

### Architecture

```
User Script (quick_start.py, etc.)
         ↓
embedding_benchmark.py
    ├── Generates test problems
    ├── Calls each method
    ├── Collects metrics
    └── Creates visualizations
         ↓
embedding_integrations.py
    ├── Wraps each algorithm
    ├── Standardizes I/O
    └── Validates results
         ↓
    Algorithms (minorminer, ATOM, CHARME, OCT, yours)
```

### Workflow

1. **Generate Test Suite**
   - Random graphs (varying size/density)
   - Structured graphs (complete, grid, cycle, tree)

2. **For Each Problem:**
   ```
   For each method:
     - Call embedding function
     - Measure time
     - Validate result
     - Compute metrics
     - Store EmbeddingResult
   ```

3. **Analysis**
   - Aggregate results by method
   - Compute statistics
   - Generate plots
   - Save CSV/JSON

### Key Design Pattern: Dependency Injection

The integration system uses dependency injection to keep code modular:

```python
# Before integration
benchmark.run_my_algorithm()  # Returns None

# After integration
integrate_all_methods(benchmark)
benchmark.run_my_algorithm()  # Calls your actual implementation!
```

This means:
- `embedding_benchmark.py` stays clean
- Easy to add new algorithms
- No need to modify core framework

---

## Metrics Explained

### Success Rate
- **Definition:** % of problems solved within timeout
- **Range:** 0-100%
- **Higher = better**
- **Why:** Reliability indicator

### Embedding Time
- **Definition:** Seconds to find embedding
- **Range:** 0 to timeout
- **Lower = better**
- **Why:** Speed matters for large-scale use

### Average Chain Length
- **Definition:** Mean physical qubits per logical qubit
- **Formula:** `total_chain_nodes / num_logical_qubits`
- **Range:** 1.0 (perfect) to ∞
- **Lower = better**
- **Why:** **Most important!** Directly affects error rates on real hardware

### Maximum Chain Length
- **Definition:** Longest single chain
- **Range:** 1 to ∞
- **Lower = better**
- **Why:** Bottleneck for coupling strength and errors

### Total Qubits Used
- **Definition:** Physical qubits in embedding
- **Lower = better**
- **Why:** Hardware efficiency

### Total Couplers Used
- **Definition:** Physical edges in embedding
- **Lower = better**
- **Why:** Limited coupler budget

---

## Troubleshooting

### "ImportError: cannot import name 'complete_to_chordal_graph'"
**Fix:** Reinstall NetworkX
```bash
pip uninstall networkx -y
pip install networkx==3.1
```

### "ATOM not compiled"
**Fix:**
```bash
cd implementations/atom
make
```

### "No module named 'minorminer'"
**Fix:**
```bash
pip install minorminer dwave-networkx
```

### Windows: "g++: command not found"
**Fix:** Install MinGW/MSYS2 or just use minorminer only (no compiler needed)

### No plots generated
**Check:** Did any embeddings succeed?
```python
import pandas as pd
df = pd.read_csv('quick_start_results/results.csv')
print(df['success'].sum())  # Should be > 0
```

---

## Citation

```bibtex
@software{embedding_benchmark_2024,
  author = {Your Name},
  title = {Minor Embedding Benchmarking Framework},
  year = {2024},
  url = {https://github.com/yourusername/embedding-benchmark}
}
```

---

## License

MIT License

---

## Support

- **Issues:** GitHub Issues
- **Questions:** GitHub Discussions

---

## Acknowledgments

- D-Wave Systems for minorminer
- Authors of ATOM, CHARME, and OCT-Based
- Quantum computing community