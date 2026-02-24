# Quick Reference Guide - Integration Steps

## 📋 Step-by-Step Integration Checklist

### Phase 1: Setup (5 minutes)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Clone all repositories
python setup_benchmark.py

# 3. Test that minorminer works
python quick_start.py
```

**Expected result:** You should see minorminer successfully embedding graphs and generating plots in `./quick_start_results/`

---

### Phase 2: Understand Each Repository (15-30 min per repo)

```bash
# 1. Run the inspector
python inspect_repos.py
```

This will show you:
- ✓ Where to find documentation
- ✓ Main Python files
- ✓ Likely embedding functions
- ✓ Example usage files

**For each repository (ATOM, CHARME, OCT-Based):**

```bash
# 2. Read their README
cd implementations/atom  # (or charme, oct_based)
cat README.md

# 3. Look for examples
ls examples/
ls demos/
ls tests/

# 4. Find the main embedding function
# Look in the files that the inspector identified
```

---

### Phase 3: Implement Integration (30-60 min per method)

Open `embedding_integrations.py` and find these three functions:

#### 1️⃣ ATOM Integration

Find this function:
```python
def run_atom_embedding(source_graph, target_graph, timeout=60.0):
```

What you need to find in their repository:
- [ ] What is the main import? (e.g., `from atom import Embedder`)
- [ ] How do you create the embedder object?
- [ ] What format does it expect? (NetworkX graph? Adjacency dict?)
- [ ] How do you call the embedding function?
- [ ] What format does it return?

Fill in the section marked `# TODO: REPLACE THIS SECTION WITH ACTUAL ATOM CODE`

#### 2️⃣ CHARME Integration

Find this function:
```python
def run_charme_embedding(source_graph, target_graph, timeout=60.0):
```

**CHARME uses RL - Special considerations:**
- [ ] Does it need a pre-trained model? Where is it?
- [ ] How do you load the model?
- [ ] What is the inference API?
- [ ] Does it need a specific environment setup?

Fill in the section marked `# TODO: REPLACE THIS SECTION WITH ACTUAL CHARME CODE`

#### 3️⃣ OCT-Based Integration

Find this function:
```python
def run_oct_embedding(source_graph, target_graph, timeout=60.0):
```

Fill in the section marked `# TODO: REPLACE THIS SECTION WITH ACTUAL OCT CODE`

---

### Phase 4: Test Your Integration (5 min per method)

```bash
# Test all methods
python embedding_integrations.py

# Or test just one
python embedding_integrations.py --method atom
python embedding_integrations.py --method charme
python embedding_integrations.py --method oct_based
```

**What to look for:**
- ✓ Should say "Success!" if embedding worked
- ✓ Should show timing and chain count
- ✓ Should say "Valid embedding" if validation passed
- ✗ If it says "Not implemented", you need to edit more
- ✗ If it says "Import error", check Python paths

---

### Phase 5: Run Full Benchmark (30 min - 2 hours)

```bash
# Option A: Run examples
python complete_examples.py

# Option B: Write your own script
python -c "
from embedding_benchmark import EmbeddingBenchmark, create_chimera_graph
from embedding_integrations import integrate_all_methods

target = create_chimera_graph(4, 4, 4)
benchmark = EmbeddingBenchmark(target, results_dir='./final_results')
integrate_all_methods(benchmark)

problems = benchmark.generate_test_problems(
    sizes=[4, 6, 8, 10],
    densities=[0.3, 0.5, 0.7],
    instances_per_config=5
)

benchmark.run_full_benchmark(problems, timeout=60.0)
benchmark.generate_report()
"
```

---

## 🔍 Common Integration Patterns

### Pattern 1: Simple Function Call
```python
from method_module import find_embedding

start_time = time.time()
embedding = find_embedding(source_graph, target_graph)
elapsed_time = time.time() - start_time

return {
    'embedding': embedding,
    'time': elapsed_time
}
```

### Pattern 2: Class-Based API
```python
from method_module import Embedder

embedder = Embedder(target_graph)
start_time = time.time()
embedding = embedder.embed(source_graph)
elapsed_time = time.time() - start_time

return {
    'embedding': embedding,
    'time': elapsed_time
}
```

### Pattern 3: Needs Format Conversion
```python
from method_module import find_embedding

# Convert NetworkX to adjacency dict
source_adj = networkx_to_adjacency_dict(source_graph)
target_adj = networkx_to_adjacency_dict(target_graph)

start_time = time.time()
embedding = find_embedding(source_adj, target_adj)
elapsed_time = time.time() - start_time

return {
    'embedding': embedding,
    'time': elapsed_time
}
```

### Pattern 4: RL-Based (like CHARME)
```python
from method_module import Agent, load_model

# Load pre-trained model
agent = load_model('path/to/model.pth')

start_time = time.time()
embedding = agent.embed(source_graph, target_graph)
elapsed_time = time.time() - start_time

return {
    'embedding': embedding,
    'time': elapsed_time
}
```

---

## 🐛 Troubleshooting

### Problem: "Import error: No module named 'xxx'"

**Solution 1:** Add the repo to Python path
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path('./implementations/atom')))
```

**Solution 2:** Install the package
```bash
cd implementations/atom
pip install -e .
```

---

### Problem: "Method returns None"

**Causes:**
1. You haven't filled in the implementation yet (it's still a placeholder)
2. The method is crashing but the error is caught
3. The method genuinely couldn't find an embedding

**Debug:**
- Add print statements in the function
- Remove the try/except temporarily to see actual errors
- Test the method independently outside the benchmark

---

### Problem: "Invalid embedding"

**Causes:**
1. Method returned embedding in wrong format
2. Embedding actually is invalid

**Debug:**
```python
# Print the embedding to see its structure
print(f"Embedding type: {type(embedding)}")
print(f"Embedding: {embedding}")

# Check format
for node, chain in embedding.items():
    print(f"Node {node} -> Chain {chain}")
```

---

### Problem: "Method is too slow"

**Solutions:**
1. Increase timeout: `timeout=120.0`
2. Test on smaller problems first
3. Check if method needs special configuration for speed
4. Some methods might need warm-up/pre-training

---

## 📊 Understanding the Results

After running the benchmark, you'll get:

### CSV Files
- `results.csv` - Every single embedding attempt with all metrics
- `summary_statistics.csv` - Aggregated stats per method

**Key columns:**
- `success`: Did it find an embedding?
- `embedding_time`: How long did it take?
- `avg_chain_length`: Average chain length (lower = better)
- `max_chain_length`: Longest chain (lower = better)
- `total_qubits_used`: How many physical qubits

### Plots
- `success_rates.png` - Which method is most reliable?
- `embedding_times.png` - Which method is fastest?
- `chain_lengths.png` - Which method produces best quality?
- `scalability.png` - How do methods scale with problem size?

---

## 🎯 Success Criteria

Your integration is successful when:

✅ `python embedding_integrations.py` shows "Success!" for all methods
✅ `test_integration()` shows valid embeddings
✅ Full benchmark runs without crashes
✅ You get comparison plots showing all four methods
✅ Results CSV contains data for all methods

---

## 💡 Pro Tips

1. **Start with ATOM** - Usually the most straightforward to integrate
2. **Test incrementally** - Don't integrate all three at once
3. **Use their examples** - Best way to understand the API
4. **Check their tests** - Often shows exact usage
5. **Read function docstrings** - May have better docs than README
6. **Start small** - Test on K4 graph before full benchmark
7. **Compare to minorminer** - Already works, use as reference

---

## 📞 Need Help?

If stuck on a specific repository:
1. Open an issue on their GitHub
2. Check their documentation more carefully
3. Look for a `examples/` or `demo/` directory
4. Search their issues for similar questions
5. Email the authors (often in README)

Remember: The hard part is understanding their API. The benchmark framework handles everything else!
