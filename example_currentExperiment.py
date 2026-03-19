import time
import heapq
import random
import numpy as np
import networkx as nx
from typing import Dict, Any

from qebench.registry import EmbeddingAlgorithm, register_algorithm
from qebench import EmbeddingBenchmark
from qeanalysis import BenchmarkAnalysis

# 1. Define and Register the Algorithm
@register_algorithm("gf_bolt")
class GFBoltAlgorithm(EmbeddingAlgorithm):
    """GF-Bolt heuristic embedding algorithm with conditional stochastic tie-breaking."""

    def embed(self, source_graph: Any, target_graph: Any, timeout: float = 60.0, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        
        seed = kwargs.get('seed', None)
        if seed is not None:
            random.seed(seed)
            
        S_nodes = list(source_graph.nodes())
        T_nodes = list(target_graph.nodes())
        
        if not S_nodes or not T_nodes:
            raise ValueError("Empty source or target graph provided.")
            
        T_avg_deg = max(1.0, sum(dict(target_graph.degree()).values()) / len(T_nodes))
        D = {v: source_graph.degree(v) / T_avg_deg for v in S_nodes}
        D_max = max(D.values()) if D else 1.0
        if D_max == 0:
            D_max = 1.0

        phi = {v: [] for v in S_nodes}
        h = {q: 0.0 for q in T_nodes}
        
        # Hyperparameters
        h_inc = kwargs.get('h_inc', 1.0)
        h_decay = kwargs.get('h_decay', 0.95)
        p_fac = kwargs.get('p_fac', 1.0)
        max_iterations = kwargs.get('max_iterations', 100)
        alpha_scale = kwargs.get('alpha_scale', 0.5)
        patience_limit = kwargs.get('patience', 20)
        
        stage = 1
        prev_sum_chains = float('inf')
        prev_max_overlap = float('inf')
        current_patience = 0
        
        while stage <= max_iterations:
            if time.time() - start_time > timeout:
                print(f"\n[DEBUG] Timeout.")
                raise TimeoutError(f"Timeout of {timeout}s exceeded at stage {stage}.")
            
            # Conditional tie-breaking: Only inject chaos if stuck
            if current_patience > 0:
                random.shuffle(S_nodes)
            else:
                S_nodes.sort(key=lambda x: x)  # Stable deterministic baseline
                
            order = sorted(S_nodes, key=lambda v: D[v], reverse=True)
                
            for v_i in order:
                w = {}
                alpha = alpha_scale * (D[v_i] / D_max)
                
                for q in T_nodes:
                    overlap_count = sum(1 for v_j in S_nodes if v_j != v_i and q in phi[v_j])
                    p_q = 1.0 + overlap_count * p_fac
                    base_q = 1.0
                    w[q] = alpha * base_q + (1.0 - alpha) * (base_q + h[q]) * p_q

                neighbors = list(source_graph.neighbors(v_i))
                embedded_neighbor_chains = [phi[u] for u in neighbors if phi[u]]
                
                try:
                    new_chain = self._find_minimal_vertex_model(target_graph, w, embedded_neighbor_chains, T_nodes)
                    phi[v_i] = new_chain
                except ValueError as e:
                    print(f"\n[DEBUG] Pathfinding failed for node {v_i} at stage {stage}: {e}")
                    raise RuntimeError(f"Disconnected target or pathing failure at node {v_i}. {e}")
            
            max_overlap = 0
            for q in T_nodes:
                count = sum(1 for v in S_nodes if q in phi[v])
                max_overlap = max(max_overlap, count)
                
            sum_chains = sum(len(chain) for chain in phi.values())
            
            # Successful embedding found, execute chain shortening
            if max_overlap <= 1:
                phi = self._shorten_chains(source_graph, target_graph, phi)
                clean_phi = {int(k): [int(q) for q in v] for k, v in phi.items()}
                return {
                    'embedding': clean_phi,
                    'time': time.time() - start_time
                }
            
            is_improving = (max_overlap < prev_max_overlap) or (sum_chains < prev_sum_chains)
            if is_improving:
                current_patience = 0
            else:
                current_patience += 1
                
            if current_patience >= patience_limit and stage > 2:
                print(f"\n[DEBUG] Max patience ({patience_limit}) reached at stage {stage}. Overlap: {max_overlap}, Chains: {sum_chains}")
                raise RuntimeError(f"Stuck in local minimum. Overlap remains {max_overlap}.")
                
            prev_max_overlap = max_overlap
            prev_sum_chains = sum_chains
            
            for q in T_nodes:
                h[q] *= h_decay
                if sum(1 for v in S_nodes if q in phi[v]) > 1:
                    h[q] += h_inc
                    
            stage += 1
            
        print(f"\n[DEBUG] Max iterations reached. Failed Embedding State:")
        raise RuntimeError(f"Max iterations ({max_iterations}) reached without resolving overlap.")

    def _shorten_chains(self, source_graph, target_graph, phi):
        """Greedily strips redundant qubits from the final embedding."""
        for v in source_graph.nodes():
            chain = list(phi[v])
            if len(chain) <= 1:
                continue
                
            neighbors = list(source_graph.neighbors(v))
            changed = True
            
            while changed:
                changed = False
                for q in chain:
                    candidate_chain = [x for x in chain if x != q]
                    
                    if not candidate_chain:
                        continue
                        
                    # 1. Check if the shortened chain remains connected
                    if not self._is_connected(target_graph, candidate_chain):
                        continue
                        
                    # 2. Check if it maintains all necessary logical edges
                    valid_connections = True
                    for u in neighbors:
                        connected_to_u = False
                        for c_node in candidate_chain:
                            for u_node in phi[u]:
                                if target_graph.has_edge(c_node, u_node):
                                    connected_to_u = True
                                    break
                            if connected_to_u:
                                break
                        if not connected_to_u:
                            valid_connections = False
                            break
                            
                    if valid_connections:
                        chain = candidate_chain
                        changed = True
                        break
            
            phi[v] = chain
        return phi

    def _is_connected(self, target_graph, chain):
        """Validates if a list of qubits forms a connected subgraph."""
        if not chain:
            return False
        visited = set()
        stack = [chain[0]]
        chain_set = set(chain)
        
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                for neighbor in target_graph.neighbors(node):
                    if neighbor in chain_set and neighbor not in visited:
                        stack.append(neighbor)
                        
        return len(visited) == len(chain)

    def _find_minimal_vertex_model(self, T, w, neighbor_chains, T_nodes):
        if not neighbor_chains:
            best_q = min(T_nodes, key=lambda q: w[q])
            return [best_q]
            
        distances = []
        predecessors = []
        
        for chain in neighbor_chains:
            dist, pred = self._node_weighted_dijkstra(T, chain, w)
            distances.append(dist)
            predecessors.append(pred)
            
        best_g = None
        min_total_dist = float('inf')
        
        for q in T_nodes:
            total_dist = sum(d.get(q, float('inf')) for d in distances)
                
            if total_dist < min_total_dist:
                min_total_dist = total_dist
                best_g = q
                
        if best_g is None or min_total_dist == float('inf'):
            raise ValueError("No valid Steiner root found.")
            
        final_chain = set([best_g])
        for pred in predecessors:
            curr = best_g
            while curr in pred and pred[curr] is not None:
                curr = pred[curr]
                final_chain.add(curr)
                
        return list(final_chain)

    def _node_weighted_dijkstra(self, T, sources, w):
        dist = {}
        pred = {}
        pq = []
        
        perimeter = set()
        for s in sources:
            for v in T.neighbors(s):
                if v not in sources:
                    perimeter.add(v)
                    
        if not perimeter:
            perimeter = set(sources)
                
        for u in perimeter:
            dist[u] = w[u]
            pred[u] = None
            heapq.heappush(pq, (w[u], u))
            
        while pq:
            d, u = heapq.heappop(pq)
            
            if d > dist.get(u, float('inf')):
                continue
                
            for v in T.neighbors(u):
                alt = d + w[v]
                if alt < dist.get(v, float('inf')):
                    dist[v] = alt
                    pred[v] = u
                    heapq.heappush(pq, (alt, v))
                    
        return dist, pred
    

@register_algorithm("gf_bolt_new")
class GFBoltAlgorithm(EmbeddingAlgorithm):
    """GF-Bolt heuristic embedding algorithm with conditional stochastic tie-breaking
    and a p_fac slow-start schedule."""

    def embed(self, source_graph: Any, target_graph: Any, timeout: float = 60.0, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        
        seed = kwargs.get('seed', None)
        if seed is not None:
            random.seed(seed)
            
        S_nodes = list(source_graph.nodes())
        T_nodes = list(target_graph.nodes())
        
        if not S_nodes or not T_nodes:
            raise ValueError("Empty source or target graph provided.")
            
        T_avg_deg = max(1.0, sum(dict(target_graph.degree()).values()) / len(T_nodes))
        D = {v: source_graph.degree(v) / T_avg_deg for v in S_nodes}
        D_max = max(D.values()) if D else 1.0
        if D_max == 0:
            D_max = 1.0

        phi = {v: [] for v in S_nodes}
        h = {q: 0.0 for q in T_nodes}
        
        # Hyperparameters
        h_inc = kwargs.get('h_inc', 1.0)
        h_decay = kwargs.get('h_decay', 0.95)
        p_fac_max = kwargs.get('p_fac', 1.0)        # target p_fac after ramp
        p_fac_start = kwargs.get('p_fac_start', 0.05) # initial p_fac (overlap cheap)
        p_fac_ramp_stages = kwargs.get('p_fac_ramp_stages', 10) # stages to reach p_fac_max
        max_iterations = kwargs.get('max_iterations', 100)
        alpha_scale = kwargs.get('alpha_scale', 0.5)
        patience_limit = kwargs.get('patience', 20)
        
        stage = 1
        prev_sum_chains = float('inf')
        prev_max_overlap = float('inf')
        current_patience = 0
        
        while stage <= max_iterations:
            if time.time() - start_time > timeout:
                print(f"\n[DEBUG] Timeout. Failed Embedding State: {phi}")
                raise TimeoutError(f"Timeout of {timeout}s exceeded at stage {stage}.")
            
            # p_fac schedule: linear ramp from p_fac_start to p_fac_max
            # over p_fac_ramp_stages iterations, then held constant
            ramp_t = min(stage - 1, p_fac_ramp_stages) / p_fac_ramp_stages
            p_fac = p_fac_start + (p_fac_max - p_fac_start) * ramp_t

            # Conditional tie-breaking: Only inject chaos if stuck
            if current_patience > 0:
                random.shuffle(S_nodes)
            else:
                S_nodes.sort(key=lambda x: x)
                
            order = sorted(S_nodes, key=lambda v: D[v], reverse=True)
                
            for v_i in order:
                w = {}
                alpha = alpha_scale * (D[v_i] / D_max)
                
                for q in T_nodes:
                    overlap_count = sum(1 for v_j in S_nodes if v_j != v_i and q in phi[v_j])
                    p_q = 1.0 + overlap_count * p_fac
                    base_q = 1.0
                    w[q] = alpha * base_q + (1.0 - alpha) * (base_q + h[q]) * p_q

                neighbors = list(source_graph.neighbors(v_i))
                embedded_neighbor_chains = [phi[u] for u in neighbors if phi[u]]
                
                try:
                    new_chain = self._find_minimal_vertex_model(target_graph, w, embedded_neighbor_chains, T_nodes)
                    phi[v_i] = new_chain
                except ValueError as e:
                    print(f"\n[DEBUG] Pathfinding failed for node {v_i} at stage {stage}: {e}")
                    raise RuntimeError(f"Disconnected target or pathing failure at node {v_i}. {e}")
            
            max_overlap = 0
            for q in T_nodes:
                count = sum(1 for v in S_nodes if q in phi[v])
                max_overlap = max(max_overlap, count)
                
            sum_chains = sum(len(chain) for chain in phi.values())
            
            if max_overlap <= 1:
                phi = self._shorten_chains(source_graph, target_graph, phi)
                clean_phi = {int(k): [int(q) for q in v] for k, v in phi.items()}
                return {
                    'embedding': clean_phi,
                    'time': time.time() - start_time
                }
            
            is_improving = (max_overlap < prev_max_overlap) or (sum_chains < prev_sum_chains)
            if is_improving:
                current_patience = 0
            else:
                current_patience += 1
                
            if current_patience >= patience_limit and stage > 2:
                print(f"\n[DEBUG] Max patience ({patience_limit}) reached at stage {stage}. Overlap: {max_overlap}, Chains: {sum_chains}")
                raise RuntimeError(f"Stuck in local minimum. Overlap remains {max_overlap}.")
                
            prev_max_overlap = max_overlap
            prev_sum_chains = sum_chains
            
            for q in T_nodes:
                h[q] *= h_decay
                if sum(1 for v in S_nodes if q in phi[v]) > 1:
                    h[q] += h_inc
                    
            stage += 1
            
        print(f"\n[DEBUG] Max iterations reached. Failed Embedding State: {phi}")
        raise RuntimeError(f"Max iterations ({max_iterations}) reached without resolving overlap.")
    def _shorten_chains(self, source_graph, target_graph, phi):
        """Greedily strips redundant qubits from the final embedding."""
        for v in source_graph.nodes():
            chain = list(phi[v])
            if len(chain) <= 1:
                continue
                
            neighbors = list(source_graph.neighbors(v))
            changed = True
            
            while changed:
                changed = False
                for q in chain:
                    candidate_chain = [x for x in chain if x != q]
                    
                    if not candidate_chain:
                        continue
                        
                    if not self._is_connected(target_graph, candidate_chain):
                        continue
                        
                    valid_connections = True
                    for u in neighbors:
                        connected_to_u = False
                        for c_node in candidate_chain:
                            for u_node in phi[u]:
                                if target_graph.has_edge(c_node, u_node):
                                    connected_to_u = True
                                    break
                            if connected_to_u:
                                break
                        if not connected_to_u:
                            valid_connections = False
                            break
                            
                    if valid_connections:
                        chain = candidate_chain
                        changed = True
                        break
            
            phi[v] = chain
        return phi

    def _is_connected(self, target_graph, chain):
        if not chain:
            return False
        visited = set()
        stack = [chain[0]]
        chain_set = set(chain)
        
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                for neighbor in target_graph.neighbors(node):
                    if neighbor in chain_set and neighbor not in visited:
                        stack.append(neighbor)
                        
        return len(visited) == len(chain)

    def _find_minimal_vertex_model(self, T, w, neighbor_chains, T_nodes):
        if not neighbor_chains:
            best_q = min(T_nodes, key=lambda q: w[q])
            return [best_q]
            
        distances = []
        predecessors = []
        
        for chain in neighbor_chains:
            dist, pred = self._node_weighted_dijkstra(T, chain, w)
            distances.append(dist)
            predecessors.append(pred)
            
        best_g = None
        min_total_dist = float('inf')
        
        for q in T_nodes:
            total_dist = sum(d.get(q, float('inf')) for d in distances)
                
            if total_dist < min_total_dist:
                min_total_dist = total_dist
                best_g = q
                
        if best_g is None or min_total_dist == float('inf'):
            raise ValueError("No valid Steiner root found.")
            
        final_chain = set([best_g])
        for pred in predecessors:
            curr = best_g
            while curr in pred and pred[curr] is not None:
                curr = pred[curr]
                final_chain.add(curr)
                
        return list(final_chain)

    def _node_weighted_dijkstra(self, T, sources, w):
        dist = {}
        pred = {}
        pq = []
        
        perimeter = set()
        for s in sources:
            for v in T.neighbors(s):
                if v not in sources:
                    perimeter.add(v)
                    
        if not perimeter:
            perimeter = set(sources)
                
        for u in perimeter:
            dist[u] = w[u]
            pred[u] = None
            heapq.heappush(pq, (w[u], u))
            
        while pq:
            d, u = heapq.heappop(pq)
            
            if d > dist.get(u, float('inf')):
                continue
                
            for v in T.neighbors(u):
                alt = d + w[v]
                if alt < dist.get(v, float('inf')):
                    dist[v] = alt
                    pred[v] = u
                    heapq.heappush(pq, (alt, v))
                    
        return dist, pred

@register_algorithm("gf_bolt_adaptive")
class GFBoltAdaptive(EmbeddingAlgorithm):
    """
    GF-Bolt-Adaptive (Strategy A).
    Prioritizes rerouting based on overlap involvement, chain length, and neighbor stress.
    Uses a build-pass initialization to avoid topological chaos.
    """

    def embed(self, source_graph: Any, target_graph: Any, timeout: float = 60.0, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        seed = kwargs.get('seed', None)
        if seed is not None:
            random.seed(seed)
            
        S_nodes = list(source_graph.nodes())
        T_nodes = list(target_graph.nodes())
        
        if not S_nodes or not T_nodes:
            return None
            
        # 1. Precompute Difficulty
        T_avg_deg = max(1.0, sum(dict(target_graph.degree()).values()) / len(T_nodes))
        D = {v: source_graph.degree(v) / T_avg_deg for v in S_nodes}
        D_max = max(D.values()) if D else 1.0
        
        # 2. Initialization: Greedy Build Pass
        phi = {v: [] for v in S_nodes}
        h = {q: 0.0 for q in T_nodes}
        
        init_order = sorted(S_nodes, key=lambda v: D[v], reverse=True)
        for v_i in init_order:
            costs = {q: 1.0 for q in T_nodes}
            neighbor_chains = [phi[u] for u in source_graph.neighbors(v_i) if phi[u]]
            phi[v_i] = self._find_minimal_vertex_model(target_graph, costs, neighbor_chains, T_nodes)

        # 3. Adaptive Hyperparameters
        w_coeffs = kwargs.get('weights', [2.0, 1.5, 2.0, 0.5]) 
        sigma_start, sigma_end = 0.5, 2.5
        max_iterations = kwargs.get('max_iterations', 100)
        alpha_scale = kwargs.get('alpha_scale', 0.5)
        
        # 4. Main Adaptive Loop
        for stage in range(1, max_iterations + 1):
            if time.time() - start_time > timeout:
                return None

            q_occupancy = {}
            for v in S_nodes:
                for q in phi[v]:
                    q_occupancy[q] = q_occupancy.get(q, 0) + 1
            
            overlaps_per_node = {v: sum(1 for q in phi[v] if q_occupancy.get(q, 0) > 1) for v in S_nodes}

            max_overlap = max(q_occupancy.values()) if q_occupancy else 0
            if max_overlap <= 1:
                phi = self._shorten_chains(source_graph, target_graph, phi)
                clean_phi = {int(k): [int(q) for q in v] for k, v in phi.items()}
                return {'embedding': clean_phi, 'time': time.time() - start_time}

            avg_len = max(1.0, np.mean([len(phi[v]) for v in S_nodes]))
            scores = []
            for v in S_nodes:
                neighbors = list(source_graph.neighbors(v))
                stress = sum(overlaps_per_node[u] for u in neighbors) / max(1, len(neighbors))
                h_max = max([h[q] for q in phi[v]]) if phi[v] else 0
                
                s = (w_coeffs[0] * overlaps_per_node[v] + 
                     w_coeffs[1] * max(0, (len(phi[v]) / avg_len) - 1) + 
                     w_coeffs[2] * stress + 
                     w_coeffs[3] * h_max)
                scores.append((v, s))

            s_values = [x[1] for x in scores]
            curr_sigma = sigma_start + (sigma_end - sigma_start) * (stage / max_iterations)
            threshold = np.mean(s_values) + curr_sigma * np.std(s_values)
            
            to_reroute = [v for v, s in scores if s >= threshold or overlaps_per_node[v] > 0]
            random.shuffle(to_reroute)
            to_reroute.sort(key=lambda v: D[v], reverse=True)

            for v_i in to_reroute:
                alpha = alpha_scale * (D[v_i] / D_max)
                costs = {}
                for q in T_nodes:
                    occ = sum(1 for v_j in S_nodes if v_j != v_i and q in phi[v_j])
                    costs[q] = alpha + (1 - alpha) * (1.0 + h[q]) * (1.0 + occ)

                neighbor_chains = [phi[u] for u in source_graph.neighbors(v_i) if phi[u]]
                phi[v_i] = self._find_minimal_vertex_model(target_graph, costs, neighbor_chains, T_nodes)

            for q in T_nodes:
                h[q] = (h[q] * 0.95) + (1.0 if q_occupancy.get(q, 0) > 1 else 0)

        return None

    def _shorten_chains(self, source_graph, target_graph, phi):
        """Standard greedy shortening pass."""
        for v in source_graph.nodes():
            chain = list(phi[v])
            if len(chain) <= 1: continue
            neighbors = list(source_graph.neighbors(v))
            changed = True
            while changed:
                changed = False
                for q in chain:
                    candidate = [x for x in chain if x != q]
                    if not candidate: continue
                    # Connectivity check using networkx
                    if not nx.is_connected(target_graph.subgraph(candidate)): continue
                    
                    valid_edges = True
                    for u in neighbors:
                        if not any(target_graph.has_edge(c, un) for c in candidate for un in phi[u]):
                            valid_edges = False; break
                    if valid_edges:
                        chain = candidate; changed = True; break
            phi[v] = chain
        return phi

    def _find_minimal_vertex_model(self, T, w, neighbor_chains, T_nodes):
        if not neighbor_chains:
            return [min(T_nodes, key=lambda q: w[q])]
        
        distances, predecessors = [], []
        for chain in neighbor_chains:
            d, p = self._node_weighted_dijkstra(T, chain, w)
            distances.append(d)
            predecessors.append(p)
        
        best_g, min_dist = None, float('inf')
        for q in T_nodes:
            d_total = sum(d.get(q, float('inf')) for d in distances)
            if d_total < min_dist:
                min_dist = d_total
                best_g = q
        
        if best_g is None: 
            return [random.choice(T_nodes)]
            
        final = {best_g}
        for p in predecessors:
            curr = best_g
            while curr in p and p[curr] is not None:
                curr = p[curr]
                final.add(curr)
        return list(final)

    def _node_weighted_dijkstra(self, T, sources, w):
        dist, pred, pq = {}, {}, []
        perimeter = {v for s in sources for v in T.neighbors(s) if v not in sources}
        if not perimeter: perimeter = set(sources)
        
        for u in perimeter:
            dist[u] = w[u]
            pred[u] = None
            heapq.heappush(pq, (w[u], u))
            
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, float('inf')): continue
            for v in T.neighbors(u):
                alt = d + w[v]
                if alt < dist.get(v, float('inf')):
                    dist[v] = alt
                    pred[v] = u
                    heapq.heappush(pq, (alt, v))
        return dist, pred

@register_algorithm("gf_adaptive_priority")
class GFAdaptivePriority(EmbeddingAlgorithm):
    """
    GF-adaptive-priority.
    Dynamic priority scheduler that selects AND sorts rerouting targets
    based on their current contribution to board congestion and stress.
    """

    def embed(self, source_graph: Any, target_graph: Any, timeout: float = 60.0, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        seed = kwargs.get('seed', None)
        if seed is not None:
            random.seed(seed)
            
        S_nodes = list(source_graph.nodes())
        T_nodes = list(target_graph.nodes())
        
        if not S_nodes or not T_nodes:
            return None
            
        # 1. Precompute Static Difficulty
        T_avg_deg = max(1.0, sum(dict(target_graph.degree()).values()) / len(T_nodes))
        D = {v: source_graph.degree(v) / T_avg_deg for v in S_nodes}
        D_max = max(D.values()) if D else 1.0
        
        # 2. Build Pass: Establish initial structure
        phi = {v: [] for v in S_nodes}
        h = {q: 0.0 for q in T_nodes}
        
        init_order = sorted(S_nodes, key=lambda v: D[v], reverse=True)
        for v_i in init_order:
            costs = {q: 1.0 for q in T_nodes}
            neighbors = [phi[u] for u in source_graph.neighbors(v_i) if phi[u]]
            phi[v_i] = self._find_minimal_vertex_model(target_graph, costs, neighbor_chains=neighbors, T_nodes=T_nodes)

        # 3. Hyperparameters
        # weights: [overlap, length_ratio, neighbor_stress, h_max]
        w_coeffs = kwargs.get('weights', [2.0, 1.0, 2.5, 0.5])
        sigma_start, sigma_end = 0.5, 2.5
        max_iterations = kwargs.get('max_iterations', 100)
        alpha_scale = kwargs.get('alpha_scale', 0.5)
        
        # 4. Iterative Dynamic Priority Phase
        for stage in range(1, max_iterations + 1):
            if time.time() - start_time > timeout:
                return None

            # Calculate Current Metrics
            q_occupancy = {}
            for v in S_nodes:
                for q in phi[v]:
                    q_occupancy[q] = q_occupancy.get(q, 0) + 1
            
            overlaps_per_node = {v: sum(1 for q in phi[v] if q_occupancy.get(q, 0) > 1) for v in S_nodes}

            # Check Termination
            max_overlap = max(q_occupancy.values()) if q_occupancy else 0
            if max_overlap <= 1:
                phi = self._shorten_chains(source_graph, target_graph, phi)
                return {
                    'embedding': {int(k): [int(q) for q in v] for k, v in phi.items()},
                    'time': time.time() - start_time
                }

            # Dynamic Scoring
            avg_len = max(1.0, np.mean([len(phi[v]) for v in S_nodes]))
            scores = []
            for v in S_nodes:
                neighbors = list(source_graph.neighbors(v))
                stress = sum(overlaps_per_node[u] for u in neighbors) / max(1, len(neighbors))
                h_max = max([h[q] for q in phi[v]]) if phi[v] else 0
                
                s = (w_coeffs[0] * overlaps_per_node[v] + 
                     w_coeffs[1] * max(0, (len(phi[v]) / avg_len) - 1) + 
                     w_coeffs[2] * stress + 
                     w_coeffs[3] * h_max)
                scores.append((v, s))

            # Sigma Thresholding
            s_values = [x[1] for x in scores]
            curr_sigma = sigma_start + (sigma_end - sigma_start) * (stage / max_iterations)
            threshold = np.mean(s_values) + curr_sigma * np.std(s_values)
            
            # --- DYNAMIC SCHEDULING ---
            # 1. Select problematic nodes
            targets = [item for item in scores if item[1] >= threshold or overlaps_per_node[item[0]] > 0]
            
            # 2. Sort problematic nodes by their badness score (Dynamic Priority)
            # This ensures those causing the most "stress" are placed first into the vacuum.
            targets.sort(key=lambda x: x[1], reverse=True)
            to_reroute = [item[0] for item in targets]
            # ---------------------------

            # Rip-up and Sequentially Reroute
            for v_i in to_reroute:
                alpha = alpha_scale * (D[v_i] / D_max)
                costs = {}
                for q in T_nodes:
                    occ = sum(1 for v_j in S_nodes if v_j != v_i and q in phi[v_j])
                    # Dynamic cost based on historical penalty and current occupancy
                    costs[q] = alpha + (1 - alpha) * (1.0 + h[q]) * (1.0 + occ)

                neighbor_chains = [phi[u] for u in source_graph.neighbors(v_i) if phi[u]]
                phi[v_i] = self._find_minimal_vertex_model(target_graph, costs, neighbor_chains, T_nodes)

            # Historical Update
            for q in T_nodes:
                h[q] = (h[q] * 0.95) + (1.0 if q_occupancy.get(q, 0) > 1 else 0)

        return None

    def _shorten_chains(self, source_graph, target_graph, phi):
        """Greedy reduction pass."""
        for v in source_graph.nodes():
            chain = list(phi[v])
            if len(chain) <= 1: continue
            neighbors = list(source_graph.neighbors(v))
            changed = True
            while changed:
                changed = False
                for q in chain:
                    candidate = [x for x in chain if x != q]
                    if not candidate: continue
                    if not nx.is_connected(target_graph.subgraph(candidate)): continue
                    
                    valid = True
                    for u in neighbors:
                        if not any(target_graph.has_edge(c, un) for c in candidate for un in phi[u]):
                            valid = False; break
                    if valid:
                        chain = candidate; changed = True; break
            phi[v] = chain
        return phi

    def _find_minimal_vertex_model(self, T, w, neighbor_chains, T_nodes):
        if not neighbor_chains:
            return [min(T_nodes, key=lambda q: w[q])]
        
        dist_maps, pred_maps = [], []
        for chain in neighbor_chains:
            d, p = self._node_weighted_dijkstra(T, chain, w)
            dist_maps.append(d)
            pred_maps.append(p)
        
        best_q, min_d = None, float('inf')
        for q in T_nodes:
            d_sum = sum(d.get(q, float('inf')) for d in dist_maps)
            if d_sum < min_d:
                min_d = d_sum
                best_q = q
        
        if best_q is None: return [random.choice(T_nodes)]
            
        final_chain = {best_q}
        for p in pred_maps:
            curr = best_q
            while curr in p and p[curr] is not None:
                curr = p[curr]
                final_chain.add(curr)
        return list(final_chain)

    def _node_weighted_dijkstra(self, T, sources, w):
        dist, pred, pq = {}, {}, []
        perimeter = {v for s in sources for v in T.neighbors(s) if v not in sources}
        if not perimeter: perimeter = set(sources)
        
        for u in perimeter:
            dist[u] = w[u]
            pred[u] = None
            heapq.heappush(pq, (w[u], u))
            
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, float('inf')): continue
            for v in T.neighbors(u):
                alt = d + w[v]
                if alt < dist.get(v, float('inf')):
                    dist[v] = alt
                    pred[v] = u
                    heapq.heappush(pq, (alt, v))
        return dist, pred

@register_algorithm("gf_adaptive_new")
class GFBoltAdaptive(EmbeddingAlgorithm):
    """
    GF-Bolt-Adaptive with p_fac slow-start schedule.
    Present congestion cost ramps from near-zero to p_fac_max over p_fac_ramp_stages,
    allowing early iterations to tolerate overlap before pressure builds.
    """

    def embed(self, source_graph: Any, target_graph: Any, timeout: float = 60.0, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        seed = kwargs.get('seed', None)
        if seed is not None:
            random.seed(seed)
            
        S_nodes = list(source_graph.nodes())
        T_nodes = list(target_graph.nodes())
        
        if not S_nodes or not T_nodes:
            return None
            
        # 1. Precompute Difficulty
        T_avg_deg = max(1.0, sum(dict(target_graph.degree()).values()) / len(T_nodes))
        D = {v: source_graph.degree(v) / T_avg_deg for v in S_nodes}
        D_max = max(D.values()) if D else 1.0
        
        # 2. Initialization: Greedy Build Pass (flat costs, no congestion yet)
        phi = {v: [] for v in S_nodes}
        h = {q: 0.0 for q in T_nodes}
        
        init_order = sorted(S_nodes, key=lambda v: D[v], reverse=True)
        for v_i in init_order:
            costs = {q: 1.0 for q in T_nodes}
            neighbor_chains = [phi[u] for u in source_graph.neighbors(v_i) if phi[u]]
            phi[v_i] = self._find_minimal_vertex_model(target_graph, costs, neighbor_chains, T_nodes)

        # 3. Hyperparameters
        w_coeffs         = kwargs.get('weights', [2.0, 1.5, 2.0, 0.5])
        sigma_start      = 0.5
        sigma_end        = 2.5
        max_iterations   = kwargs.get('max_iterations', 100)
        alpha_scale      = kwargs.get('alpha_scale', 0.5)
        h_inc            = kwargs.get('h_inc', 1.0)
        h_decay          = kwargs.get('h_decay', 0.95)
        p_fac_max        = kwargs.get('p_fac', 1.0)
        p_fac_start      = kwargs.get('p_fac_start', 0.05)
        p_fac_ramp_stages = kwargs.get('p_fac_ramp_stages', 10)
        
        # 4. Main Adaptive Loop
        for stage in range(1, max_iterations + 1):
            if time.time() - start_time > timeout:
                return None

            # p_fac schedule: linear ramp from p_fac_start to p_fac_max
            ramp_t = min(stage - 1, p_fac_ramp_stages) / p_fac_ramp_stages
            p_fac  = p_fac_start + (p_fac_max - p_fac_start) * ramp_t

            # Snapshot occupancy once per iteration — all nodes route against same state
            q_occupancy = {}
            for v in S_nodes:
                for q in phi[v]:
                    q_occupancy[q] = q_occupancy.get(q, 0) + 1
            
            overlaps_per_node = {
                v: sum(1 for q in phi[v] if q_occupancy.get(q, 0) > 1)
                for v in S_nodes
            }

            max_overlap = max(q_occupancy.values()) if q_occupancy else 0
            if max_overlap <= 1:
                phi = self._shorten_chains(source_graph, target_graph, phi)
                clean_phi = {int(k): [int(q) for q in v] for k, v in phi.items()}
                return {'embedding': clean_phi, 'time': time.time() - start_time}

            # Adaptive scoring: which nodes to reroute this iteration
            avg_len = max(1.0, np.mean([len(phi[v]) for v in S_nodes]))
            scores = []
            for v in S_nodes:
                neighbors = list(source_graph.neighbors(v))
                stress = sum(overlaps_per_node[u] for u in neighbors) / max(1, len(neighbors))
                h_max  = max((h[q] for q in phi[v]), default=0)
                
                s = (w_coeffs[0] * overlaps_per_node[v] +
                     w_coeffs[1] * max(0, (len(phi[v]) / avg_len) - 1) +
                     w_coeffs[2] * stress +
                     w_coeffs[3] * h_max)
                scores.append((v, s))

            s_values   = [x[1] for x in scores]
            curr_sigma = sigma_start + (sigma_end - sigma_start) * (stage / max_iterations)
            threshold  = np.mean(s_values) + curr_sigma * np.std(s_values)
            
            to_reroute = [v for v, s in scores if s >= threshold or overlaps_per_node[v] > 0]
            random.shuffle(to_reroute)
            to_reroute.sort(key=lambda v: D[v], reverse=True)

            for v_i in to_reroute:
                alpha = alpha_scale * (D[v_i] / D_max)
                costs = {}
                for q in T_nodes:
                    # Use snapshot occupancy, not live — keeps negotiation fair
                    occ   = sum(1 for v_j in S_nodes if v_j != v_i and q in phi[v_j])
                    p_q   = 1.0 + occ * p_fac
                    costs[q] = alpha * 1.0 + (1.0 - alpha) * (1.0 + h[q]) * p_q

                neighbor_chains = [phi[u] for u in source_graph.neighbors(v_i) if phi[u]]
                phi[v_i] = self._find_minimal_vertex_model(target_graph, costs, neighbor_chains, T_nodes)

            # Update historical congestion against end-of-iteration state
            for q in T_nodes:
                h[q] *= h_decay
                if q_occupancy.get(q, 0) > 1:
                    h[q] += h_inc

            stage += 1

        return None
     
    def _shorten_chains(self, source_graph, target_graph, phi):
        """Greedy reduction pass."""
        for v in source_graph.nodes():
            chain = list(phi[v])
            if len(chain) <= 1: continue
            neighbors = list(source_graph.neighbors(v))
            changed = True
            while changed:
                changed = False
                for q in chain:
                    candidate = [x for x in chain if x != q]
                    if not candidate: continue
                    if not nx.is_connected(target_graph.subgraph(candidate)): continue
                    
                    valid = True
                    for u in neighbors:
                        if not any(target_graph.has_edge(c, un) for c in candidate for un in phi[u]):
                            valid = False; break
                    if valid:
                        chain = candidate; changed = True; break
            phi[v] = chain
        return phi

    def _find_minimal_vertex_model(self, T, w, neighbor_chains, T_nodes):
        if not neighbor_chains:
            return [min(T_nodes, key=lambda q: w[q])]
        
        dist_maps, pred_maps = [], []
        for chain in neighbor_chains:
            d, p = self._node_weighted_dijkstra(T, chain, w)
            dist_maps.append(d)
            pred_maps.append(p)
        
        best_q, min_d = None, float('inf')
        for q in T_nodes:
            d_sum = sum(d.get(q, float('inf')) for d in dist_maps)
            if d_sum < min_d:
                min_d = d_sum
                best_q = q
        
        if best_q is None: return [random.choice(T_nodes)]
            
        final_chain = {best_q}
        for p in pred_maps:
            curr = best_q
            while curr in p and p[curr] is not None:
                curr = p[curr]
                final_chain.add(curr)
        return list(final_chain)

    def _node_weighted_dijkstra(self, T, sources, w):
        dist, pred, pq = {}, {}, []
        perimeter = {v for s in sources for v in T.neighbors(s) if v not in sources}
        if not perimeter: perimeter = set(sources)
        
        for u in perimeter:
            dist[u] = w[u]
            pred[u] = None
            heapq.heappush(pq, (w[u], u))
            
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, float('inf')): continue
            for v in T.neighbors(u):
                alt = d + w[v]
                if alt < dist.get(v, float('inf')):
                    dist[v] = alt
                    pred[v] = u
                    heapq.heappush(pq, (alt, v))
        return dist, pred

@register_algorithm("gf_spring_adaptive")
class GFSpringAdaptive(EmbeddingAlgorithm):
    """
    GF-spring-adaptive.
    Force-directed (Spring) initialization using gravity wells to seed placement,
    combined with adaptive priority rerouting for congestion management.
    """

    def embed(self, source_graph: Any, target_graph: Any, timeout: float = 60.0, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        seed = kwargs.get('seed', None)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        S_nodes = list(source_graph.nodes())
        T_nodes = list(target_graph.nodes())
        
        # 1. Force-Directed Layout (Spring Method)
        # k controls the optimal distance between nodes (1/sqrt(n) is standard)
        pos_S = nx.spring_layout(source_graph, seed=seed, k=kwargs.get('k', None))
        
        # Mapping projection to target grid
        num_T = len(T_nodes)
        side = int(np.sqrt(num_T / 8)) if num_T >= 8 else 1
        
        # 2. Gravity Well Seeds
        gravity_seeds = {}
        for v, coords in pos_S.items():
            x = int((coords[0] + 1) / 2 * (side - 1))
            y = int((coords[1] + 1) / 2 * (side - 1))
            q_idx = (y * side + x) * 8
            gravity_seeds[v] = T_nodes[min(int(q_idx), num_T - 1)]

        # 3. Initialization Build Pass with Soft Seeding
        phi = {v: [] for v in S_nodes}
        h = {q: 0.0 for q in T_nodes}
        
        T_avg_deg = max(1.0, sum(dict(target_graph.degree()).values()) / len(T_nodes))
        D = {v: source_graph.degree(v) / T_avg_deg for v in S_nodes}
        D_max = max(D.values()) if D else 1.0
        
        init_order = sorted(S_nodes, key=lambda v: D[v], reverse=True)
        for v_i in init_order:
            # Create gravity well cost: Seed qubit is cheap, others are normal
            seed_q = gravity_seeds[v_i]
            base_costs = {q: 1.0 for q in T_nodes}
            base_costs[seed_q] = 0.1 # The Gravity Well
            
            # Neighbor attraction
            neighbor_chains = [phi[u] for u in source_graph.neighbors(v_i) if phi[u]]
            phi[v_i] = self._find_minimal_vertex_model(target_graph, base_costs, neighbor_chains, T_nodes)

        # 4. Adaptive Priority Phase
        w_coeffs = kwargs.get('weights', [2.0, 1.5, 2.5, 0.5])
        sigma_start, sigma_end = 0.5, 2.5
        max_iterations = kwargs.get('max_iterations', 100)
        
        for stage in range(1, max_iterations + 1):
            if time.time() - start_time > timeout: return None

            q_occupancy = {}
            for v in S_nodes:
                for q in phi[v]:
                    q_occupancy[q] = q_occupancy.get(q, 0) + 1
            
            overlaps_per_node = {v: sum(1 for q in phi[v] if q_occupancy.get(q, 0) > 1) for v in S_nodes}

            if max(q_occupancy.values(), default=0) <= 1:
                phi = self._shorten_chains(source_graph, target_graph, phi)
                return {
                    'embedding': {int(k): [int(q) for q in v] for k, v in phi.items()},
                    'time': time.time() - start_time
                }

            # Scoring and Scheduling
            avg_len = max(1.0, np.mean([len(phi[v]) for v in S_nodes]))
            scores = []
            for v in S_nodes:
                stress = sum(overlaps_per_node[u] for u in source_graph.neighbors(v)) / max(1, source_graph.degree(v))
                h_max = max([h[q] for q in phi[v]]) if phi[v] else 0
                s = (w_coeffs[0] * overlaps_per_node[v] + 
                     w_coeffs[1] * max(0, (len(phi[v]) / avg_len) - 1) + 
                     w_coeffs[2] * stress + 
                     w_coeffs[3] * h_max)
                scores.append((v, s))

            curr_sigma = sigma_start + (sigma_end - sigma_start) * (stage / max_iterations)
            threshold = np.mean([x[1] for x in scores]) + curr_sigma * np.std([x[1] for x in scores])
            
            targets = [item for item in scores if item[1] >= threshold or overlaps_per_node[item[0]] > 0]
            targets.sort(key=lambda x: x[1], reverse=True)

            # Reroute targets
            for v_i, _ in targets:
                alpha = 0.5 * (D[v_i] / D_max)
                # Soft-seeding persists slightly in early iterations to maintain layout
                seed_bias = 0.1 if stage < 5 else 1.0
                seed_q = gravity_seeds[v_i]
                
                costs = {}
                for q in T_nodes:
                    occ = sum(1 for v_j in S_nodes if v_j != v_i and q in phi[v_j])
                    bias = seed_bias if q == seed_q else 1.0
                    costs[q] = (alpha + (1 - alpha) * (1.0 + h[q]) * (1.0 + occ)) * bias

                neighbor_chains = [phi[u] for u in source_graph.neighbors(v_i) if phi[u]]
                phi[v_i] = self._find_minimal_vertex_model(target_graph, costs, neighbor_chains, T_nodes)

            # History Update
            for q in T_nodes:
                h[q] = (h[q] * 0.95) + (1.0 if q_occupancy.get(q, 0) > 1 else 0)

        return None

    def _shorten_chains(self, source_graph, target_graph, phi):
        """Standard connectivity-aware reducer."""
        for v in source_graph.nodes():
            chain = list(phi[v]); neighbors = list(source_graph.neighbors(v))
            changed = True
            while changed:
                changed = False
                for q in chain:
                    candidate = [x for x in chain if x != q]
                    if not candidate or not nx.is_connected(target_graph.subgraph(candidate)): continue
                    if all(any(target_graph.has_edge(c, un) for c in candidate for un in phi[u]) for u in neighbors):
                        chain = candidate; changed = True; break
            phi[v] = chain
        return phi

    def _find_minimal_vertex_model(self, T, w, neighbor_chains, T_nodes):
        if not neighbor_chains: return [min(T_nodes, key=lambda q: w[q])]
        dist_maps, pred_maps = [], []
        for chain in neighbor_chains:
            d, p = self._node_weighted_dijkstra(T, chain, w)
            dist_maps.append(d); pred_maps.append(p)
        best_q = min(T_nodes, key=lambda q: sum(d.get(q, float('inf')) for d in dist_maps))
        final = {best_q}
        for p in pred_maps:
            curr = best_q
            while curr in p and p[curr] is not None:
                curr = p[curr]; final.add(curr)
        return list(final)

    def _node_weighted_dijkstra(self, T, sources, w):
        dist, pred, pq = {}, {}, []
        perimeter = {v for s in sources for v in T.neighbors(s) if v not in sources}
        if not perimeter: perimeter = set(sources)
        for u in perimeter:
            dist[u] = w[u]; pred[u] = None; heapq.heappush(pq, (w[u], u))
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, float('inf')): continue
            for v in T.neighbors(u):
                alt = d + w[v]
                if alt < dist.get(v, float('inf')):
                    dist[v] = alt; pred[v] = u; heapq.heappush(pq, (alt, v))
        return dist, pred


# 2. Execute the Benchmark
if __name__ == "__main__":
    bench = EmbeddingBenchmark(target_graph=None)
    
    # Run benchmark including the newly registered 'gf_bolt'
    direc = bench.run_full_benchmark(
        graph_selection='dense',
        topologies=[
                    'chimera_4x4x4',
                    'chimera_8x8x4'
                    ],
        methods=[
                'minorminer',
                'atom',
                 ],
        n_trials=5,
        warmup_trials=1,
        timeout=60,
        n_workers=5,
        verbose=True,
        batch_note='Testing shared-graph intersection', # Corrected mismatch
    )

    if direc:
        an = BenchmarkAnalysis(direc)
        an.generate_report()
    # else: cancelled — resume with load_benchmark()