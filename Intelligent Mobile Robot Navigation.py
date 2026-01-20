import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from matplotlib.widgets import Button, Slider, RadioButtons
import random
from collections import deque
import time
import math


def simple_ttest(sample1, sample2):
    """Simple independent t-test implementation"""
    n1, n2 = len(sample1), len(sample2)
    mean1, mean2 = np.mean(sample1), np.mean(sample2)
    var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = math.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    
    # T-statistic
    t_stat = (mean1 - mean2) / (pooled_std * math.sqrt(1/n1 + 1/n2))
    
    # Degrees of freedom
    df = n1 + n2 - 2
    
    # Approximate p-value using normal distribution (good enough for df > 30)
    # For smaller df, this is an approximation
    from math import erf
    p_value = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / math.sqrt(2))))
    
    return t_stat, p_value


class ControllerInputs:
    """Structured input representation for the controller"""
    def __init__(self, pos, goal, grid):
        self.position = pos
        self.goal = goal
        self.distance_to_goal = np.hypot(goal[0] - pos[0], goal[1] - pos[1])
        max_dist = np.hypot(grid.shape[1], grid.shape[0])
        self.distance_to_goal_normalized = self.distance_to_goal / max_dist
        self.angle_to_goal = np.arctan2(goal[1] - pos[1], goal[0] - pos[0])
        self.angle_to_goal_degrees = np.degrees(self.angle_to_goal)
        self.obstacle_distances = self._compute_obstacle_distances(pos, grid)
        self.min_obstacle_distance = np.min(self.obstacle_distances)
        self.avg_obstacle_distance = np.mean(self.obstacle_distances)
        self.front_clear = self.obstacle_distances[0] > 2
        self.left_clear = self.obstacle_distances[6] > 2
        self.right_clear = self.obstacle_distances[2] > 2
        self.back_clear = self.obstacle_distances[4] > 2

    def _compute_obstacle_distances(self, pos, grid):
        x, y = pos
        directions = [(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1)]
        distances = []
        max_range = 7
        for dx, dy in directions:
            distance = 0
            for step in range(1, max_range + 1):
                cx, cy = int(x + dx * step), int(y + dy * step)
                if not (0 <= cx < grid.shape[1] and 0 <= cy < grid.shape[0]):
                    distance = step
                    break
                if grid[cy, cx] == 1:
                    distance = step
                    break
                distance = step
            distances.append(distance)
        return np.array(distances)

    def to_vector(self):
        return np.array([
            self.distance_to_goal_normalized,
            self.angle_to_goal / np.pi,
            *self.obstacle_distances / 7.0,
        ])


class ControllerOutputs:
    """Structured output representation for the controller"""
    def __init__(self, direction, confidence=1.0, method="fuzzy"):
        self.direction = direction
        self.dx, self.dy = direction
        self.confidence = confidence
        self.method = method
        self.movement_angle = np.arctan2(self.dy, self.dx)
        self.movement_angle_degrees = np.degrees(self.movement_angle)


class PerformanceMetrics:
    """Comprehensive performance evaluation metrics"""
    def __init__(self):
        self.start_time = time.time()
        self.metrics = {
            'success': False,
            'steps_taken': 0,
            'path_length': 0.0,
            'execution_time': 0.0,
            'optimal_distance': 0.0,
            'actual_distance': 0.0,
            'path_efficiency': 0.0,
            'straightness_ratio': 0.0,
            'collisions': 0,
            'near_misses': 0,
            'min_obstacle_clearance': float('inf'),
            'avg_obstacle_clearance': 0.0,
            'total_turns': 0,
            'sharp_turns': 0,
            'smoothness_score': 0.0,
            'avg_turn_angle': 0.0,
            'total_distance_traveled': 0.0,
            'backtracking_count': 0,
            'revisited_cells': 0,
            'nn_confidence_avg': 0.0,
            'fuzzy_score_avg': 0.0,
            'hybrid_agreement': 0.0
        }
        self.path_history = []
        self.decision_history = []
        self.clearance_history = []

    def update_step(self, pos, inputs, outputs):
        self.metrics['steps_taken'] += 1
        self.path_history.append(tuple(pos))
        self.decision_history.append(outputs)
        self.clearance_history.append(inputs.min_obstacle_distance)
        if inputs.min_obstacle_distance < self.metrics['min_obstacle_clearance']:
            self.metrics['min_obstacle_clearance'] = inputs.min_obstacle_distance
        if inputs.min_obstacle_distance < 2.0:
            self.metrics['near_misses'] += 1

    def finalize(self, success, start, goal):
        self.metrics['success'] = success
        self.metrics['execution_time'] = time.time() - self.start_time
        if len(self.path_history) < 2:
            return
        path = np.array(self.path_history)
        distances = np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))
        self.metrics['actual_distance'] = np.sum(distances)
        self.metrics['path_length'] = len(self.path_history)
        self.metrics['optimal_distance'] = np.hypot(goal[0] - start[0], goal[1] - start[1])
        if self.metrics['actual_distance'] > 0:
            self.metrics['path_efficiency'] = (
                self.metrics['optimal_distance'] / self.metrics['actual_distance'] * 100
            )
        if len(path) > 0:
            end_to_end = np.hypot(path[-1][0] - path[0][0], path[-1][1] - path[0][1])
            self.metrics['straightness_ratio'] = end_to_end / self.metrics['actual_distance'] * 100
        self._calculate_smoothness(path)
        unique_cells = len(set(self.path_history))
        self.metrics['revisited_cells'] = len(self.path_history) - unique_cells
        if len(self.clearance_history) > 0:
            self.metrics['avg_obstacle_clearance'] = np.mean(self.clearance_history)
        nn_decisions = [d for d in self.decision_history if hasattr(d, 'confidence')]
        if nn_decisions:
            self.metrics['nn_confidence_avg'] = np.mean([d.confidence for d in nn_decisions])

    def _calculate_smoothness(self, path):
        if len(path) < 3:
            self.metrics['smoothness_score'] = 100.0
            return
        turn_angles = []
        for i in range(1, len(path) - 1):
            v1 = path[i] - path[i-1]
            v2 = path[i+1] - path[i]
            angle1 = np.arctan2(v1[1], v1[0])
            angle2 = np.arctan2(v2[1], v2[0])
            turn = angle2 - angle1
            turn = np.arctan2(np.sin(turn), np.cos(turn))
            turn_angles.append(abs(turn))
            self.metrics['total_turns'] += 1
            if abs(turn) > np.pi/2:
                self.metrics['sharp_turns'] += 1
        if turn_angles:
            self.metrics['avg_turn_angle'] = np.degrees(np.mean(turn_angles))
            self.metrics['smoothness_score'] = max(0, (1 - np.mean(turn_angles) / np.pi) * 100)

    def get_overall_score(self):
        score = 0
        if self.metrics['success']:
            score += 40
        if self.metrics['path_efficiency'] > 0:
            score += min(20, self.metrics['path_efficiency'] / 5)
        collision_penalty = min(20, self.metrics['collisions'] * 2)
        score += (20 - collision_penalty)
        score += self.metrics['smoothness_score'] / 10
        if self.metrics['steps_taken'] > 0:
            speed_score = max(0, 10 - self.metrics['steps_taken'] / 50)
            score += speed_score
        return min(100, max(0, score))

    def to_dict(self):
        return self.metrics.copy()


class NeuralNetwork:
    """Neural Network for path prediction"""
    def __init__(self):
        # DEEPER network with better initialization
        self.W1 = np.random.randn(10, 32) * np.sqrt(2.0 / 10)  # He initialization
        self.b1 = np.zeros((1, 32))
        self.W2 = np.random.randn(32, 16) * np.sqrt(2.0 / 32)
        self.b2 = np.zeros((1, 16))
        self.W3 = np.random.randn(16, 8) * np.sqrt(2.0 / 16)
        self.b3 = np.zeros((1, 8))
        # Add momentum for stable training
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)
        self.v_W3 = np.zeros_like(self.W3)
        self.v_b3 = np.zeros_like(self.b3)

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        # NEW LAYER
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.softmax(self.z3.flatten())
        return self.a3

    def train(self, X, y, lr=0.015, momentum=0.9):
        output = self.forward(X)

        # Backprop through 3 layers
        dz3 = (output - y).reshape(1, -1)
        dW3 = np.dot(self.a2.T, dz3)
        db3 = dz3

        da2 = np.dot(dz3, self.W3.T)
        dz2 = da2 * (self.z2 > 0)
        dW2 = np.dot(self.a1.T, dz2)
        db2 = dz2
        
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (self.z1 > 0)
        dW1 = np.dot(X.T, dz1)
        db1 = dz1
        
        # UPDATE WITH MOMENTUM
        self.v_W3 = momentum * self.v_W3 - lr * dW3
        self.v_b3 = momentum * self.v_b3 - lr * db3
        self.v_W2 = momentum * self.v_W2 - lr * dW2
        self.v_b2 = momentum * self.v_b2 - lr * db2
        self.v_W1 = momentum * self.v_W1 - lr * dW1
        self.v_b1 = momentum * self.v_b1 - lr * db1
        
        self.W3 += self.v_W3
        self.b3 += self.v_b3
        self.W2 += self.v_W2
        self.b2 += self.v_b2
        self.W1 += self.v_W1
        self.b1 += self.v_b1
        
        # ADD THIS LINE - Return loss for monitoring
        loss = -np.sum(y * np.log(output + 1e-8))
        return loss


class FuzzyController:
    """Advanced Fuzzy Logic Controller"""
    def __init__(self, obstacle_w=0.8, goal_w=0.9, smooth_w=0.6):
        self.obstacle_w = obstacle_w
        self.goal_w = goal_w
        self.smooth_w = smooth_w

    def is_safe_diagonal_move(self, x, y, dx, dy, grid):
        if dx != 0 and dy != 0:
            if (0 <= x+dx < grid.shape[1] and 0 <= y+dy < grid.shape[0]):
                if grid[y, x+dx] == 1 and grid[y+dy, x] == 1:
                    return False
        return True

    def fuzzy_rules(self, controller_inputs, prev_dir, visited_positions=None, grid=None):
        pos = controller_inputs.position
        goal = controller_inputs.goal
        angle_to_goal = controller_inputs.angle_to_goal
        dist_to_goal = controller_inputs.distance_to_goal
        obstacle_dists = controller_inputs.obstacle_distances
        x, y = pos
        directions = [(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1)]
        scores = []

        for i, (dx, dy) in enumerate(directions):
            nx, ny = x + dx, y + dy
            if grid is not None:
                if not (0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]):
                    scores.append(-10000)
                    continue
                if grid[ny, nx] == 1:
                    scores.append(-10000)
                    continue
                if not self.is_safe_diagonal_move(x, y, dx, dy, grid):
                    scores.append(-10000)
                    continue
            
            obstacle_dist_this_dir = obstacle_dists[i]
            if obstacle_dist_this_dir < 0.8:
                scores.append(-10000)
                continue

            dir_angle = np.arctan2(dy, dx)
            angle_diff = abs(angle_to_goal - dir_angle)
            angle_diff = min(angle_diff, 2*np.pi - angle_diff)
            goal_score = (1.0 - angle_diff / np.pi) * self.goal_w * 5.0

            if obstacle_dist_this_dir < 1.5:
                obs_score = -10.0 * (1.5 - obstacle_dist_this_dir)
            elif obstacle_dist_this_dir < 3.0:
                obs_score = -2.0 * (3.0 - obstacle_dist_this_dir) / 1.5
            else:
                obs_score = 2.0 * self.obstacle_w

            new_dist = np.hypot(goal[0] - nx, goal[1] - ny)
            progress = (dist_to_goal - new_dist) * 4.0

            smooth = 0
            if prev_dir:
                if (dx, dy) == prev_dir:
                    smooth = 2.0 * self.smooth_w
                else:
                    prev_angle = np.arctan2(prev_dir[1], prev_dir[0])
                    turn = abs(prev_angle - dir_angle)
                    turn = min(turn, 2*np.pi - turn)
                    smooth = -1.5 * turn

            visit_penalty = 0
            if visited_positions and (nx, ny) in visited_positions:
                visit_count = visited_positions.count((nx, ny))
                visit_penalty = -5.0 * visit_count

            exploration = 0
            if visited_positions:
                nearby_visits = sum(1 for vx, vy in visited_positions[-20:]
                                  if abs(vx-nx) <= 2 and abs(vy-ny) <= 2)
                exploration = -0.5 * nearby_visits

            left_bonus = 0
            if dx == -1:
                if pos[0] > goal[0] + 3:
                    left_bonus = 3.0
                if obstacle_dists[6] > 4:
                    left_bonus += 1.5

            right_bonus = 0
            if dx == 1:
                if pos[0] < goal[0] - 3:
                    right_bonus = 3.0
                if obstacle_dists[2] > 4:
                    right_bonus += 1.5

            up_bonus = 0
            if dy == -1:
                if pos[1] > goal[1] + 3:
                    up_bonus = 3.0

            down_bonus = 0
            if dy == 1:
                if pos[1] < goal[1] - 3:
                    down_bonus = 3.0

            total = (goal_score + obs_score + progress + smooth + 
                    visit_penalty + exploration + left_bonus + right_bonus + 
                    up_bonus + down_bonus)
            scores.append(total)

        return scores, directions


class HybridIntelligentSystem:
    """Hybrid system combining Fuzzy, GA, and NN"""
    def __init__(self, grid_size=20):
        self.fuzzy = FuzzyController()
        self.nn = NeuralNetwork()
        self.grid_size = grid_size
        self.training_buffer = deque(maxlen=1000)
        self.exploration_rate = 0.05  # Small exploration for variation

    def decide(self, controller_inputs, prev_dir, visited_positions=None, use_nn_weight=0.5, grid=None):
        fuzzy_scores, dirs = self.fuzzy.fuzzy_rules(
            controller_inputs, prev_dir, visited_positions, grid
        )
        
        fuzzy_scores_arr = np.array(fuzzy_scores)
        valid_mask = fuzzy_scores_arr > -9000
        
        # Add exploration for trial variation
        if random.random() < self.exploration_rate and valid_mask.any():
            valid_idxs = np.where(valid_mask)[0]
            if len(valid_idxs) > 0:
                best_idx = random.choice(valid_idxs)
                confidence = 0.5
                method = "exploration"
                direction = dirs[best_idx]
                return ControllerOutputs(direction, confidence, method)
        
        if use_nn_weight > 0 and len(self.training_buffer) > 150:
            nn_input = controller_inputs.to_vector().reshape(1, -1)
            nn_probs = self.nn.forward(nn_input)
            
            if valid_mask.any():
                valid_scores = fuzzy_scores_arr[valid_mask]
                fmin, fmax = valid_scores.min(), valid_scores.max()
                
                fuzzy_normalized = np.zeros_like(fuzzy_scores_arr)
                if fmax > fmin:
                    # CRITICAL: Proper normalization to [0,1]
                    fuzzy_normalized[valid_mask] = (valid_scores - fmin) / (fmax - fmin)
                    # Apply power scaling to amplify differences
                    fuzzy_normalized[valid_mask] = np.power(fuzzy_normalized[valid_mask], 0.5)
                else:
                    fuzzy_normalized[valid_mask] = 1.0
                
                # ADAPTIVE WEIGHTING based on NN confidence
                nn_confidence = np.max(nn_probs)
                if nn_confidence > 0.6:
                    adaptive_weight = min(0.7, use_nn_weight * 1.5)
                else:
                    adaptive_weight = use_nn_weight
                
                # Combine scores
                combined = (1 - adaptive_weight) * fuzzy_normalized + adaptive_weight * nn_probs
                combined[~valid_mask] = -10000
                
                best_idx = np.argmax(combined)
                confidence = float(combined[best_idx])
                method = "hybrid"
            else:
                best_idx = np.argmax(fuzzy_scores_arr)
                confidence = 0.0
                method = "fuzzy_fallback"
        else:
            best_idx = np.argmax(fuzzy_scores_arr)
            confidence = 1.0 if fuzzy_scores_arr[best_idx] > -9000 else 0.0
            method = "fuzzy"
        
        direction = dirs[best_idx]
        return ControllerOutputs(direction, confidence, method)

    def train_nn_batch(self, epochs=50):
        """IMPROVED training with better monitoring"""
        if len(self.training_buffer) < 50:
            print("Not enough training data (need at least 50 samples)")
            return
    
        print(f"\n{'='*70}")
        print(f"🧠 TRAINING NEURAL NETWORK")
        print(f"{'='*70}")
        print(f"Training samples: {len(self.training_buffer)}")
        print(f"Training epochs: {epochs}")
        print("-" * 70)
        
        data = list(self.training_buffer)
        losses = []
        
        for epoch in range(epochs):
            random.shuffle(data)
            epoch_loss = 0
            batch_size = min(100, len(data))
            
            for nn_in, label_idx in data[:batch_size]:
                target = np.zeros(8)
                target[label_idx] = 1.0
                loss = self.nn.train(nn_in, target, lr=0.015)
                epoch_loss += loss
            
            avg_loss = epoch_loss / batch_size
            losses.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch:3d}/{epochs}: Loss = {avg_loss:.4f}")
        
        # Test accuracy
        correct = 0
        test_samples = min(50, len(data))
        for nn_in, label_idx in data[:test_samples]:
            pred = self.nn.forward(nn_in)
            if np.argmax(pred) == label_idx:
                correct += 1
        
        accuracy = (correct / test_samples) * 100
        print(f"\n✅ Training Complete!")
        print(f"   Final Loss: {losses[-1]:.4f}")
        print(f"   Test Accuracy: {accuracy:.1f}%")
        
        if accuracy < 40:
            print("\n⚠️  WARNING: Low accuracy! NN may hurt performance.")
            print("   Recommendation: Collect more data and retrain.")
        elif accuracy < 60:
            print("\n💡 TIP: Accuracy is moderate. Hybrid may slightly improve.")
        else:
            print("\n🎯 EXCELLENT! NN is well-trained. Hybrid should outperform!")
        
        print(f"{'='*70}\n")
        
        return accuracy  


class GeneticOptimizer:
    """GA for parameter optimization"""
    def __init__(self, pop_size=20, gens=10):
        self.pop_size = pop_size
        self.gens = gens

    def create(self):
        return {
            'obstacle': random.uniform(0.5, 1.0),
            'goal': random.uniform(0.7, 1.0),
            'smooth': random.uniform(0.4, 0.8)
        }

    def fitness(self, params, start, goal, grid):
        controller = HybridIntelligentSystem()
        controller.fuzzy.obstacle_w = params['obstacle']
        controller.fuzzy.goal_w = params['goal']
        controller.fuzzy.smooth_w = params['smooth']
        pos = list(start)
        visited = [tuple(start)]
        prev = None
        metrics = PerformanceMetrics()
        for step in range(300):
            inputs = ControllerInputs(pos, goal, grid)
            if pos[0] == goal[0] and pos[1] == goal[1]:
                metrics.finalize(True, start, goal)
                return metrics.get_overall_score()
            outputs = controller.decide(inputs, prev, visited, use_nn_weight=0, grid=grid)
            dx, dy = outputs.direction
            nx, ny = pos[0]+dx, pos[1]+dy
            if (0<=nx<grid.shape[1] and 0<=ny<grid.shape[0] and grid[ny,nx]!=1):
                pos = [nx, ny]
                visited.append(tuple(pos))
                prev = (dx, dy)
                metrics.update_step(pos, inputs, outputs)
            else:
                metrics.metrics['collisions'] += 1
                if metrics.metrics['collisions'] > 20:
                    break
        metrics.finalize(False, start, goal)
        return metrics.get_overall_score() * 0.5

    def optimize(self, start, goal, grid):
        print("\n" + "="*60)
        print("GENETIC ALGORITHM OPTIMIZATION")
        print("="*60)
        pop = [self.create() for _ in range(self.pop_size)]
        
        for gen in range(self.gens):
            fits = [self.fitness(ind, start, goal, grid) for ind in pop]
            
            # FIXED: Sort by fitness score only, not by dictionary
            paired = sorted(zip(fits, pop), key=lambda x: x[0], reverse=True)
            sorted_pop = [ind for _, ind in paired]
            fits = [fit for fit, _ in paired]
            
            print(f"Gen {gen+1:2d}: Best={fits[0]:6.1f}, Avg={np.mean(fits):6.1f}")
            elite = sorted_pop[:4]
            offspring = []
            
            while len(offspring) < self.pop_size - 4:
                p1, p2 = random.sample(elite + sorted_pop[:8], 2)
                child = {k: (p1[k]+p2[k])/2 for k in p1.keys()}
                if random.random() < 0.3:
                    k = random.choice(list(child.keys()))
                    child[k] += random.gauss(0, 0.1)
                    child[k] = max(0.3, min(1.0, child[k]))
                offspring.append(child)
            pop = elite + offspring
        
        best = pop[0]
        print("\n" + "="*60)
        print("OPTIMIZED PARAMETERS:")
        for k, v in best.items():
            print(f"  {k:10s}: {v:.4f}")
        print(f"  Fitness Score: {self.fitness(best, start, goal, grid):.2f}/100")
        print("="*60 + "\n")
        return best


class NavigationSimulator:
    """Main simulator with enhanced comparison capabilities"""
    def __init__(self):
        self.grid_size = 20
        self.maps = {
            'simple': self.make_simple(),
            'complex': self.make_complex()
        }
        self.map_name = 'simple'
        self.grid = self.maps[self.map_name]
        self.start = (1, 1)
        self.goal = (18, 18)
        self.system = HybridIntelligentSystem(self.grid_size)
        
        # MODIFIED: Add mode selection
        self.mode = "hybrid"  # Default mode: "fuzzy" or "hybrid"
        
        self.reset()
        self.running = False
        self.use_nn = True
        self.nn_weight = 0.3
        self.anim = None
        self.fuzzy_only_results = None
        self.hybrid_results = None
        self.trial_results = []
        self.setup_gui()

    def make_simple(self):
        g = np.zeros((self.grid_size, self.grid_size))
        for i in range(5, 15):
            g[i, 15] = 1
        for i in range(10, 18):
            g[10, i] = 1
        for i in range(2, 10):
            g[5, i] = 1
        return g

    def make_complex(self):
        g = np.zeros((self.grid_size, self.grid_size))
        for i in range(self.grid_size):
            g[0, i] = 1
            g[self.grid_size-1, i] = 1
            g[i, 0] = 1
            g[i, self.grid_size-1] = 1
        g[1, 1] = 0
        g[18, 18] = 0
        for i in range(2, 18, 2):
            for j in range(2, 18):
                if (i // 2) % 2 == 0:
                    if j not in [3, 4, 7, 8, 11, 12, 15, 16]:
                        g[j, i] = 1
                else:
                    if j not in [2, 5, 6, 9, 10, 13, 14, 17]:
                        g[j, i] = 1
        for i in range(2, 18, 2):
            for j in range(2, 18):
                if (i // 2) % 2 == 0:
                    if j not in [3, 4, 7, 8, 11, 12, 15, 16]:
                        g[i, j] = 1
                else:
                    if j not in [2, 5, 6, 9, 10, 13, 14, 17]:
                        g[i, j] = 1
        for i in range(6, 14):
            g[6, i] = 1
            g[13, i] = 1
        for i in range(6, 14):
            g[i, 6] = 1
            g[i, 13] = 1
        for i in range(8, 12):
            g[8, i] = 1
            g[11, i] = 1
        for i in range(8, 12):
            g[i, 8] = 1
            g[i, 11] = 1
        g[6, 10] = 0
        g[13, 9] = 0
        return g

    def reset(self):
        self.pos = list(self.start)
        self.path = [tuple(self.start)]
        self.visited_positions = [tuple(self.start)]
        self.prev_dir = None
        self.metrics = PerformanceMetrics()
        self.current_inputs = None
        self.current_outputs = None

    def step(self):
        """Modified to respect selected mode"""
        if self.metrics.metrics['success'] or self.metrics.metrics['steps_taken'] > 500:
            return False
        
        self.current_inputs = ControllerInputs(self.pos, self.goal, self.grid)
        
        # MODIFIED: Use selected mode
        if self.mode == "fuzzy":
            # Use fuzzy only (no NN, no GA parameters)
            use_nn_weight = 0
        else:  # hybrid mode
            # Use hybrid with current NN weight
            use_nn_weight = self.nn_weight if self.use_nn else 0
        
        self.current_outputs = self.system.decide(
            self.current_inputs,
            self.prev_dir,
            self.visited_positions,
            use_nn_weight=use_nn_weight,
            grid=self.grid
        )
        
        dx, dy = self.current_outputs.direction
        
        # Only train NN in hybrid mode
        if self.mode == "hybrid" and self.use_nn:
            nn_in = self.current_inputs.to_vector().reshape(1, -1)
            dirs = [(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1)]
            label = dirs.index((dx, dy))
            self.system.training_buffer.append((nn_in, label))
            
        nx, ny = self.pos[0]+dx, self.pos[1]+dy
        if (0 <= nx < self.grid.shape[1] and 0 <= ny < self.grid.shape[0] and
            self.grid[ny, nx] != 1):
            if dx != 0 and dy != 0:
                if (0 <= self.pos[0]+dx < self.grid.shape[1] and 
                    0 <= self.pos[1]+dy < self.grid.shape[0]):
                    if (self.grid[self.pos[1], self.pos[0]+dx] == 1 and 
                        self.grid[self.pos[1]+dy, self.pos[0]] == 1):
                        self.metrics.metrics['collisions'] += 1
                        return True
            self.pos = [nx, ny]
            self.path.append(tuple(self.pos))
            self.visited_positions.append(tuple(self.pos))
            self.prev_dir = (dx, dy)
            self.metrics.update_step(self.pos, self.current_inputs, self.current_outputs)
            if self.pos[0] == self.goal[0] and self.pos[1] == self.goal[1]:
                self.metrics.finalize(True, self.start, self.goal)
                return False
        else:
            self.metrics.metrics['collisions'] += 1
            if self.metrics.metrics['collisions'] > 3:
                valid_dirs = []
                for test_dx, test_dy in [(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1)]:
                    test_x, test_y = self.pos[0] + test_dx, self.pos[1] + test_dy
                    if (0 <= test_x < self.grid.shape[1] and 0 <= test_y < self.grid.shape[0] and
                        self.grid[test_y, test_x] != 1):
                        if test_dx != 0 and test_dy != 0:
                            if (0 <= self.pos[0]+test_dx < self.grid.shape[1] and 
                                0 <= self.pos[1]+test_dy < self.grid.shape[0]):
                                if (self.grid[self.pos[1], self.pos[0]+test_dx] == 1 and 
                                    self.grid[self.pos[1]+test_dy, self.pos[0]] == 1):
                                    continue
                        valid_dirs.append((test_dx, test_dy))
                if valid_dirs:
                    escape_dir = random.choice(valid_dirs)
                    self.pos[0] += escape_dir[0]
                    self.pos[1] += escape_dir[1]
                    self.path.append(tuple(self.pos))
                    self.visited_positions.append(tuple(self.pos))
                    self.prev_dir = escape_dir
            if self.metrics.metrics['collisions'] > 15:
                return False
        return True

    def run_comparison(self, num_trials=10):
        """Enhanced comparison between Fuzzy-only and Hybrid"""
        print("\n" + "="*80)
        print("🔬 COMPREHENSIVE PERFORMANCE COMPARISON")
        print("   Fuzzy Logic ONLY vs Hybrid (Fuzzy + GA + Neural Network)")
        print(f"   Running {num_trials} trials per approach - Please wait...")
        print("="*80 + "\n")
        
        original_nn_weight = self.nn_weight
        
        # Phase 1: Fuzzy Only
        print("🔵 PHASE 1: Testing FUZZY LOGIC ONLY")
        print("-" * 80)
        self.nn_weight = 0.0
        self.use_nn = False
        fuzzy_results = []
        fuzzy_paths = []
        
        for trial in range(num_trials):
            print(f"  Trial {trial + 1}/{num_trials}...", end=" ", flush=True)
            self.reset()
            steps = 0
            while self.step():
                steps += 1
                if steps % 50 == 0:
                    print(".", end="", flush=True)
                if self.metrics.metrics['steps_taken'] > 500:
                    break
            if not self.metrics.metrics['success']:
                self.metrics.finalize(False, self.start, self.goal)
            result = self.metrics.to_dict()
            pm = PerformanceMetrics()
            pm.metrics = result
            result['overall_score'] = pm.get_overall_score()
            fuzzy_results.append(result)
            fuzzy_paths.append(self.path.copy())
            status = "✓ Success" if result['success'] else "✗ Failed"
            print(f" {status} | Steps: {result['steps_taken']:3d} | Score: {result['overall_score']:5.1f}")
        
        # Phase 2: Hybrid (Fuzzy + GA + NN)
        print(f"\n🟩 PHASE 2: Testing HYBRID (Fuzzy + GA + Neural Network)")
        print("-" * 80)
        
        # First optimize with GA
        print("\n  🧬 Running Genetic Algorithm Optimization...")
        ga = GeneticOptimizer(pop_size=15, gens=8)
        best_params = ga.optimize(self.start, self.goal, self.grid)
        
        # Save original system state
        original_system = self.system
        original_mode = self.mode
        
        # Run hybrid trials with independent systems
        print("\n  Testing Hybrid System (each trial is independent):")
        self.nn_weight = original_nn_weight
        self.use_nn = True
        self.mode = "hybrid"
        hybrid_results = []
        hybrid_paths = []
        
        for trial in range(num_trials):
            print(f"  Trial {trial + 1}/{num_trials}...", end=" ", flush=True)
            
            # Create fresh hybrid system for each trial
            trial_system = HybridIntelligentSystem(self.grid_size)
            trial_system.fuzzy.obstacle_w = best_params['obstacle']
            trial_system.fuzzy.goal_w = best_params['goal']
            trial_system.fuzzy.smooth_w = best_params['smooth']
            trial_system.exploration_rate = 0.05  # Add some exploration
            
            # Replace system for this trial
            self.system = trial_system
            
            # Collect initial training data
            print("[Collect data...]", end=" ", flush=True)
            self.reset()
            warmup_steps = 0
            while warmup_steps < 50:
                if not self.step():
                    break
                warmup_steps += 1
            
            # Train NN for this trial
            if len(self.system.training_buffer) > 30:
                print("[Train NN...]", end=" ", flush=True)
                self.system.train_nn_batch(epochs=20)
            
            # Run the actual trial
            print("[Run trial...]", end=" ", flush=True)
            self.reset()
            steps = 0
            path_for_trial = []
            
            while True:
                if not self.step():
                    break
                steps += 1
                path_for_trial.append(tuple(self.pos))
                if self.metrics.metrics['steps_taken'] > 500:
                    break
                if self.pos[0] == self.goal[0] and self.pos[1] == self.goal[1]:
                    break
            
            if not self.metrics.metrics['success']:
                self.metrics.finalize(False, self.start, self.goal)
            
            result = self.metrics.to_dict()
            pm = PerformanceMetrics()
            pm.metrics = result
            result['overall_score'] = pm.get_overall_score()
            hybrid_results.append(result)
            hybrid_paths.append(path_for_trial.copy())
            
            status = "✓ Success" if result['success'] else "✗ Failed"
            print(f" {status} | Steps: {result['steps_taken']:3d} | Score: {result['overall_score']:5.1f}")
        
        # Restore original system and mode
        self.system = original_system
        self.mode = original_mode
        
        self.fuzzy_only_results = fuzzy_results
        self.hybrid_results = hybrid_results
        
        # Comprehensive analysis
        print("\n" + "="*80)
        print("📊 Generating detailed comparison report...")
        print("="*80)
        self.print_detailed_comparison(fuzzy_results, hybrid_results)
        self.plot_comprehensive_comparison(fuzzy_results, hybrid_results, fuzzy_paths, hybrid_paths)
        
        self.nn_weight = original_nn_weight

    def print_detailed_comparison(self, fuzzy_results, hybrid_results):
        """Print comprehensive statistical comparison"""
        print("\n" + "="*80)
        print("📊 DETAILED STATISTICAL COMPARISON")
        print("="*80)
        
        metrics_to_compare = [
            ('success', 'Success Rate (%)', True, 100),
            ('steps_taken', 'Steps to Goal', False, 1),
            ('collisions', 'Collision Count', False, 1),
            ('path_efficiency', 'Path Efficiency (%)', True, 1),
            ('straightness_ratio', 'Path Straightness (%)', True, 1),
            ('smoothness_score', 'Smoothness Score (%)', True, 1),
            ('avg_obstacle_clearance', 'Avg Clearance', True, 1),
            ('sharp_turns', 'Sharp Turns (>90°)', False, 1),
            ('overall_score', 'Overall Score', True, 1)
        ]
        
        print(f"\n{'Metric':<25} | {'Fuzzy Only':<20} | {'Hybrid (F+GA+NN)':<20} | {'Improvement':<15} | {'p-value':<10}")
        print("-" * 105)
        
        for metric_key, metric_name, higher_better, multiplier in metrics_to_compare:
            fuzzy_vals = [r[metric_key] * multiplier for r in fuzzy_results]
            hybrid_vals = [r[metric_key] * multiplier for r in hybrid_results]
            
            fuzzy_mean = np.mean(fuzzy_vals)
            fuzzy_std = np.std(fuzzy_vals)
            hybrid_mean = np.mean(hybrid_vals)
            hybrid_std = np.std(hybrid_vals)
            
            # Calculate improvement
            if higher_better:
                improvement = hybrid_mean - fuzzy_mean
                pct_improvement = (improvement / (fuzzy_mean + 0.001)) * 100
            else:
                improvement = fuzzy_mean - hybrid_mean
                pct_improvement = (improvement / (fuzzy_mean + 0.001)) * 100
            
            # Statistical significance test
            t_stat, p_value = simple_ttest(fuzzy_vals, hybrid_vals)
            significance = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""
            
            fuzzy_str = f"{fuzzy_mean:6.1f} ± {fuzzy_std:4.1f}"
            hybrid_str = f"{hybrid_mean:6.1f} ± {hybrid_std:4.1f}"
            improvement_str = f"{improvement:+6.1f} ({pct_improvement:+5.1f}%)"
            
            print(f"{metric_name:<25} | {fuzzy_str:<20} | {hybrid_str:<20} | {improvement_str:<15} | {p_value:>8.4f}{significance}")
        
        print("\nSignificance: *** p<0.01  ** p<0.05  * p<0.1")
        
        # Winner determination
        fuzzy_score = np.mean([r['overall_score'] for r in fuzzy_results])
        hybrid_score = np.mean([r['overall_score'] for r in hybrid_results])
        
        print("\n" + "="*80)
        if hybrid_score > fuzzy_score:
            improvement = ((hybrid_score - fuzzy_score) / fuzzy_score) * 100
            print(f"🏆 WINNER: HYBRID (Fuzzy + GA + NN) by {improvement:.1f}%")
        else:
            print(f"🏆 WINNER: FUZZY LOGIC ONLY")
        print("="*80)
        
        # Summary statistics
        print("\n📈 SUMMARY STATISTICS:")
        print(f"  Fuzzy Only:  {sum(r['success'] for r in fuzzy_results)}/{len(fuzzy_results)} successes ({fuzzy_score:.1f}/100 avg)")
        print(f"  Hybrid:      {sum(r['success'] for r in hybrid_results)}/{len(hybrid_results)} successes ({hybrid_score:.1f}/100 avg)")
        print()

    def plot_comprehensive_comparison(self, fuzzy_results, hybrid_results, fuzzy_paths, hybrid_paths):
        """Create comprehensive comparison visualizations"""
        print("\n📊 Generating comparison plots...")
        
        # Create a new figure with smaller subplots
        fig = plt.figure(figsize=(18, 12))
        # Reduce spacing between subplots
        gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.4)  # Increased hspace
        
        fig.suptitle('FUZZY LOGIC vs HYBRID COMPARISON', fontsize=16, fontweight='bold')
        
        # CHANGED: Dark green for hybrid
        colors = ['#3498db', '#006400']  # Blue for fuzzy, Dark green for hybrid
        
        # 1. Success Rate
        ax1 = fig.add_subplot(gs[0, 0])
        success_rates = [
            sum(r['success'] for r in fuzzy_results) / len(fuzzy_results) * 100,
            sum(r['success'] for r in hybrid_results) / len(hybrid_results) * 100
        ]
        bars = ax1.bar(['Fuzzy', 'Hybrid'], success_rates, 
                      color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Success Rate (%)', fontweight='bold', fontsize=10)
        ax1.set_title('Success Rate', fontweight='bold', fontsize=11)
        ax1.set_ylim([0, 105])
        ax1.grid(True, alpha=0.3, axis='y')
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{success_rates[i]:.1f}%', ha='center', va='bottom', 
                    fontweight='bold', fontsize=9)
        
        # 2. Steps Comparison - FIXED: Remove all error bars and ± symbols
        ax2 = fig.add_subplot(gs[0, 1])
        fuzzy_steps = [r['steps_taken'] for r in fuzzy_results]
        hybrid_steps = [r['steps_taken'] for r in hybrid_results]
        
        # Calculate averages
        avg_fuzzy_steps = np.mean(fuzzy_steps)
        avg_hybrid_steps = np.mean(hybrid_steps)
        
        # Create bar plot
        avg_steps = [avg_fuzzy_steps, avg_hybrid_steps]
        
        bars = ax2.bar(['Fuzzy', 'Hybrid'], avg_steps, 
                      color=colors, alpha=0.7, edgecolor='black', width=0.6)
        
        ax2.set_ylabel('Average Steps', fontweight='bold', fontsize=10)
        ax2.set_title('Steps to Goal', fontweight='bold', fontsize=11)
        ax2.set_ylim([0, max(avg_steps) * 1.2])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # NO error bars added
        
        # Add value labels - just the average value
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{avg_steps[i]:.0f}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=9)
        
        # 3. Collisions Comparison - FIXED: Remove all error bars and ± symbols
        ax3 = fig.add_subplot(gs[0, 2])
        fuzzy_coll = [r['collisions'] for r in fuzzy_results]
        hybrid_coll = [r['collisions'] for r in hybrid_results]
        avg_coll = [np.mean(fuzzy_coll), np.mean(hybrid_coll)]
        
        bars = ax3.bar(['Fuzzy', 'Hybrid'], avg_coll, 
                      color=colors, alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Average Collisions', fontweight='bold', fontsize=10)
        ax3.set_title('Collision Analysis', fontweight='bold', fontsize=11)
        
        # Set y-axis to start from 0
        y_max = max(avg_coll) * 1.3 if max(avg_coll) > 0 else 1
        ax3.set_ylim([0, y_max])
        
        ax3.grid(True, alpha=0.3, axis='y')
        
        # NO error bars added
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{avg_coll[i]:.1f}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=9)
        
        # 4. Overall Score Distribution - FIXED: Remove all error bars and ± symbols
        ax4 = fig.add_subplot(gs[0, 3])
        fuzzy_scores = [r['overall_score'] for r in fuzzy_results]
        hybrid_scores = [r['overall_score'] for r in hybrid_results]
        
        # Calculate averages
        avg_fuzzy_scores = np.mean(fuzzy_scores)
        avg_hybrid_scores = np.mean(hybrid_scores)
        
        # Create bar plot
        avg_scores = [avg_fuzzy_scores, avg_hybrid_scores]
        bars = ax4.bar(['Fuzzy', 'Hybrid'], avg_scores, 
                      color=colors, alpha=0.7, edgecolor='black', width=0.6)
        
        ax4.set_ylabel('Average Overall Score', fontweight='bold', fontsize=10)
        ax4.set_title('Overall Performance', fontweight='bold', fontsize=11)
        ax4.set_ylim([0, 105])
        ax4.grid(True, alpha=0.3, axis='y')
        
        # NO error bars added
        
        # Add value labels - just the average value
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{avg_scores[i]:.1f}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=9)
        
        # 5. Path Efficiency - FIXED: Remove all error bars and ± symbols
        ax5 = fig.add_subplot(gs[1, 0])
        fuzzy_eff = [r['path_efficiency'] for r in fuzzy_results if r['path_efficiency'] > 0]
        hybrid_eff = [r['path_efficiency'] for r in hybrid_results if r['path_efficiency'] > 0]
        if fuzzy_eff and hybrid_eff:
            avg_eff = [np.mean(fuzzy_eff), np.mean(hybrid_eff)]
            
            bars = ax5.bar(['Fuzzy', 'Hybrid'], avg_eff, 
                          color=colors, alpha=0.7, edgecolor='black')
            ax5.set_ylabel('Path Efficiency (%)', fontweight='bold', fontsize=10)
            ax5.set_title('Path Efficiency', fontweight='bold', fontsize=11)
            ax5.set_ylim([0, max(avg_eff) * 1.3])
            ax5.grid(True, alpha=0.3, axis='y')
            
            # NO error bars added
            
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{avg_eff[i]:.1f}%', ha='center', va='bottom', 
                        fontweight='bold', fontsize=9)
        
        # 6. Smoothness Score - FIXED: Remove all error bars and ± symbols
        ax6 = fig.add_subplot(gs[1, 1])
        fuzzy_smooth = [r['smoothness_score'] for r in fuzzy_results]
        hybrid_smooth = [r['smoothness_score'] for r in hybrid_results]
        avg_smooth = [np.mean(fuzzy_smooth), np.mean(hybrid_smooth)]
        
        bars = ax6.bar(['Fuzzy', 'Hybrid'], avg_smooth, 
                      color=colors, alpha=0.7, edgecolor='black')
        ax6.set_ylabel('Smoothness Score (%)', fontweight='bold', fontsize=10)
        ax6.set_title('Path Smoothness', fontweight='bold', fontsize=11)
        ax6.set_ylim([0, max(avg_smooth) * 1.3])
        ax6.grid(True, alpha=0.3, axis='y')
        
        # NO error bars added
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{avg_smooth[i]:.1f}%', ha='center', va='bottom', 
                    fontweight='bold', fontsize=9)
        
        # 7. Trial-by-Trial Comparison - Show only first 5 trials
        ax7 = fig.add_subplot(gs[1, 2:])
        # Show only first 5 trials
        num_trials_to_show = 5
        trials_to_show = min(num_trials_to_show, len(fuzzy_results))
        trials = range(1, trials_to_show + 1)
        
        # Get only first 5 trials
        fuzzy_scores_5 = fuzzy_scores[:trials_to_show]
        hybrid_scores_5 = hybrid_scores[:trials_to_show]
        
        # Find the minimum score among all displayed trials
        all_scores = fuzzy_scores_5 + hybrid_scores_5
        min_score = min(all_scores)
        max_score = max(all_scores)
        
        # Set y-axis limits starting from slightly below the minimum score
        # to make differences more visible
        y_min = max(0, min_score - 5)  # Start 5 points below min, but not below 0
        y_max = min(105, max_score + 5)  # End 5 points above max, but not above 105
        
        # Plot individual trial scores as scatter points
        ax7.scatter(trials, fuzzy_scores_5, color=colors[0], s=80, 
                   alpha=0.7, label='Fuzzy', edgecolors='black', linewidth=0.5)
        ax7.scatter(trials, hybrid_scores_5, color=colors[1], s=80, 
                   alpha=0.7, label='Hybrid', edgecolors='black', linewidth=0.5, marker='s')
        
        # Connect points with faint lines
        ax7.plot(trials, fuzzy_scores_5, color=colors[0], linewidth=1, alpha=0.3)
        ax7.plot(trials, hybrid_scores_5, color=colors[1], linewidth=1, alpha=0.3)
        
        # Add value labels for each trial
        for i, trial in enumerate(trials):
            # Fuzzy scores
            ax7.text(trial, fuzzy_scores_5[i] + (y_max - y_min) * 0.02, f'{fuzzy_scores_5[i]:.0f}', 
                    ha='center', va='bottom', fontsize=8, color=colors[0])
            # Hybrid scores
            ax7.text(trial, hybrid_scores_5[i] - (y_max - y_min) * 0.02, f'{hybrid_scores_5[i]:.0f}', 
                    ha='center', va='top', fontsize=8, color=colors[1])
        
        ax7.set_xlabel('Trial Number', fontweight='bold', fontsize=11)
        ax7.set_ylabel('Overall Score', fontweight='bold', fontsize=11)
        ax7.set_title('Trial-by-Trial Performance', fontweight='bold', fontsize=12)
        ax7.set_ylim([y_min, y_max])  # Dynamic y-axis limits based on data
        ax7.set_xticks(trials)
        ax7.legend(loc='best', fontsize=9)
        ax7.grid(True, alpha=0.3)
        
        # 8 & 9. Path Visualization Comparison (Best Trials)
        fuzzy_best_idx = np.argmax(fuzzy_scores)
        hybrid_best_idx = np.argmax(hybrid_scores)
        
        ax8 = fig.add_subplot(gs[2, 0:2])
        self._plot_path_on_grid(ax8, fuzzy_paths[fuzzy_best_idx], 
                               f'Fuzzy Only - Best Trial (Score: {fuzzy_scores[fuzzy_best_idx]:.1f})',
                               colors[0])
        
        ax9 = fig.add_subplot(gs[2, 2:])
        self._plot_path_on_grid(ax9, hybrid_paths[hybrid_best_idx],
                               f'Hybrid - Best Trial (Score: {hybrid_scores[hybrid_best_idx]:.1f})',
                               colors[1])
        
        plt.tight_layout()
        
        # Show the plot in a new window
        print("✅ Plots generated! Displaying comparison window...")
        plt.show(block=False)
        plt.pause(0.1)
        
        print("\n💡 TIP: The comparison plot window is now open!")
        print("   Close it to return to the main simulation.\n")

    def _plot_path_on_grid(self, ax, path, title, color):
        """Helper to plot a path on grid"""
        ax.clear()
        ax.set_xlim(-0.5, self.grid_size-0.5)
        ax.set_ylim(-0.5, self.grid_size-0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.grid(alpha=0.3)
        ax.set_title(title, fontweight='bold', fontsize=11)
        
        # Draw obstacles
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.grid[y,x] == 1:
                    rect = patches.Rectangle((x-0.5,y-0.5), 1, 1,
                                            fc='dimgray', ec='black', lw=0.5)
                    ax.add_patch(rect)
        
        # Draw path
        if len(path) > 1:
            pa = np.array(path)
            ax.plot(pa[:,0], pa[:,1], '-', color=color, lw=2, alpha=0.7)
            ax.plot(pa[0,0], pa[0,1], 's', color='green', markersize=10, 
                   markeredgecolor='darkgreen', markeredgewidth=2, label='Start')
            ax.plot(pa[-1,0], pa[-1,1], 's', color='red', markersize=10,
                   markeredgecolor='darkred', markeredgewidth=2, label='Goal')
            
            # Add direction arrows (fewer to reduce clutter)
            for i in range(0, len(pa)-1, max(1, len(pa)//8)):
                if i < len(pa)-1:
                    dx = pa[i+1,0] - pa[i,0]
                    dy = pa[i+1,1] - pa[i,1]
                    if abs(dx) > 0 or abs(dy) > 0:
                        ax.arrow(pa[i,0], pa[i,1], dx*0.6, dy*0.6,
                               head_width=0.25, head_length=0.25,
                               fc=color, ec=color, alpha=0.6)
            
            ax.legend(loc='upper right', fontsize=8)
            ax.text(0.02, 0.98, f'Steps: {len(path)}', 
                   transform=ax.transAxes, fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
    def setup_gui(self):
        self.fig, self.ax = plt.subplots(figsize=(13, 10))
        plt.subplots_adjust(left=0.1, bottom=0.35)
        self.draw()
        
        # Row 1 buttons
        self.btn_start = Button(plt.axes([0.10,0.26,0.08,0.04]), 'Start')
        self.btn_reset = Button(plt.axes([0.19,0.26,0.08,0.04]), 'Reset')
        self.btn_ga = Button(plt.axes([0.28,0.26,0.10,0.04]), 'Optimize GA')
        self.btn_train = Button(plt.axes([0.39,0.26,0.08,0.04]), 'Train NN')
        self.btn_map = Button(plt.axes([0.48,0.26,0.07,0.04]), 'Change Map')
        self.btn_eval = Button(plt.axes([0.56,0.26,0.10,0.04]), 'Evaluate')
        self.btn_compare = Button(plt.axes([0.67,0.26,0.13,0.04]), 'Compare Fuzzy vs Hybrid')
        
        # MODIFIED: Add mode selector radio buttons
        self.mode_ax = plt.axes([0.81, 0.26, 0.08, 0.08])
        self.mode_selector = RadioButtons(self.mode_ax, ['Fuzzy', 'Hybrid'], 
                                         active=1)  # Default to Hybrid (index 1)
        self.mode_selector.on_clicked(self.change_mode)
        
        self.btn_start.on_clicked(self.toggle)
        self.btn_reset.on_clicked(self.reset_sim)
        self.btn_ga.on_clicked(self.run_ga)
        self.btn_train.on_clicked(self.train_nn)
        self.btn_map.on_clicked(self.switch_map)
        self.btn_eval.on_clicked(self.run_eval)
        self.btn_compare.on_clicked(self.run_comp)
        
        self.sl_obs = Slider(plt.axes([0.15,0.18,0.60,0.02]), 'Obstacle Weight',
                            0, 1, valinit=self.system.fuzzy.obstacle_w)
        self.sl_goal = Slider(plt.axes([0.15,0.14,0.60,0.02]), 'Goal Weight',
                             0, 1, valinit=self.system.fuzzy.goal_w)
        self.sl_smooth = Slider(plt.axes([0.15,0.10,0.60,0.02]), 'Smoothness Weight',
                               0, 1, valinit=self.system.fuzzy.smooth_w)
        self.sl_nn = Slider(plt.axes([0.15,0.06,0.60,0.02]), 'NN Weight',
                          0, 0.5, valinit=self.nn_weight)
        
        self.sl_obs.on_changed(lambda v: setattr(self.system.fuzzy, 'obstacle_w', v))
        self.sl_goal.on_changed(lambda v: setattr(self.system.fuzzy, 'goal_w', v))
        self.sl_smooth.on_changed(lambda v: setattr(self.system.fuzzy, 'smooth_w', v))
        self.sl_nn.on_changed(lambda v: setattr(self, 'nn_weight', v))
        
        self.txt = self.fig.text(0.02, 0.97, '', fontsize=8, family='monospace',
                                verticalalignment='top',
                                bbox=dict(boxstyle='round', fc='wheat', alpha=0.9))
        self.update_stats()

    def draw(self):
        self.ax.clear()
        self.ax.set_xlim(-0.5, self.grid_size-0.5)
        self.ax.set_ylim(-0.5, self.grid_size-0.5)
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()
        self.ax.grid(alpha=0.3)
        # MODIFIED: Include current mode in title
        self.ax.set_title(f'Mobile Robot Navigation - {self.map_name.capitalize()} Map - Mode: {self.mode.upper()}',
                         fontsize=13, fontweight='bold')
        
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.grid[y,x] == 1:
                    rect = patches.Rectangle((x-0.5,y-0.5), 1, 1,
                                            fc='dimgray', ec='black', lw=0.5)
                    self.ax.add_patch(rect)
        
        if len(self.path) > 1:
            pa = np.array(self.path)
            self.ax.plot(pa[:,0], pa[:,1], 'b-', lw=2, alpha=0.7, label='Path')
            for i, (x, y) in enumerate(pa):
                color_intensity = i / len(pa)
                color = (color_intensity, 0, 1-color_intensity)
                self.ax.plot(x, y, 'o', color=color, markersize=6, alpha=0.8)
            self.ax.plot(pa[0,0], pa[0,1], 's', color='green', markersize=15, 
                        markeredgecolor='darkgreen', markeredgewidth=2, label='Start')
            self.ax.plot(self.goal[0], self.goal[1], 's', color='red', markersize=15,
                        markeredgecolor='darkred', markeredgewidth=2, label='Goal')
            self.ax.plot(self.pos[0], self.pos[1], 'o', color='yellow', markersize=12,
                        markeredgecolor='orange', markeredgewidth=2, label='Current')
        
        if len(self.path) <= 1:
            self.ax.plot(self.start[0], self.start[1], 's', color='green', markersize=15,
                        markeredgecolor='darkgreen', markeredgewidth=2, label='Start')
            self.ax.plot(self.goal[0], self.goal[1], 's', color='red', markersize=15,
                        markeredgecolor='darkred', markeredgewidth=2, label='Goal')
        
        distance = np.hypot(self.goal[0] - self.pos[0], self.goal[1] - self.pos[1])
        self.ax.text(0.02, 0.98, f'Distance to Goal: {distance:.1f}', 
                    transform=self.ax.transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        self.ax.legend(loc='upper right', fontsize=9)
        plt.draw()

    def update_stats(self):
        m = self.metrics.metrics
        score = PerformanceMetrics()
        score.metrics = m
        # MODIFIED: Include mode in stats display
        txt = (f"Mode:          {self.mode.upper()}\n"
               f"Status:        {'SUCCESS ✓' if m['success'] else 'Running...'}\n"
               f"Steps:         {m['steps_taken']}\n"
               f"Collisions:    {m['collisions']}\n"
               f"Smoothness:    {m['smoothness_score']:.1f}%\n"
               f"Path Efficiency:{m['path_efficiency']:.1f}%\n"
               f"Overall Score: {score.get_overall_score():.1f}/100\n"
               f"Map:           {self.map_name}\n"
               f"NN Weight:     {self.nn_weight:.2f}")
        self.txt.set_text(txt)

    def toggle(self, e):
        self.running = not self.running
        if self.running and not self.metrics.metrics['success']:
            if self.anim is not None:
                self.anim.event_source.stop()
            self.anim = FuncAnimation(self.fig, self.animate, interval=50,
                                     repeat=False, cache_frame_data=False)
            plt.draw()
        elif self.anim is not None:
            self.anim.event_source.stop()

    def animate(self, frame):
        if self.running and not self.metrics.metrics['success']:
            cont = self.step()
            self.draw()
            self.update_stats()
            if not cont:
                self.running = False
                if self.anim is not None:
                    self.anim.event_source.stop()

    def reset_sim(self, e=None):
        self.running = False
        if self.anim is not None:
            self.anim.event_source.stop()
        self.reset()
        self.draw()
        self.update_stats()

    def switch_map(self, e):
        maps = ['simple', 'complex']
        idx = (maps.index(self.map_name) + 1) % 2
        self.map_name = maps[idx]
        self.grid = self.maps[self.map_name]
        self.reset_sim()

    def change_mode(self, label):
        """Handle mode change from radio buttons"""
        self.mode = label.lower()
        print(f"\n🔄 Switching to {self.mode.upper()} mode")
        if self.mode == "fuzzy":
            print("   Using pure fuzzy logic only (no NN, no GA)")
        else:
            print("   Using hybrid system (Fuzzy + GA + NN)")
        self.draw()
        self.update_stats()

    def run_ga(self, e):
        was_running = self.running
        if self.running:
            self.running = False
            if self.anim is not None:
                self.anim.event_source.stop()
        plt.pause(0.1)
        ga = GeneticOptimizer(pop_size=15, gens=8)
        best = ga.optimize(self.start, self.goal, self.grid)
        self.system.fuzzy.obstacle_w = best['obstacle']
        self.system.fuzzy.goal_w = best['goal']
        self.system.fuzzy.smooth_w = best['smooth']
        self.sl_obs.set_val(best['obstacle'])
        self.sl_goal.set_val(best['goal'])
        self.sl_smooth.set_val(best['smooth'])
        print("✓ Fuzzy parameters optimized!\n")
        if was_running:
            self.reset_sim()

    def train_nn(self, e):
        was_running = self.running
        if self.running:
            self.running = False
            if self.anim is not None:
                self.anim.event_source.stop()
        plt.pause(0.1)
        
        # CHECK if there's enough data
        if len(self.system.training_buffer) < 50:
            print("\n" + "!"*70)
            print("⚠️  NOT ENOUGH TRAINING DATA!")
            print(f"   Current buffer: {len(self.system.training_buffer)} samples")
            print("   Required: 50+ samples")
            print("\n💡 SOLUTION: Run the simulation first to collect data:")
            print("   1. Click 'Start' and let it run for 100+ steps")
            print("   2. Click 'Reset' and repeat 2-3 times")
            print("   3. Then click 'Train NN' again")
            print("!"*70 + "\n")
            return
        
        # Collect more data if needed
        print("\n" + "="*70)
        print("🧠 PREPARING NEURAL NETWORK TRAINING")
        print("="*70)
        
        if len(self.system.training_buffer) < 200:
            print(f"📊 Current training data: {len(self.system.training_buffer)} samples")
            print("🔄 Collecting more training data (need 200+ for best results)...")
            
            original_nn = self.use_nn
            self.use_nn = False  # Collect with fuzzy only
            
            for run in range(3):
                print(f"   Collection run {run+1}/3...", end=" ", flush=True)
                self.reset()
                steps = 0
                while self.step() and steps < 150:
                    steps += 1
                print(f"✓ ({steps} steps)")
            
            self.use_nn = original_nn
            print(f"✅ Data collection complete! Total samples: {len(self.system.training_buffer)}\n")
        
        # Now train
        self.system.train_nn_batch(epochs=50)
        
        if was_running:
            self.reset_sim()

    def run_eval(self, e):
        """Run evaluation with multiple trials"""
        was_running = self.running
        if self.running:
            self.running = False
            if self.anim is not None:
                self.anim.event_source.stop()
        plt.pause(0.1)
        
        print("\n" + "="*70)
        print(f"RUNNING MULTI-TRIAL EVALUATION (5 trials) - Mode: {self.mode.upper()}")
        print("="*70 + "\n")
        
        self.trial_results = []
        for trial in range(5):
            print(f"\n--- Trial {trial + 1}/5 ---")
            self.reset()
            while self.step():
                pass
            if not self.metrics.metrics['success']:
                self.metrics.finalize(False, self.start, self.goal)
            self.trial_results.append(self.metrics.to_dict())
        
        self.print_statistical_summary()
        self.plot_trial_comparison()
        
        if was_running:
            self.reset_sim()

    def run_comp(self, e):
        """Run comparison between Fuzzy and Hybrid"""
        was_running = self.running
        if self.running:
            self.running = False
            if self.anim is not None:
                self.anim.event_source.stop()
        
        # Force update the main window
        plt.pause(0.5)
        
        print("\n" + "="*80)
        print("🔬 STARTING COMPARISON - This will take a few minutes...")
        print("="*80)
        
        try:
            self.run_comparison(10)
            print("\n✅ Comparison completed successfully!")
        except Exception as ex:
            print(f"\n❌ Error during comparison: {ex}")
            import traceback
            traceback.print_exc()
        
        if was_running:
            self.reset_sim()

    def print_statistical_summary(self):
        """Print statistical analysis of multiple trials"""
        if not self.trial_results:
            return
        
        print("\n" + "="*70)
        print(f"STATISTICAL SUMMARY OF TRIALS - Mode: {self.mode.upper()}")
        print("="*70)
        
        success_rate = sum(r['success'] for r in self.trial_results) / len(self.trial_results) * 100
        steps = [r['steps_taken'] for r in self.trial_results]
        collisions = [r['collisions'] for r in self.trial_results]
        efficiency = [r['path_efficiency'] for r in self.trial_results if r['path_efficiency'] > 0]
        smoothness = [r['smoothness_score'] for r in self.trial_results]
        scores = []
        
        for r in self.trial_results:
            pm = PerformanceMetrics()
            pm.metrics = r
            scores.append(pm.get_overall_score())
        
        print(f"\n📊 Success Rate: {success_rate:.1f}% ({sum(r['success'] for r in self.trial_results)}/{len(self.trial_results)})")
        print(f"\n📈 Steps Taken:")
        print(f"   Mean:   {np.mean(steps):.1f}")
        print(f"   Std:    {np.std(steps):.1f}")
        print(f"   Min:    {np.min(steps)}")
        print(f"   Max:    {np.max(steps)}")
        print(f"\n💥 Collisions:")
        print(f"   Mean:   {np.mean(collisions):.1f}")
        print(f"   Std:    {np.std(collisions):.1f}")
        print(f"   Total:  {sum(collisions)}")
        if efficiency:
            print(f"\n⚡ Path Efficiency:")
            print(f"   Mean:   {np.mean(efficiency):.1f}%")
            print(f"   Std:    {np.std(efficiency):.1f}%")
        print(f"\n🔄 Smoothness:")
        print(f"   Mean:   {np.mean(smoothness):.1f}%")
        print(f"   Std:    {np.std(smoothness):.1f}%")
        print(f"\n🏆 Overall Scores:")
        print(f"   Mean:   {np.mean(scores):.1f}/100")
        print(f"   Std:    {np.std(scores):.1f}")
        print(f"   Best:   {np.max(scores):.1f}/100")
        print("="*70 + "\n")

    def plot_trial_comparison(self):
        """Plot comparison graphs for multiple trials"""
        if len(self.trial_results) < 2:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f'Multi-Trial Performance Analysis - Mode: {self.mode.upper()}', 
                    fontsize=14, fontweight='bold')
        
        trials = range(1, len(self.trial_results) + 1)
        
        # Success rate
        successes = [1 if r['success'] else 0 for r in self.trial_results]
        axes[0,0].bar(trials, successes, color=['green' if s else 'red' for s in successes])
        axes[0,0].set_ylabel('Success')
        axes[0,0].set_xlabel('Trial')
        axes[0,0].set_title('Success/Failure')
        axes[0,0].set_ylim([0, 1.2])
        
        # Steps
        steps = [r['steps_taken'] for r in self.trial_results]
        axes[0,1].plot(trials, steps, 'bo-', linewidth=2)
        axes[0,1].axhline(np.mean(steps), color='r', linestyle='--', label=f'Mean: {np.mean(steps):.1f}')
        axes[0,1].set_ylabel('Steps')
        axes[0,1].set_xlabel('Trial')
        axes[0,1].set_title('Steps Taken')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Collisions
        collisions = [r['collisions'] for r in self.trial_results]
        axes[0,2].bar(trials, collisions, color='salmon')
        axes[0,2].set_ylabel('Collisions')
        axes[0,2].set_xlabel('Trial')
        axes[0,2].set_title('Collision Count')
        axes[0,2].grid(True, alpha=0.3, axis='y')
        
        # Path efficiency
        efficiency = [r['path_efficiency'] if r['path_efficiency'] > 0 else 0
                     for r in self.trial_results]
        axes[1,0].plot(trials, efficiency, 'go-', linewidth=2)
        axes[1,0].axhline(np.mean([e for e in efficiency if e > 0]),
                         color='r', linestyle='--', label=f'Mean: {np.mean([e for e in efficiency if e > 0]):.1f}%')
        axes[1,0].set_ylabel('Efficiency (%)')
        axes[1,0].set_xlabel('Trial')
        axes[1,0].set_title('Path Efficiency')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Smoothness
        smoothness = [r['smoothness_score'] for r in self.trial_results]
        axes[1,1].plot(trials, smoothness, 'mo-', linewidth=2)
        axes[1,1].axhline(np.mean(smoothness), color='r', linestyle='--',
                         label=f'Mean: {np.mean(smoothness):.1f}%')
        axes[1,1].set_ylabel('Smoothness (%)')
        axes[1,1].set_xlabel('Trial')
        axes[1,1].set_title('Path Smoothness')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # Overall scores
        scores = []
        for r in self.trial_results:
            pm = PerformanceMetrics()
            pm.metrics = r
            scores.append(pm.get_overall_score())
        
        axes[1,2].bar(trials, scores, color='skyblue')
        axes[1,2].axhline(np.mean(scores), color='r', linestyle='--',
                         label=f'Mean: {np.mean(scores):.1f}')
        axes[1,2].set_ylabel('Score')
        axes[1,2].set_xlabel('Trial')
        axes[1,2].set_title('Overall Performance Score')
        axes[1,2].set_ylim([0, 105])
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()

    def run(self):
        plt.show()


if __name__ == "__main__":
    print("="*80)
    print("ENHANCED FUZZY vs HYBRID NAVIGATION COMPARISON")
    print("="*80)
    print("\n🎮 CONTROLS:")
    print("  • Use the radio buttons (Fuzzy/Hybrid) to switch between modes")
    print("  • 'Start': Run the simulation with selected mode")
    print("  • 'Optimize GA': Tune fuzzy parameters (only affects hybrid mode)")
    print("  • 'Train NN': Train neural network (only affects hybrid mode)")
    print("  • 'Compare Fuzzy vs Hybrid': Full comparison between both modes")
    print("\n🔬 MODES:")
    print("  1. FUZZY: Pure fuzzy logic controller only")
    print("  2. HYBRID: Fuzzy + Genetic Algorithm + Neural Network")
    print("\n💡 TIP: Start with Fuzzy mode to see baseline, then switch to Hybrid!")
    print("="*80 + "\n")
        
    sim = NavigationSimulator()
    sim.run()