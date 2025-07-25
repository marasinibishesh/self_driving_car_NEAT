# NEAT Self-Driving Car Simulation

## Overview
This project implements a self-driving car simulation using the NEAT (NeuroEvolution of Augmenting Topologies) algorithm. Cars learn to navigate race tracks through evolutionary algorithms, improving their driving capabilities over generations. The simulation includes multiple tracks with varying difficulties and provides visual feedback of the learning process.

## Key Features
- 🚗 Neural network-controlled cars with sensor inputs
- � 4 different race tracks with unique layouts
- 🧬 Genetic algorithm implementation for evolving car behaviors
- 📊 Real-time performance statistics and visualizations
- 💾 Automatic model saving and backup system
- 🖥️ Interactive PyGame-based visualization

## What is NEAT?
NEAT (NeuroEvolution of Augmenting Topologies) is a genetic algorithm for evolving artificial neural networks. Key concepts:

1. **NeuroEvolution**: Combining neural networks with evolutionary algorithms
2. **Genetic Encoding**: Genomes represent neural network structures
3. **Speciation**: Protecting innovation through species formation
4. **Complexification**: Starting with simple networks that grow in complexity

In this project, NEAT evolves car controllers that learn to navigate tracks without human intervention.

## Physics and Mathematics

### Car Movement
```python
# Velocity vector calculation
self.vel_vector = pygame.math.Vector2(0.8, 0)
self.rect.center += self.vel_vector * 2.5

# Rotation handling
self.vel_vector.rotate_ip(self.rotation_vel)
```



# NEAT Self-Driving Car Simulation

A self-driving car simulation using NEAT (NeuroEvolution of Augmenting Topologies) and PyGame. Cars learn to drive on custom tracks using radar sensors and an evolving neural network.

---

## 🔧 Key Equations

- **Velocity**:  
  `v = Δs / Δt`

- **Rotation**:  
  `new_angle = current_angle ± rotation_velocity`

- **Position Update**:  
  `x = x₀ + v·cos(θ)·Δt`  
  `y = y₀ - v·sin(θ)·Δt`

---

## 📡 Sensor System

Cars use **4 radar sensors** at angles **-60°, -30°, 0°, 30°**:

```python
for radar_angle in (-60, -30, 0, 30):
    self.radar(radar_angle)
````

**Distance calculation:**

```python
dist = math.sqrt((car_x - point_x)**2 + (car_y - point_y)**2)
```

---

## 🧠 Neural Network Architecture

* **Inputs**: 4 radar distances + current speed (5 total)
* **Outputs**: Turn left, Turn right
* **Activation Function**: Tanh
* **Topology**: Evolved using NEAT algorithm

---

## 💥 Collision Detection

Using **point-in-color** detection:

```python
if track_surface.get_at(collision_point) == border_color:
    self.alive = False
```

---

## 🧮 Fitness Function

The evolutionary driver:

```python
# Reward for speed
genome.fitness += car.speed * 0.1

# Penalty for crashing
genome.fitness -= 10
```

* Rewarded for maintaining higher speeds
* Penalized for collisions
* Distance traveled contributes to fitness

---

## 📦 Installation

1. **Clone repository:**

```bash
https://github.com/marasinibishesh/self_driving_car_NEAT.git
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

**Run the simulation:**

```bash
python neat_car_simulation.py
```

**Controls:**

* Select track using the buttons
* Click **"Train Model"** to start training
* Click **"Run Best Model"** to view best car
* Press **ESC** to return to menu

---

## 📁 File Structure

```
├── tracks/                # Track images
├── main5.py # Main simulation code
├── config.txt             # NEAT configuration
├── best_genome_track*.pkl # Saved models * astrik for 1,2,3,4 as their are 4 tracks
├── backups/               # Automatic model backups
└── requirements.txt       # Dependencies
```

---

## ⚙️ Configuration

The `config.txt` file includes:

* Population size
* Fitness threshold
* Activation functions
* Crossover and mutation rates
* Species parameters

---

## 🏋️ Training Process

1. Create an initial population of random neural networks
2. Evaluate each car's performance
3. Select top-performing genomes
4. Apply crossover and mutation
5. Repeat for multiple generations

---

## 📊 Results Interpretation

* **Fitness**: Higher values indicate better performance
* **Species Count**: Diversity of neural network structures
* **Distance**: Distance traveled before crashing
* **Speed**: Average speed maintained

---

## 🎨 Customization

* **Add new tracks**:
  Add PNG images to `tracks/` and update `TRACKS` dictionary

* **Modify physics**:
  Adjust `rotation_vel` or velocity multipliers in `Car` class

* **Tune evolution**:
  Edit `config.txt` parameters

---

## 🛠️ Troubleshooting

* **Missing tracks**: Add images to `tracks/` folder
* **Slow performance**: Reduce population size in config
* **No learning**: Adjust fitness rewards and penalties

---

## 🌟 Future Enhancements

* Complex track designs
* Traffic and moving obstacles
* Time trials and races
* Weather effects, day/night cycles
* Hybrid reinforcement learning approach

---

## 🙏 Acknowledgments

* NEAT Algorithm by **Kenneth O. Stanley**
* **PyGame** development team
* **NEAT-Python** library contributors


