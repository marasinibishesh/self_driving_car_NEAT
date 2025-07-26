import neat
import pygame
import os
import math
import sys
import pickle
import numpy as np
from datetime import datetime, timedelta

# Constants
SCREEN_WIDTH = 960
SCREEN_HEIGHT = 540
UI_FONT = "Arial"
UI_FONT_SIZE = 24
UI_COLOR = (220, 220, 220)
UI_HIGHLIGHT = (255, 215, 0)
SAVE_INTERVAL_MINUTES = 5  # Save every 5 minutes

# Initialize pygame
pygame.init()
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("NEAT Self-Driving Car")
clock = pygame.time.Clock()

# Create backups directory if needed
if not os.path.exists('backups'):
    os.makedirs('backups')
    
# Track definitions
TRACKS = {
    "track1": {
        "name": "Track 1",
        "start_pos": (480, 270),
        "model_prefix": "track1",
        "border_color": (255, 255, 255)  # White border
    },
    "track2": {
        "name": "Track 2",
        "start_pos": (480, 270),
        "model_prefix": "track2",
        "border_color": (255, 255, 255)  # White border
    },
    "track3": {
        "name": "Track 3",
        "start_pos": (480, 270),
        "model_prefix": "track3",
        "border_color": (255, 255, 255)  # White border
    },
    "track4": {
        "name": "Track 4",
        "start_pos": (480, 270),
        "model_prefix": "track4",
        "border_color": (255, 255, 255)  # White border
    }
}

# Load assets
def load_track(name):
    try:
        track = pygame.image.load(os.path.join("tracks", f"{name}.png")).convert()
        overlay = pygame.image.load(os.path.join("tracks", f"{name}-overlay.png")).convert_alpha()
        return track, overlay
    except Exception as e:
        print(f"Track {name} not found! Using default track. Error: {e}")
        # Create a better default track with visible borders
        track = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        track.fill((50, 50, 50))  # Dark gray background
        
        # Draw track with visible borders
        pygame.draw.rect(track, (100, 100, 100), (50, 50, SCREEN_WIDTH-100, SCREEN_HEIGHT-100))
        pygame.draw.rect(track, TRACKS[name]["border_color"], (50, 50, SCREEN_WIDTH-100, SCREEN_HEIGHT-100), 5)
        
        # Draw start/finish line
        pygame.draw.line(track, (0, 255, 0), (100, SCREEN_HEIGHT//2), (200, SCREEN_HEIGHT//2), 5)
        
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        return track, overlay

# Preload all tracks
TRACK_IMAGES = {}
for track_id in TRACKS:
    TRACK_IMAGES[track_id] = load_track(track_id)

class Car(pygame.sprite.Sprite):
    def __init__(self, track_id, pos=None):
        super().__init__()
        track_info = TRACKS[track_id]
        start_pos = pos if pos else track_info["start_pos"]
        
        try:
            self.original_image = pygame.image.load(os.path.join("tracks", "car4.png")).convert_alpha()
        except:
            # Create a simple car graphic
            self.original_image = pygame.Surface((30, 15), pygame.SRCALPHA)
            pygame.draw.rect(self.original_image, (0, 150, 255), (0, 0, 30, 15))
            pygame.draw.polygon(self.original_image, (200, 60, 60), [(30, 0), (40, 7), (30, 15)])
        
        self.image = self.original_image
        self.rect = self.image.get_rect(center=start_pos)
        self.vel_vector = pygame.math.Vector2(0.8, 0)
        self.angle = 0
        self.rotation_vel = 5
        self.direction = 0
        self.alive = True
        self.radars = []
        self.distance = 0
        self.speed = 0
        self.dead_penalized = False  # Track if penalty has been applied
        self.track_id = track_id
        self.track_surface, self.track_overlay = TRACK_IMAGES[track_id]
        self.border_color = TRACKS[track_id]["border_color"]

    def update(self):
        if not self.alive:
            return
            
        self.radars.clear()
        self.drive()
        self.rotate()
        
        # Fixed to exactly 4 radar angles to match 5 inputs (including speed)
        for radar_angle in (-60, -30, 0, 30):
            self.radar(radar_angle)
            
        self.collision()
        self.distance += abs(self.speed)
        self.speed = math.sqrt(self.vel_vector.x**2 + self.vel_vector.y**2) * 2.5

    def drive(self):
        self.rect.center += self.vel_vector * 2.5
    
    def collision(self):
        length = 25
        collision_point_right = [
            int(self.rect.center[0] + math.cos(math.radians(self.angle + 18)) * length),
            int(self.rect.center[1] - math.sin(math.radians(self.angle + 18)) * length)
        ]
        collision_point_left = [
            int(self.rect.center[0] + math.cos(math.radians(self.angle - 18)) * length),
            int(self.rect.center[1] - math.sin(math.radians(self.angle - 18)) * length)
        ]

        # Die on collision with track border
        border_color = self.border_color
        try:
            # Check if collision points are within screen bounds
            if (0 <= collision_point_right[0] < SCREEN_WIDTH and 
                0 <= collision_point_right[1] < SCREEN_HEIGHT and
                self.track_surface.get_at(collision_point_right) == border_color):
                self.alive = False
            
            if (0 <= collision_point_left[0] < SCREEN_WIDTH and 
                0 <= collision_point_left[1] < SCREEN_HEIGHT and
                self.track_surface.get_at(collision_point_left) == border_color):
                self.alive = False
        except IndexError:
            # If point is outside screen, consider it a collision
            self.alive = False

        # Draw collision points
        pygame.draw.circle(SCREEN, (0, 255, 255), collision_point_right, 3)
        pygame.draw.circle(SCREEN, (0, 255, 255), collision_point_left, 3)

    def rotate(self):
        if self.direction == 1:
            self.angle -= self.rotation_vel
            self.vel_vector.rotate_ip(self.rotation_vel)
        elif self.direction == -1:
            self.angle += self.rotation_vel
            self.vel_vector.rotate_ip(-self.rotation_vel)

        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 1)
        self.rect = self.image.get_rect(center=self.rect.center)

    def radar(self, radar_angle):
        length = 0
        x, y = self.rect.center
        border_color = self.border_color

        while length < 200:
            x = int(self.rect.center[0] + math.cos(math.radians(self.angle + radar_angle)) * length)
            y = int(self.rect.center[1] - math.sin(math.radians(self.angle + radar_angle)) * length)

            if not (0 <= x < SCREEN_WIDTH and 0 <= y < SCREEN_HEIGHT):
                break

            try:
                if self.track_surface.get_at((x, y)) == border_color:
                    break
            except IndexError:
                break

            length += 1

        # Draw radar
        pygame.draw.line(SCREEN, (255, 100, 100), self.rect.center, (x, y), 1)
        pygame.draw.circle(SCREEN, (0, 200, 0), (x, y), 3)

        dist = int(math.sqrt((self.rect.center[0] - x) ** 2 + (self.rect.center[1] - y) ** 2))
        self.radars.append([radar_angle, dist])

    def get_data(self):
        # Return exactly 5 inputs: 4 radar distances + speed
        return [radar[1] for radar in self.radars] + [self.speed]
        
    def draw_debug(self, surface):
        font = pygame.font.SysFont(UI_FONT, 14)
        data = self.get_data()
        text = [
            f"Track: {TRACKS[self.track_id]['name']}",
            f"Alive: {'Yes' if self.alive else 'No'}",
            f"Speed: {self.speed:.1f}",
            f"Distance: {self.distance:.0f}",
            f"Radars: {data[:4]}",
            f"Speed Input: {data[4]:.1f}",
            f"Border Color: {self.border_color}"
        ]
        
        y_offset = 10
        for line in text:
            text_surf = font.render(line, True, UI_COLOR)
            surface.blit(text_surf, (10, y_offset))
            y_offset += 20

class Button:
    def __init__(self, x, y, width, height, text, color=(70, 70, 100), hover_color=(100, 100, 150)):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.hover_color = hover_color
        self.text = text
        self.is_hovered = False
        self.font = pygame.font.SysFont(UI_FONT, UI_FONT_SIZE)

    def draw(self, surface):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(surface, color, self.rect, border_radius=5)
        pygame.draw.rect(surface, UI_COLOR, self.rect, 2, border_radius=5)
        
        text_surf = self.font.render(self.text, True, UI_COLOR)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def check_hover(self, pos):
        self.is_hovered = self.rect.collidepoint(pos)
        return self.is_hovered

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.is_hovered:
                return True
        return False

class NEATManager:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = neat.config.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
        self.population = None
        self.best_genome = None
        self.stats = neat.StatisticsReporter()
        self.generation = 0
        self.max_fitness = 0
        self.fitness_history = []
        self.species_history = []
        self.start_time = datetime.now()
        self.last_save_time = datetime.now()
        self.save_interval = timedelta(minutes=SAVE_INTERVAL_MINUTES)
        self.current_track = "track1"
        self.last_generation_report = datetime.now()
        
    def initialize_population(self):
        self.population = neat.Population(self.config)
        self.population.add_reporter(neat.StdOutReporter(True))
        self.population.add_reporter(self.stats)
        self.population.add_reporter(neat.Checkpointer(5, filename_prefix=f'neat-checkpoint-{self.current_track}-'))
        print(f"\n=== Starting training on {TRACKS[self.current_track]['name']} ===")
        print(f"Initial population created with {self.config.pop_size} genomes")
        
    def load_best_model(self):
        filename = f'best_genome_{self.current_track}.pkl'
        try:
            with open(filename, 'rb') as f:
                self.best_genome = pickle.load(f)
            print(f"Loaded best model for {TRACKS[self.current_track]['name']} with fitness: {self.best_genome.fitness:.2f}")
            return True
        except:
            print(f"No existing best model found for {TRACKS[self.current_track]['name']}")
            return False
            
    def save_best_model(self, genome):
        filename = f'best_genome_{self.current_track}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(genome, f)
        print(f"Saved best model for {TRACKS[self.current_track]['name']} with fitness: {genome.fitness:.2f}")
            
    def run_generation(self):
        if not self.population:
            self.initialize_population()
            
        self.population.run(self.eval_genomes, 1)
        self.generation += 1
        
        # Update best genome
        current_best = self.stats.best_genome()
        if current_best and (not self.best_genome or current_best.fitness > self.best_genome.fitness):
            print(f"🚗 New best model found! Fitness: {current_best.fitness:.2f}")
            self.best_genome = current_best
            self.save_best_model(self.best_genome)
            self.max_fitness = current_best.fitness
            
        # Update history
        if self.stats.get_fitness_mean():
            self.fitness_history.append(self.stats.get_fitness_mean()[-1])
        if self.stats.get_species_sizes():
            self.species_history.append(len(self.stats.get_species_sizes()[-1]))
        
        # Periodic saving
        current_time = datetime.now()
        if current_time - self.last_save_time >= self.save_interval:
            if self.best_genome:
                timestamp = current_time.strftime("%Y%m%d_%H%M%S")
                filename = f'backups/best_genome_{self.current_track}_{timestamp}.pkl'
                with open(filename, 'wb') as f:
                    pickle.dump(self.best_genome, f)
                print(f"⏱️ Periodic save: Saved backup to {filename}")
            self.last_save_time = current_time
            
        # Print progress report every 30 seconds
        if (current_time - self.last_generation_report).seconds >= 30:
            alive_count = len([g for g in self.population.population.values() if g.fitness > 0])
            print(f"⏳ Generation {self.generation}: Best fitness = {self.max_fitness:.2f}, Species = {len(self.stats.get_species_sizes()[-1])}, Alive = {alive_count}/{self.config.pop_size}")
            self.last_generation_report = current_time
        
    def eval_genomes(self, genomes, config):
        cars = []
        ge = []
        nets = []
        
        track_surface, track_overlay = TRACK_IMAGES[self.current_track]

        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            car_group = pygame.sprite.GroupSingle(Car(self.current_track))
            cars.append(car_group)
            ge.append(genome)
            nets.append(net)
            genome.fitness = 0

        running = True
        while running and any(car.sprite.alive for car in cars):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            SCREEN.blit(track_surface, (0, 0))

            # Update and draw cars
            for i, car_group in enumerate(cars):
                car = car_group.sprite
                car_group.update()
                
                if car.alive:
                    # Get neural network output using current data
                    output = nets[i].activate(car.get_data())
                    
                    # Control the car
                    car.direction = 0
                    if output[0] > 0.7:
                        car.direction = 1
                    if output[1] > 0.7:
                        car.direction = -1
                        
                    # Update fitness
                    ge[i].fitness += car.speed * 0.1
                    
                    # Draw only alive cars
                    car_group.draw(SCREEN)
                else:
                    # Apply death penalty only once
                    if not car.dead_penalized:
                        ge[i].fitness -= 10  # Reduced penalty
                        car.dead_penalized = True

            SCREEN.blit(track_overlay, (0, 0))
            self.draw_stats()
            pygame.display.flip()
            clock.tick(60)
            
    def run_best_model(self):
        if not self.best_genome:
            if not self.load_best_model():
                print("⚠️ No best model available! Train a model first.")
                return
                
        net = neat.nn.FeedForwardNetwork.create(self.best_genome, self.config)
        car_group = pygame.sprite.GroupSingle(Car(self.current_track))
        car = car_group.sprite
        track_surface, track_overlay = TRACK_IMAGES[self.current_track]
        
        print(f"\n=== Running best model on {TRACKS[self.current_track]['name']} ===")
        print(f"Model fitness: {self.best_genome.fitness:.2f}")
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            SCREEN.blit(track_surface, (0, 0))
            
            # Update car
            car_group.update()
            
            if car.alive:
                output = net.activate(car.get_data())
                car.direction = 0
                if output[0] > 0.7:
                    car.direction = 1
                if output[1] > 0.7:
                    car.direction = -1
                    
                # Draw only alive car
                car_group.draw(SCREEN)
                
            car.draw_debug(SCREEN)
            
            SCREEN.blit(track_overlay, (0, 0))
            self.draw_stats()
            
            # Draw controls info
            font = pygame.font.SysFont(UI_FONT, 20)
            info = [
                "BEST MODEL DEMO",
                f"Track: {TRACKS[self.current_track]['name']}",
                "Press ESC to return to menu",
                f"Speed: {car.speed:.1f}",
                f"Distance: {car.distance:.0f}",
                f"Border Color: {car.border_color}"
            ]
            
            y_offset = SCREEN_HEIGHT - 140
            for line in info:
                text_surf = font.render(line, True, UI_COLOR)
                SCREEN.blit(text_surf, (SCREEN_WIDTH - text_surf.get_width() - 10, y_offset))
                y_offset += 25
            
            pygame.display.flip()
            clock.tick(60)
            
    def draw_stats(self):
        font = pygame.font.SysFont(UI_FONT, 20)
        
        stats = [
            f"Track: {TRACKS[self.current_track]['name']}",
            f"Generation: {self.generation}",
            f"Species: {len(self.stats.get_species_sizes()[-1]) if self.generation > 0 else 0}",
            f"Best Fitness: {self.max_fitness:.1f}",
            f"Time: {(datetime.now() - self.start_time).seconds // 60}m:{(datetime.now() - self.start_time).seconds % 60}s"
        ]
        
        y_offset = 10
        for line in stats:
            text_surf = font.render(line, True, UI_HIGHLIGHT)
            SCREEN.blit(text_surf, (SCREEN_WIDTH - text_surf.get_width() - 10, y_offset))
            y_offset += 30

def main_menu():
    config_path = os.path.join(os.path.dirname(__file__), 'config.txt')
    neat_manager = NEATManager(config_path)
    
    # Create buttons in two columns
    track_buttons = []
    left_x = SCREEN_WIDTH // 2 - 250
    right_x = SCREEN_WIDTH // 2 + 50
    
    # Left column tracks
    track_buttons.append(Button(left_x, 150, 200, 40, TRACKS["track1"]["name"]))
    track_buttons.append(Button(left_x, 200, 200, 40, TRACKS["track2"]["name"]))
    
    # Right column tracks
    track_buttons.append(Button(right_x, 150, 200, 40, TRACKS["track3"]["name"]))
    track_buttons.append(Button(right_x, 200, 200, 40, TRACKS["track4"]["name"]))
    
    # Action buttons centered below
    train_button = Button(SCREEN_WIDTH//2 - 100, 300, 200, 40, "Train Model")
    run_button = Button(SCREEN_WIDTH//2 - 100, 350, 200, 40, "Run Best Model")
    quit_button = Button(SCREEN_WIDTH//2 - 100, 400, 200, 40, "Quit")
    
    buttons = track_buttons + [train_button, run_button, quit_button]
    
    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            for i, button in enumerate(buttons):
                button.check_hover(mouse_pos)
                if button.handle_event(event):
                    if i < len(track_buttons):  # Track selection button
                        track_id = list(TRACKS.keys())[i]
                        neat_manager.current_track = track_id
                        neat_manager.best_genome = None  # Reset best genome for new track
                        print(f"\n🔁 Switched to {TRACKS[track_id]['name']}")
                        neat_manager.load_best_model()  # Try to load best model for this track
                    elif button == train_button:
                        print("\n🏁 Starting training...")
                        neat_manager.run_generation()
                        print("✅ Training completed!")
                    elif button == run_button:
                        print("\n🏎️ Running best model...")
                        neat_manager.run_best_model()
                        print("🏁 Demo completed!")
                    elif button == quit_button:
                        running = False
        
        # Draw UI
        SCREEN.fill((40, 40, 60))
        
        # Draw title
        title_font = pygame.font.SysFont(UI_FONT, 48)
        title = title_font.render("NEAT Self-Driving Car", True, UI_HIGHLIGHT)
        SCREEN.blit(title, (SCREEN_WIDTH//2 - title.get_width()//2, 50))
        
        # Draw section headers
        section_font = pygame.font.SysFont(UI_FONT, 32)
        track_header = section_font.render("Select Track", True, UI_COLOR)
        SCREEN.blit(track_header, (SCREEN_WIDTH//2 - track_header.get_width()//2, 110))
        
        # Draw current track info
        current_font = pygame.font.SysFont(UI_FONT, 28)
        current_track = current_font.render(
            f"Current: {TRACKS[neat_manager.current_track]['name']}", 
            True, 
            UI_HIGHLIGHT
        )
        SCREEN.blit(current_track, (SCREEN_WIDTH//2 - current_track.get_width()//2, 500))
        
        # Draw buttons
        for button in buttons:
            button.draw(SCREEN)
            
            # Highlight current track button
            if button in track_buttons:
                track_id = list(TRACKS.keys())[track_buttons.index(button)]
                if track_id == neat_manager.current_track:
                    pygame.draw.rect(SCREEN, UI_HIGHLIGHT, button.rect, 3, border_radius=5)
            
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    print("=== NEAT Self-Driving Car Simulation ===")
    print("Controls:")
    print("- Select a track using the buttons")
    print("- Click 'Train Model' to start/continue training")
    print("- Click 'Run Best Model' to see the best car in action")
    print("- Models are automatically saved every 5 minutes")
    print("- Press ESC in demo mode to return to menu")
    main_menu()