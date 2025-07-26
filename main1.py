import neat
import pygame
import os
import math
import sys
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
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

# Load assets
def load_track(name):
    try:
        track = pygame.image.load(os.path.join("tracks", f"{name}.png"))
        overlay = pygame.image.load(os.path.join("tracks", f"{name}-overlay.png"))
        return track, overlay
    except:
        print(f"Track {name} not found! Using default track.")
        track = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        track.fill((50, 50, 50))
        pygame.draw.rect(track, (200, 200, 200), (100, 100, SCREEN_WIDTH-200, SCREEN_HEIGHT-200), 2)
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        return track, overlay

TRACK, TRACK_OVERLAY = load_track("track1")

class Car(pygame.sprite.Sprite):
    def __init__(self, pos=(480, 270)):
        super().__init__()
        try:
            self.original_image = pygame.image.load(os.path.join("tracks", "car4.png"))
        except:
            self.original_image = pygame.Surface((30, 15), pygame.SRCALPHA)
            pygame.draw.rect(self.original_image, (0, 150, 255), (0, 0, 30, 15))
            pygame.draw.polygon(self.original_image, (200, 60, 60), [(30, 0), (40, 7), (30, 15)])
        
        self.image = self.original_image
        self.rect = self.image.get_rect(center=pos)
        self.vel_vector = pygame.math.Vector2(0.8, 0)
        self.angle = 0
        self.rotation_vel = 5
        self.direction = 0
        self.alive = True
        self.radars = []
        self.distance = 0
        self.speed = 0
        self.dead_penalized = False  # Track if penalty has been applied

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

        # Die on collision with track border (white)
        if (0 <= collision_point_right[0] < SCREEN_WIDTH and 0 <= collision_point_right[1] < SCREEN_HEIGHT and
            TRACK.get_at(collision_point_right) == pygame.Color(255, 255, 255)) or \
           (0 <= collision_point_left[0] < SCREEN_WIDTH and 0 <= collision_point_left[1] < SCREEN_HEIGHT and
            TRACK.get_at(collision_point_left) == pygame.Color(255, 255, 255)):
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

        while length < 200:
            x = int(self.rect.center[0] + math.cos(math.radians(self.angle + radar_angle)) * length)
            y = int(self.rect.center[1] - math.sin(math.radians(self.angle + radar_angle)) * length)

            if not (0 <= x < SCREEN_WIDTH and 0 <= y < SCREEN_HEIGHT):
                break

            if TRACK.get_at((x, y)) == pygame.Color(255, 255, 255):
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
            f"Alive: {'Yes' if self.alive else 'No'}",
            f"Speed: {self.speed:.1f}",
            f"Distance: {self.distance:.0f}",
            f"Radars: {data[:4]}",
            f"Speed Input: {data[4]:.1f}"
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
        
    def initialize_population(self):
        self.population = neat.Population(self.config)
        self.population.add_reporter(neat.StdOutReporter(True))
        self.population.add_reporter(self.stats)
        self.population.add_reporter(neat.Checkpointer(5, filename_prefix='neat-checkpoint-'))
        
    def load_best_model(self, filename='best_genome.pkl'):
        try:
            with open(filename, 'rb') as f:
                self.best_genome = pickle.load(f)
            return True
        except:
            return False
            
    def save_best_model(self, genome, filename='best_genome.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(genome, f)
            
    def run_generation(self):
        if not self.population:
            self.initialize_population()
            
        self.population.run(self.eval_genomes, 1)
        self.generation += 1
        
        # Update best genome
        current_best = self.stats.best_genome()
        if current_best and (not self.best_genome or current_best.fitness > self.best_genome.fitness):
            self.best_genome = current_best
            self.save_best_model(self.best_genome)
            self.max_fitness = current_best.fitness
            
        # Update history
        self.fitness_history.append(self.stats.get_fitness_mean()[-1])
        self.species_history.append(len(self.stats.get_species_sizes()[-1]))
        
        # Periodic saving
        current_time = datetime.now()
        if current_time - self.last_save_time >= self.save_interval:
            if self.best_genome:
                timestamp = current_time.strftime("%Y%m%d_%H%M%S")
                filename = f'backups/best_genome_{timestamp}.pkl'
                self.save_best_model(self.best_genome, filename)
                print(f"Periodic save: saved best genome to {filename}")
            self.last_save_time = current_time
        
    def eval_genomes(self, genomes, config):
        cars = []
        ge = []
        nets = []

        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            cars.append(pygame.sprite.GroupSingle(Car()))
            ge.append(genome)
            nets.append(net)
            genome.fitness = 0

        running = True
        while running and any(car.sprite.alive for car in cars):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            SCREEN.blit(TRACK, (0, 0))

            # Update and draw cars
            for i, car in enumerate(cars):
                # Update car first to ensure radar data is current
                car.update()
                
                if car.sprite.alive:
                    # Get neural network output using current data
                    output = nets[i].activate(car.sprite.get_data())
                    
                    # Control the car
                    car.sprite.direction = 0
                    if output[0] > 0.7:
                        car.sprite.direction = 1
                    if output[1] > 0.7:
                        car.sprite.direction = -1
                        
                    # Update fitness
                    ge[i].fitness += car.sprite.speed * 0.1
                else:
                    # Apply death penalty only once
                    if not car.sprite.dead_penalized:
                        ge[i].fitness -= 50
                        car.sprite.dead_penalized = True
                    
                car.draw(SCREEN)

            SCREEN.blit(TRACK_OVERLAY, (0, 0))
            self.draw_stats()
            pygame.display.flip()
            clock.tick(60)
            
    def run_best_model(self):
        if not self.best_genome:
            if not self.load_best_model():
                print("No best model available!")
                return
                
        net = neat.nn.FeedForwardNetwork.create(self.best_genome, self.config)
        car = pygame.sprite.GroupSingle(Car())
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            SCREEN.blit(TRACK, (0, 0))
            
            # Update car first to ensure data is current
            car.update()
            
            if car.sprite.alive:
                output = net.activate(car.sprite.get_data())
                car.sprite.direction = 0
                if output[0] > 0.7:
                    car.sprite.direction = 1
                if output[1] > 0.7:
                    car.sprite.direction = -1
                    
            car.draw(SCREEN)
            car.sprite.draw_debug(SCREEN)
            
            SCREEN.blit(TRACK_OVERLAY, (0, 0))
            self.draw_stats()
            
            # Draw controls info
            font = pygame.font.SysFont(UI_FONT, 20)
            info = [
                "BEST MODEL DEMO",
                "Press ESC to return to menu",
                f"Speed: {car.sprite.speed:.1f}",
                f"Distance: {car.sprite.distance:.0f}"
            ]
            
            y_offset = SCREEN_HEIGHT - 100
            for line in info:
                text_surf = font.render(line, True, UI_COLOR)
                SCREEN.blit(text_surf, (SCREEN_WIDTH - text_surf.get_width() - 10, y_offset))
                y_offset += 25
            
            pygame.display.flip()
            clock.tick(60)
            
    def draw_stats(self):
        font = pygame.font.SysFont(UI_FONT, 20)
        
        stats = [
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
            
    def plot_training_progress(self):
        if len(self.fitness_history) < 2:
            return None
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Fitness plot
        ax1.plot(range(1, len(self.fitness_history)+1), self.fitness_history, 'b-', label='Average Fitness')
        ax1.set_title('Training Progress')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.grid(True)
        ax1.legend()
        
        # Species plot
        ax2.plot(range(1, len(self.species_history)+1), self.species_history, 'g-', label='Species Count')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Species')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        
        # Convert to pygame surface
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        
        surf = pygame.image.fromstring(raw_data, size, "RGB")
        plt.close(fig)
        return surf

def main_menu():
    config_path = os.path.join(os.path.dirname(__file__), 'config.txt')
    neat_manager = NEATManager(config_path)
    
    # Create buttons
    train_button = Button(SCREEN_WIDTH//2 - 100, 150, 200, 50, "Train Model")
    run_button = Button(SCREEN_WIDTH//2 - 100, 220, 200, 50, "Run Best Model")
    plot_button = Button(SCREEN_WIDTH//2 - 100, 290, 200, 50, "Show Training Plot")
    quit_button = Button(SCREEN_WIDTH//2 - 100, 360, 200, 50, "Quit")
    
    buttons = [train_button, run_button, plot_button, quit_button]
    
    # Plot surface
    plot_surface = None
    
    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                plot_surface = None
                
            for button in buttons:
                button.check_hover(mouse_pos)
                if button.handle_event(event):
                    if button == train_button:
                        neat_manager.run_generation()
                    elif button == run_button:
                        neat_manager.run_best_model()
                    elif button == plot_button:
                        plot_surface = neat_manager.plot_training_progress()
                    elif button == quit_button:
                        running = False
        
        # Draw UI
        SCREEN.fill((40, 40, 60))
        
        # Draw title
        title_font = pygame.font.SysFont(UI_FONT, 48)
        title = title_font.render("NEAT Self-Driving Car", True, UI_HIGHLIGHT)
        SCREEN.blit(title, (SCREEN_WIDTH//2 - title.get_width()//2, 50))
        
        # Draw buttons
        for button in buttons:
            button.draw(SCREEN)
            
        # Draw plot if available
        if plot_surface:
            SCREEN.fill((0, 0, 0))
            plot_rect = plot_surface.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
            SCREEN.blit(plot_surface, plot_rect)
            
            # Draw close instruction
            font = pygame.font.SysFont(UI_FONT, 24)
            text = font.render("Press ESC to return", True, UI_COLOR)
            SCREEN.blit(text, (SCREEN_WIDTH//2 - text.get_width()//2, SCREEN_HEIGHT - 50))
        
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main_menu()