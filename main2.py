import neat.config
import pygame
import os
import math
import sys
import neat

SCREEN_WIDTH = 960
SCREEN_HEIGHT = 540

pygame.init()
clock = pygame.time.Clock()  # FPS cap

SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Load base track and overlay
TRACK = pygame.image.load(os.path.join("tracks", "track3.png"))
TRACK_OVERLAY = pygame.image.load(os.path.join("tracks", "track3-overlay.png"))

class Car(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.original_image = pygame.image.load(os.path.join("tracks", "car4.png"))
        self.image = self.original_image
        self.rect = self.image.get_rect(center=(480, 270))  # Visible position
        #self.drive_state = False
        self.vel_vector = pygame.math.Vector2(0.8, 0)
        self.angle = 0
        self.rotation_vel = 5
        self.direction = 0
        self.alive=True
        self.radars=[]

    def update(self):
        self.radars.clear()
        self.drive()
        self.rotate()
        for radar_angle in (-60,-30,0,30,60):
            self.radar(radar_angle)
        self.collision()
        self.data()

    def drive(self):
        #if self.drive_state:
        self.rect.center += self.vel_vector * 2.5  # Reduced speed
    
    def collision(self):
        length=25
        collision_point_right=[int(self.rect.center[0]+math.cos(math.radians(self.angle+18))*length),int(self.rect.center[1]-math.sin(math.radians(self.angle+18))*length)]
        collision_point_left=[int(self.rect.center[0]+math.cos(math.radians(self.angle-18))*length),int(self.rect.center[1]-math.sin(math.radians(self.angle-18))*length)]

        #Die on Collision
        if SCREEN.get_at(collision_point_right)==pygame.Color(255, 255, 255)\
        or SCREEN.get_at(collision_point_left)==pygame.Color(255, 255, 255):
            self.alive=False
            print("Car is dead")
        
        #Draw Collision Points
        pygame.draw.circle(SCREEN,(0,255,255,0),collision_point_right,3)
        pygame.draw.circle(SCREEN,(0,255,255,0),collision_point_left,3)

    def rotate(self):
        if self.direction == 1:
            self.angle -= self.rotation_vel
            self.vel_vector.rotate_ip(self.rotation_vel)

        if self.direction == -1:
            self.angle += self.rotation_vel
            self.vel_vector.rotate_ip(-self.rotation_vel)

        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 1)
        self.rect = self.image.get_rect(center=self.rect.center)

    def radar(self,radar_angle):
        length = 0
        x, y = int(self.rect.center[0]), int(self.rect.center[1])

        # Use TRACK instead of SCREEN for pixel color detection
        while length < 200:
            x = int(self.rect.center[0] + math.cos(math.radians(self.angle+radar_angle)) * length)
            y = int(self.rect.center[1] - math.sin(math.radians(self.angle+radar_angle)) * length)

            if 0 <= x < SCREEN_WIDTH and 0 <= y < SCREEN_HEIGHT:
                if TRACK.get_at((x, y)) != pygame.Color(255, 255, 255):
                    length += 1
                else:
                    break
            else:
                break

        # Draw radar
        pygame.draw.line(SCREEN, (255, 0, 0), self.rect.center, (x, y), 1)
        pygame.draw.circle(SCREEN, (0, 255, 0), (x, y), 3)

        dist=int(math.sqrt(math.pow(self.rect.center[0]-x,2)+math.pow(self.rect.center[1]-y,2)))

        self.radars.append([radar_angle,dist])

    def data(self):
        input=[0,0,0,0,0]
        for i,radar in enumerate(self.radars):
            input[i]=int(radar[1])
        return input

#car = pygame.sprite.GroupSingle(Car())
def remove(index):
    cars.pop(index)
    ge.pop(index)
    nets.pop(index)

def eval_genomes(genomes,config):
    global cars,ge,nets

    cars=[]
    ge=[]
    nets=[]

    for genome_id,genome in genomes:
        cars.append(pygame.sprite.GroupSingle(Car()))
        ge.append(genome)
        net=neat.nn.FeedForwardNetwork.create(genome,config)
        nets.append(net)
        genome.fitness=0


    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        SCREEN.blit(TRACK, (0, 0))  # Base track

        if len(cars)==0:
            break

        for i, car in enumerate(cars):
            ge[i].fitness+=1
            if not car.sprite.alive:
                remove(i)

        for i,car in enumerate(cars):
            output=nets[i].activate(car.sprite.data())
            if output[0]>0.7:
                car.sprite.direction=1
            if output[1]>0.7:
                car.sprite.direction=-1
            if output[0]<=0.7 and output[1]<=0.7:
                car.sprite.direction=0

        # # USER INPUT
        # user_input = pygame.key.get_pressed()
        # if sum(user_input) <= 1:
        #     car.sprite.drive_state = False
        #     car.sprite.direction = 0

        # # DRIVE
        # if user_input[pygame.K_UP]:
        #     car.sprite.drive_state = True

        # # STEERING
        # if user_input[pygame.K_RIGHT]:
        #     car.sprite.direction = 1
        # if user_input[pygame.K_LEFT]:
        #     car.sprite.direction = -1

        # UPDATE
        for car in cars:
         car.draw(SCREEN)
         car.update()

        SCREEN.blit(TRACK_OVERLAY, (0, 0))  # Overlay top layer
        pygame.display.update()

        clock.tick(60)  # Cap FPS to 60

#eval_genomes()


#Setup NEAT Neural Network
def run(config_path):
    global pop 
    config=neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    pop=neat.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    stats=neat.StatisticsReporter()
    pop.add_reporter(stats)


    pop.run(eval_genomes,50)


if __name__=='__main__':
    local_dir=os.path.dirname(__file__)
    config_path=os.path.join(local_dir,'config.txt')
    run(config_path)
