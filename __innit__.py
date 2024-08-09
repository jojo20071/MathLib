def add(a,b):
    return a+b

def lerp(a,b,t):
    return a*(1-t)+b*(t) 


pygame.init()

width, height = 800, 600
window = pygame.display.set_mode((width, height))
pygame.display.set_caption('Moving Point')

# Define the initial position of the point

point_color = (255, 0, 0)  # Red color
point_radius = 5

# Set up the clock for controlling the frame rate
clock = pygame.time.Clock()
t1 = 0

point_pos1 = [100,100]
point_pos2 = [300,300]
point_pos3 = [500,100]
# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update the position of the point (example: move right by 1 pixel per frame)
    if t1 >= 100:
        t1 = 0
    t1 += 1
    point_pos4=lerp(np.array(point_pos1),np.array(point_pos2),(t1/100)).tolist()
    point_pos5=lerp(np.array(point_pos2),np.array(point_pos3),(t1/100)).tolist()

    t2 = t1/100

    point_pos6=(((1-t2)**2)*np.array(point_pos1))+(2*(1-t2)*t2*np.array(point_pos2))+((t2**2)*np.array(point_pos3))
    point_pos7=(np.array(2*(1-t2)*(np.array(point_pos1)-np.array(point_pos2))+2*t2*(np.array(point_pos3)-np.array(point_pos2)))*0.1)+np.array(point_pos6)
    print(np.linalg.norm((np.array(2*(1-t2)*(np.array(point_pos1)-np.array(point_pos2))+2*t2*(np.array(point_pos3)-np.array(point_pos2)))*0.1)))
    

    

    point_pos3 = pygame.mouse.get_pos()


    

    # Clear the screen
    window.fill((0, 0, 0))  # Fill the screen with black

    # Draw the point
    pygame.draw.circle(window, point_color, point_pos1, point_radius)
    pygame.draw.circle(window, point_color, point_pos2, point_radius)
    pygame.draw.circle(window, point_color, point_pos3, point_radius)

    pygame.draw.circle(window, point_color, point_pos4, point_radius)
    pygame.draw.circle(window, point_color, point_pos5, point_radius)

    pygame.draw.circle(window, (0, 255, 0), point_pos6, point_radius)
    pygame.draw.circle(window, (0, 0, 255), point_pos7, point_radius)
    pygame.draw.aaline(window, (0, 0, 255), point_pos6,point_pos7, 1)



    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(60)

# Quit Pygame
pygame.quit()
sys.exit()

