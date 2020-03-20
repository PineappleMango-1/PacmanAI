import pygame

def game():
    #initialize pygame module
    pygame.init()
    clock = pygame.time.Clock()
    '''
    # load and set the logo
    logo = pygame.image.load("logo32x32.png")
    pygame.display.set_icon(logo)
    pygame.display.set_caption("minimal program")
    '''

    #create a surface on the screen
    screen = Screen(400, 300)
    surface = pygame.display.set_mode((screen.width, screen.height))

    paddle = Paddle(screen)
    ball = Ball(screen)

    running = True

    while running:
        msElapsed = clock.tick(30)
        paddle.update(ball)
        ball.collide(screen, paddle)
        ball.update()
        surface.fill((0,0,0))
        pygame.display.update()
        ball.drawCircle(surface)
        paddle.drawRect(surface)

        #event handling
        for event in pygame.event.get():
            #when quit, quit
            if event.type == pygame.QUIT:
                running = False

class Screen:
    def __init__(self, width, height):
        self.width = width
        self.height = height

class Paddle:
    def __init__(self, screen):
        self.height = 15
        self.width = 60
        self.x = screen.width/2- self.width/2
        self.y = screen.height - self.height
        self.speed = 10

    def update(self, ball):
        if self.x < ball.x:
            self.x += self.speed
        elif self.x > ball.x:
            self.x -= self.speed
        

    def drawRect(self, surface):
        pygame.draw.rect(surface, (255, 255, 255), (self.x, self.y, self.width, self.height))
        pygame.display.update()

class Ball:
    def __init__(self, screen):
        self.radius = 10
        self.x = int(screen.width/2- self.radius/2)
        self.y = int(0 + self.radius)
        self.dx = 10
        self.dy = 10

    def drawCircle(self, surface):
        pygame.draw.circle(surface, (255, 0, 0), (self.x,self.y), self.radius)

    def update(self):
        self.x += self.dx
        self.y += self.dy

    def collide(self, screen, paddle):
        if self.x + self.radius == paddle.x + paddle.width and self.y + self.radius == paddle.y + paddle.height:
            self.dy = -self.dy
            self.dx = -self.dy
        elif self.x + self.radius >= screen.width or self.x - self.radius <= 0:
            self.dx = -self.dx
        elif self.y + self.radius >= screen.height or self.y - self.radius <= 0:
            self.dy = -self.dy

game()