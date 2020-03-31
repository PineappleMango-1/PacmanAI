#to-do:
#Add ghost locations to tiles
#Create input array for NN (tiles, directions of pacman + ghosts)

from random import choice
import turtle
from freegames import floor, vector
import numpy
from numpy import random

class PacmanGame:
    def __init__(self):
        self.prev_aim = vector(5,0)
        self.pac = turtle.Turtle(visible=False)
        self.ghost = turtle.Turtle(visible=False)
        self.blinky = [vector(-180, 160), vector(5, 0), "red"]
        self.pinky =  [vector(100, 160), vector(0, -5), "pink"]
        self.inky = [vector(100, -160), vector(-5, 0), "cyan"]
        self.clyde = [vector(-180, -160), vector(0, 5), "orange"]
        self.ghosts = [self.blinky, self.pinky, self.inky, self.clyde]
        self.window = turtle.Screen()
        self.state = {'score': 0}
        self.path = turtle.Turtle(visible=False)
        self.writer = turtle.Turtle(visible=False)
        self.aim = vector(5, 0)
        self.pacman = [vector(-40, -80), vector(0,0)]
        self.chase = False
        self.tunnel = [160, 161, 162, 163, 173, 174, 175, 176]
        self.tiles = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
            0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
            0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0,
            0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
            0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
            0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0,
            0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]
        self.prev_index = self.offset(self.pacman[0])
        self.attempt = 0




    def square(self, x, y):
        "Draw square using path at (x, y)."
        self.path.up()
        self.path.goto(x, y)
        self.path.down()
        self.path.begin_fill()

        for count in range(4):
            self.path.forward(20)
            self.path.left(90)

        self.path.end_fill()

    def offset(self, point):
        "Return offset of point in tiles."
        x = (floor(point.x, 20) + 200) / 20
        y = (180 - floor(point.y, 20)) / 20
        index = int(x + y * 20)
        return index

    def valid(self, point, isPac):
        "Return True if point is valid in tiles."
        index = self.offset(point)
        #if index > len(tiles) - 1:
            #return False
        if self.tiles[index] == 0:
            return False
        if index in self.tunnel and isPac == False:
            return False

        index = self.offset(point + 19)

        if self.tiles[index] == 0:
            return False
        if index in self.tunnel and isPac == False:
            return False
            
        return point.x % 20 == 0 or point.y % 20 == 0

    def world(self):
        "Draw world using path."
        self.window.bgcolor('blue')
        self.path.color('black')
        for index in range(len(self.tiles)):
            tile = self.tiles[index]

            if tile > 0:
                x = (index % 20) * 20 - 200
                y = 180 - (index // 20) * 20
                self.square(x, y)

                if tile == 1:
                    self.path.up()
                    self.path.goto(x + 10, y + 10)
                    self.path.dot(2, 'white')

    def movePacman(self):
        # global prev_aim
        # global aim
        # global pacman
        # global attempt
        if self.offset(self.pacman[0]) == 160:
            self.pacman[0] = vector(115,20)
        if self.offset(self.pacman[0]) == 176:
            self.pacman[0] = vector(-185 , 20)
        if self.attempt == 10:
            self.aim = self.prev_aim.copy()
            self.attempt = 0
        if self.valid(self.pacman[0] + self.aim, True):
            self.pacman[0].move(self.aim)
            self.prev_aim = self.aim.copy()
        else:
            if self.valid(self.pacman[0] + self.prev_aim, True):
                self.pacman[0].move(self.prev_aim)
                self.attempt += 1

        
    def moveBlinky(self, blinky):
        loc = self.blinky[0]
        course = self.blinky[1]
        col = self.blinky[2]
        plan = vector(0,0)
        toPac = self.findPac(self.pacman[0], loc)
        if self.valid(loc + toPac, False) and (toPac != -course):
            course.x = toPac.x
            course.y = toPac.y
        if self.valid(loc + course, False):
            loc.move(course)
        else:
            options = [
                vector(5, 0),
                vector(-5, 0),
                vector(0, 5),
                vector(0, -5),
                ]
            plan = choice(options)
            while (self.valid(loc + plan, False) == False)  or (plan == -course):
                options = [
                vector(5, 0),
                vector(-5, 0),
                vector(0, 5),
                vector(0, -5),
                ]
                plan = choice(options)
            course.x = plan.x
            course.y = plan.y
            loc.move(course)
        self.ghost.up()
        self.ghost.goto(loc.x + 10, loc.y + 10)
        self.ghost.dot(20, col)

    def moveClyde(self, clyde):
        loc = self.clyde[0]
        course = self.clyde[1]
        col = self.clyde[2]
        # global chase
        if abs(self.pacman[0] - loc) > 500:
            self.chase = True
            return
        if abs(self.pacman[0] - loc) < 100:
            self.chase = False
        if self.chase == True:
            self.moveBlinky(clyde)
            return
        else:
            plan = vector(0,0)
            goal = -self.findPac(self.pacman[0], loc)
            if self.valid(loc + goal, False) and (goal != -course):
                course.x = goal.x
                course.y = goal.y
            if self.valid(loc + course, False):
                loc.move(course)
            else:
                options = [
                    vector(5, 0),
                    vector(-5, 0),
                    vector(0, 5),
                    vector(0, -5),
                    ]
                plan = choice(options)
                while (self.valid(loc + plan, False) == False) or (plan == -course):
                    options = [
                    vector(5, 0),
                    vector(-5, 0),
                    vector(0, 5),
                    vector(0, -5)
                    ]
                    plan = choice(options)
                course.x = plan.x
                course.y = plan.y
                loc.move(course)
            self.ghost.up()
            self.ghost.goto(loc.x + 10, loc.y + 10)
            self.ghost.dot(20, col)

    def movePinky(self, pinky, pacAim):
        loc = self.pinky[0]
        course = self.pinky[1]
        col = self.pinky[2]
        goal = self.findPac(self.pacman[0] + 10*pacAim, loc)
        if self.valid(loc + goal, False) and (goal != -course):
            course.x = goal.x
            course.y = goal.y
        if self.valid(loc + course, False):
            loc.move(course)
        else:
            options = [
                vector(5, 0),
                vector(-5, 0),
                vector(0, 5),
                vector(0, -5),
                ]
            plan = choice(options)
            while (self.valid(loc + plan, False) == False)  or (plan == -course):
                options = [
                vector(5, 0),
                vector(-5, 0),
                vector(0, 5),
                vector(0, -5),
                ]
                plan = choice(options)
            course.x = plan.x
            course.y = plan.y
            loc.move(course)
        self.ghost.up()
        self.ghost.goto(loc.x + 10, loc.y + 10)
        self.ghost.dot(20, col)
        return

    def moveInky(self, inky, pacAim):
        loc = inky[0]
        course = inky[1]
        col = inky[2]
        goalTile = self.blinky[0] + ((self.pacman[0] + 2*pacAim) - self.blinky[0]) * 2
        goal = self.findPac(goalTile, loc)
        if self.valid(loc + goal, False) and (goal != -course):
            course.x = goal.x
            course.y = goal.y
        if self.valid(loc + course, False):
            loc.move(course)
        else:
            options = [
                vector(5, 0),
                vector(-5, 0),
                vector(0, 5),
                vector(0, -5),
                ]
            plan = choice(options)
            while (self.valid(loc + plan, False) == False)  or (plan == -course):
                options = [
                vector(5, 0),
                vector(-5, 0),
                vector(0, 5),
                vector(0, -5),
                ]
                plan = choice(options)
            course.x = plan.x
            course.y = plan.y
            loc.move(course)
        self.ghost.up()
        self.ghost.goto(loc.x + 10, loc.y + 10)
        self.ghost.dot(20, col)
        return

    def findPac(self, pacLoc, loc):
        direction = pacLoc - loc
        if abs(direction.x) > abs(direction.y):
            if direction.x < 0:
                plan = vector(-5, 0)
            else:
                plan = vector(5 , 0)
        else:
            if direction.y < 0:
                plan = vector(0, -5)
            else: 
                plan = vector(0,5)
        return(plan)

    def right(self, prev_aim):
        self.aim.x = self.prev_aim.y
        self.aim.y = -self.prev_aim.x

    def left(self, prev_aim):
        self.aim.x = -self.prev_aim.y
        self.aim.y = self.prev_aim.x

    def turn_around(self, prev_aim):
        self.aim.x = self.prev_aim.x
        self.aim.y = -self.prev_aim.y

    def get_input(self, NN_output):
        #to be used with the NN
        # global prev_aim
        if NN_output[0] == 1:
            self.left(self.prev_aim)
        elif NN_output[1] == 1:
            self.right(self.prev_aim)
        elif NN_output[2] == 1:
            self.turn_around(self.prev_aim)

    def get_output(self):
        #dummy function to replicate NN output
        output = [0,0,0,0]
        choice = numpy.random.randint(4)
        output[choice] = 1
        return output

    # self.window.listen()
    # self.window.onkey(lambda: self.run(), 'Right')
    def update(self):
        #output = get_output()
        #this simulates the NN output
        #get_input(output)
        #this uses the NN output to control Pacman
        self.writer.undo()
        self.writer.write(self.state['score'])
        self.pac.clear()
        self.ghost.clear()
        self.tiles[self.prev_index] = 2
        self.prev_index = self.offset(self.pacman[0])
        self.movePacman()
        self.movePinky(self.pinky, self.prev_aim)
        self.moveInky(self.inky, self.prev_aim)
        self.moveBlinky(self.blinky)
        self.moveClyde(self.clyde)
        index = self.offset(self.pacman[0])

        if self.tiles[index] == 1:
            self.tiles[index] = 2
            self.state['score'] += 1
            x = (index % 20) * 20 - 200
            y = 180 - (index // 20) * 20
            self.square(x, y)
        self.tiles[index] = 3
        self.pac.up()
        self.pac.goto(self.pacman[0].x + 10, self.pacman[0].y + 10)
        self.pac.dot(20, 'yellow')
        
        for point, course, col in self.ghosts:
            if abs(self.pacman[0] - point) < 10:
                print("you died")
                return   
        self.window.ontimer(self.update, 50)

    def run(self):
        # global attempt

        self.attempt = 0
        self.window.setup(420, 420, 370, 0)
        self.pac.hideturtle()
        self.ghost.hideturtle()
        self.writer.up()
        self.window.tracer(False)
        self.writer.goto(160, 160)
        self.writer.color('white')
        self.writer.write(self.state['score'])
        self.window.onkey(lambda: self.right(self.prev_aim), 'Right')
        self.window.onkey(lambda: self.left(self.prev_aim), 'Left')
        #window.onkey(lambda: change(0, 5), 'Up')
        self.window.onkey(lambda: self.turn_around(self.prev_aim), 'Down')
        self.world()
        self.update()
        turtle.done()

        

        
    # self.turtle.done()

game = PacmanGame()
game.run()
