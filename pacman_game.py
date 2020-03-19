"""Pacman, classic arcade game.
Exercises
1. Change the board.
2. Change the number of ghosts.
3. Change where pacman starts.
4. Make the ghosts faster/slower.
5. Make the ghosts smarter.
"""

from random import choice
import turtle
from freegames import floor, vector
prev_aim = vector(5,0)
pac = turtle.Turtle(visible=False)
ghost = turtle.Turtle(visible=False)
blinky = [vector(-180, 160), vector(5, 0), "red"]
pinky =  [vector(100, 160), vector(0, -5), "pink"]
inky = [vector(100, -160), vector(-5, 0), "cyan"]
clyde = [vector(-180, -160), vector(0, 5), "orange"]
ghosts = [blinky, pinky, inky, clyde]
window = turtle.Screen()
state = {'score': 0}
path = turtle.Turtle(visible=False)
writer = turtle.Turtle(visible=False)
aim = vector(5, 0)
pacman = vector(-40, -80)
chase = False
tiles = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
    0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
    0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    3, 3, 3, 3, 1, 1, 1, 0, 0, 0, 1, 1, 1, 3, 3, 3, 3, 0, 0, 0,
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

def square(x, y):
    "Draw square using path at (x, y)."
    path.up()
    path.goto(x, y)
    path.down()
    path.begin_fill()

    for count in range(4):
        path.forward(20)
        path.left(90)

    path.end_fill()

def offset(point):
    "Return offset of point in tiles."
    x = (floor(point.x, 20) + 200) / 20
    y = (180 - floor(point.y, 20)) / 20
    index = int(x + y * 20)
    return index

def valid(point, isPac):
    "Return True if point is valid in tiles."
    index = offset(point)
    #if index > len(tiles) - 1:
        #return False
    if tiles[index] == 0:
        return False
    if tiles[index] == 3 and isPac == False:
        return False

    index = offset(point + 19)

    if tiles[index] == 0:
        return False
    if tiles[index] == 3 and isPac == False:
        return False
        
    return point.x % 20 == 0 or point.y % 20 == 0

def world():
    "Draw world using path."
    window.bgcolor('blue')
    path.color('black')

    for index in range(len(tiles)):
        tile = tiles[index]

        if tile > 0:
            x = (index % 20) * 20 - 200
            y = 180 - (index // 20) * 20
            square(x, y)

            if tile == 1:
                path.up()
                path.goto(x + 10, y + 10)
                path.dot(2, 'white')

def move():
    "Move pacman and all ghosts."
    writer.undo()
    writer.write(state['score'])
    global prev_aim
    global aim
    global attempt
    global pacman
    pac.clear()
    ghost.clear()
    if offset(pacman) == 160:
        print("tp")
        pacman = vector(115,20)
    if offset(pacman) == 176:
        print("tp")
        pacman = vector(-185 , 20)
    if attempt == 10:
        aim = prev_aim.copy()
        attempt = 0
    if valid(pacman + aim, True):
        pacman.move(aim)
        prev_aim = aim.copy()
    else:
        if valid(pacman + prev_aim, True):
            pacman.move(prev_aim)
            attempt += 1
    movePinky(pinky, prev_aim)
    moveInky(inky, prev_aim)
    moveBlinky(blinky)
    moveClyde(clyde)
    index = offset(pacman)

    if tiles[index] == 1:
        tiles[index] = 2
        state['score'] += 1
        x = (index % 20) * 20 - 200
        y = 180 - (index // 20) * 20
        square(x, y)

    pac.up()
    pac.goto(pacman.x + 10, pacman.y + 10)
    pac.dot(20, 'yellow')
    
    for point, course, col in ghosts:
        if abs(pacman - point) < 10:
            print("you died")
            return

    window.ontimer(move, 50)


def moveBlinky(blinky):
    loc = blinky[0]
    course = blinky[1]
    col = blinky[2]
    plan = vector(0,0)
    toPac = findPac(pacman, loc)
    if valid(loc + toPac, False) and (toPac != -course):
        course.x = toPac.x
        course.y = toPac.y
    if valid(loc + course, False):
        loc.move(course)
    else:
        options = [
            vector(5, 0),
            vector(-5, 0),
            vector(0, 5),
            vector(0, -5),
            ]
        plan = choice(options)
        while (valid(loc + plan, False) == False)  or (plan == -course):
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
    ghost.up()
    ghost.goto(loc.x + 10, loc.y + 10)
    ghost.dot(20, col)

def moveClyde(clyde):
    loc = clyde[0]
    course = clyde[1]
    col = clyde[2]
    global chase
    if abs(pacman - loc) > 500:
        chase = True
        return
    if abs(pacman - loc) < 100:
        chase = False
    if chase == True:
        moveBlinky(clyde)
        return
    else:
        plan = vector(0,0)
        goal = -findPac(pacman, loc)
        if valid(loc + goal, False) and (goal != -course):
            course.x = goal.x
            course.y = goal.y
        if valid(loc + course, False):
            loc.move(course)
        else:
            options = [
                vector(5, 0),
                vector(-5, 0),
                vector(0, 5),
                vector(0, -5),
                ]
            plan = choice(options)
            while (valid(loc + plan, False) == False) or (plan == -course):
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
        ghost.up()
        ghost.goto(loc.x + 10, loc.y + 10)
        ghost.dot(20, col)

def movePinky(pinky, pacAim):
    loc = pinky[0]
    course = pinky[1]
    col = pinky[2]
    goal = findPac(pacman + 10*pacAim, loc)
    if valid(loc + goal, False) and (goal != -course):
        course.x = goal.x
        course.y = goal.y
    if valid(loc + course, False):
        loc.move(course)
    else:
        options = [
            vector(5, 0),
            vector(-5, 0),
            vector(0, 5),
            vector(0, -5),
            ]
        plan = choice(options)
        while (valid(loc + plan, False) == False)  or (plan == -course):
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
    ghost.up()
    ghost.goto(loc.x + 10, loc.y + 10)
    ghost.dot(20, col)
    return

def moveInky(inky, pacAim):
    loc = inky[0]
    course = inky[1]
    col = inky[2]
    goalTile = blinky[0] + ((pacman + 2*pacAim) - blinky[0]) * 2
    goal = findPac(goalTile, loc)
    if valid(loc + goal, False) and (goal != -course):
        course.x = goal.x
        course.y = goal.y
    if valid(loc + course, False):
        loc.move(course)
    else:
        options = [
            vector(5, 0),
            vector(-5, 0),
            vector(0, 5),
            vector(0, -5),
            ]
        plan = choice(options)
        while (valid(loc + plan, False) == False)  or (plan == -course):
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
    ghost.up()
    ghost.goto(loc.x + 10, loc.y + 10)
    ghost.dot(20, col)
    return

def change(x, y):
    #if valid(pacman + vector(x, y)):
    aim.x = x
    aim.y = y

def findPac(pacLoc, loc):
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


window.listen()
window.onkey(lambda: run(), 'Right')
def run():
    global attempt
    attempt = 0
    window.setup(420, 420, 370, 0)
    pac.hideturtle()
    ghost.hideturtle()
    writer.up()
    window.tracer(False)
    writer.goto(160, 160)
    writer.color('white')
    writer.write(state['score'])
    window.onkey(lambda: change(5, 0), 'Right')
    window.onkey(lambda: change(-5, 0), 'Left')
    window.onkey(lambda: change(0, 5), 'Up')
    window.onkey(lambda: change(0, -5), 'Down')
    world()
    move()
turtle.done()