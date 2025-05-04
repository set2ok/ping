import random


class Paddle():

    def __init__(self, width, height, bound,speed,type = "bot"):
        self.speed = speed
        self.width = width
        self.height = height
        self.type = type # bot or player
        self.bound = bound # [(x,y) * 4] corners rectangle, 0 top right, 1 top left, 2 bottom right,
                            # 3 bottom left
        self.spawn()# creates cordinates

    #create the outer corners of the paddle
    def creat_figure(self):
        figure = [] # corners
        side = self.side_on_rectangle()
        if side == False:
            side = 0
        n_side = (side + 1) % 4 # next side/ point,

        a = 1 # reverse actions for other pair of side, ex 0,2 and 1,3 defult 0 and 1
        if side >1:
            a = -1

        if side == 0 or side == 2: # upp and down

            if self.is_point_on_line(self.x + a*self.width, self.y, self.bound[side][0],self.bound[side][1], self.bound[n_side][0] + a*self.height/2,self.bound[n_side][1]): # check if curve
                figure.append((self.x, self.y - a * self.height / 2))  # 1 corner
                figure.append((self.x + a * self.width, self.y - a * self.height / 2))
                figure.append((self.x + a*self.width, self.y + a*self.height / 2))

            else: # if curved
                if abs(self.x - self.bound[n_side][0]) <= self.height/2: #smothe transiton
                    figure.append((self.bound[n_side][0] - a* self.height/2,self.y - a*self.height/2 + a*(self.height/2 - abs(self.x - self.bound[n_side][0])))) # first corner
                    figure.append((self.bound[n_side][0] + a * self.height / 2, self.y - a*self.height/2 + a*(self.height/2 - abs(self.x - self.bound[n_side][0]))))  # outer corner
                else:
                    figure.append((self.x, self.y - a * self.height / 2))  # 1 corner
                    figure.append((self.bound[n_side][0] + a*self.height/2 , self.y - a*self.height/2)) # outer corner
                figure.append((self.bound[n_side][0] + a*self.height/2 , self.bound[n_side][1] + a*(self.width - abs(self.bound[n_side][0] - self.x) )))
                figure.append((self.bound[n_side][0] - a*self.height/2 , self.bound[n_side][1] + a*(self.width- abs(self.bound[n_side][0] - self.x) )))
                if not abs(self.x - self.bound[n_side][0]) <= self.height / 2:
                    figure.append((self.bound[n_side][0] - a*self.height/2 , self.bound[n_side][1] + a*self.height/2)) # inner corner

            if not abs(self.x - self.bound[n_side][0]) <= self.height / 2:
                figure.append((self.x, self.y + a*self.height / 2,)) #last corner

        elif side == 1 or side == 3: # left and right

            if self.is_point_on_line(self.x, self.y + a*self.width, self.bound[side][0],self.bound[side][1], self.bound[n_side][0],self.bound[n_side][1] + a*self.height/2): # check if curve
                figure.append((self.x + a * self.height / 2, self.y))  # 1 corner
                figure.append((self.x + a * self.height / 2, self.y + a * self.width))
                figure.append((self.x - a * self.height / 2, self.y + a * self.width))

            else: # if curved
                if abs(self.y - self.bound[n_side][0]) <= self.height/2:
                    figure.append((self.x + a * self.height / 2 - a * (self.height / 2 - abs(self.y - self.bound[n_side][1])), self.bound[n_side][1] - a* self.height/2))
                    figure.append((self.x + a * self.height / 2 - a * (self.height / 2 - abs(self.y - self.bound[n_side][1])), self.bound[n_side][1] + a* self.height/2))  # outer corner

                else:
                    figure.append((self.x + a * self.height / 2, self.y))  # 1 corner
                    figure.append((self.bound[n_side][0] + a*self.height/2 , self.bound[n_side][1] + a*self.height/2)) # outer corner
                figure.append((self.bound[n_side][0] - a*(self.width - abs(self.bound[n_side][0] - self.y)), self.bound[n_side][1] + a*self.height/2))
                figure.append((self.bound[n_side][0] - a*(self.width - abs(self.bound[n_side][0] - self.y)) , self.bound[n_side][1] - a*self.height/2))
                if not abs(self.y - self.bound[n_side][0]) <= self.height / 2:
                    figure.append((self.bound[n_side][0] - a*self.height/2 , self.bound[n_side][1] - a*self.height/2)) # inner corner

            if not abs(self.y - self.bound[n_side][0]) <= self.height/2:
                figure.append((self.x - a*self.height / 2, self.y)) # last corner

        return figure

    # find witch side the point is on
    def side_on_rectangle(self):
        for i in range(4):
            x1, y1 = self.bound[i]
            x2, y2 = self.bound[(i + 1) % 4]  # Wrap around to form edges
            if self.is_point_on_line(self.x, self.y, x1, y1, x2, y2):
                return i
        return False

    # check if a point is on a line segment
    def is_point_on_line(self,x, y, x1, y1, x2, y2):
        # Check collinearity
        if (y - y1) * (x2 - x1) != (x - x1) * (y2 - y1):
            return False
        # Check if point is within segment bounds
        return min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)

    def clamp_to_edge(self):
        side = self.side_on_rectangle()
        if side is not False:
            x1, y1 = self.bound[side]
            x2, y2 = self.bound[(side + 1) % 4]
            # linjeparameter t för projicering
            dx, dy = x2 - x1, y2 - y1
            t = max(0, min(1, ((self.x - x1) * dx + (self.y - y1) * dy) / (dx * dx + dy * dy)))
            self.x = x1 + t * dx
            self.y = y1 + t * dy

    def move(self,direction,dt):
        side = self.side_on_rectangle()
        n_side =  (side + 1) % 4 # next side/ point,
        a = 1
        if side >= 2:
            a = -1
        dist = direction * dt * self.speed * a
        if side == 0 or side == 2:
            if not self.is_point_on_line(self.x + dist ,self.y,self.bound[side][0],self.bound[side][1],self.bound[n_side][0],self.bound[n_side][1]): # check if outside of bounds
                if abs(self.bound[side][0]- (self.x + dist)) < abs(self.bound[n_side][0]- (self.x + dist)): # check closest point
                    self.y = self.bound[side][1] + a*abs(self.bound[side][1]- (self.y + dist))
                    self.x = self.bound[side][0]

                else:
                    self.y = self.bound[n_side][1] + a*abs(self.bound[n_side][1]- (self.y + dist))
                    self.x = self.bound[n_side][0]
            else:
                self.x += dist

        if side == 1 or side == 3:
            if not self.is_point_on_line(self.x,self.y + dist ,self.bound[side][0],self.bound[side][1],self.bound[n_side][0],self.bound[n_side][1]): # check if outside of bounds
                if abs(self.bound[side][1] - (self.y + dist)) < abs(self.bound[n_side][1] - (self.y + dist)): # check closest point
                    self.y = self.bound[side][1]
                    self.x = self.bound[side][0] - a*abs(self.bound[side][0] - (self.x + dist))
                else:
                    self.y = self.bound[n_side][1]
                    self.x = self.bound[n_side][0] - a*abs(self.bound[n_side][0] - (self.x + dist))
            else:
                self.y += dist
        self.clamp_to_edge()
        self.figure = self.creat_figure()

    def spawn(self):
        side = random.randint(0,3)
        n_side = (side + 1) % 4
        if side == 0 or side == 2:
            self.y = self.bound[side][1]
            self.x = random.uniform(min(self.bound[side][0],self.bound[n_side][0]) + self.height ,
                                    max(self.bound[side][0],self.bound[n_side][0]) - self.height)
        else:
            self.x = self.bound[side][0]
            self.y = random.uniform(min(self.bound[side][1], self.bound[n_side][1]) + self.height,
                                    max(self.bound[side][1], self.bound[n_side][1]) - self.height)
        self.figure = self.creat_figure()