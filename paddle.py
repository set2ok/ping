class Paddle():

    def __init__(self,pos, width, height, state = "hor"):
        self.x,self.y = pos
        self.width = width
        self.height = height
        self.state = state # hor and ver, horizontal, vertical