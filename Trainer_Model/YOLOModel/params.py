class Params:
    def __init__(self, version:str):
        if version == 'n':
            self.depth, self.width, self.ratio = 1/3,1/4,2.0
        elif version == 's':
            self.depth, self.width, self.ratio = 1/3,1/2,2.0
        elif version == 'm':
            self.depth, self.width, self.ratio = 2/3,3/2,1.5
        elif version == 'l':
            self.depth, self.width, self.ratio = 1.0,1.0,1.0
        elif version == 'x':
            self.depth, self.width, self.ratio = 1.0,1.25,1.0

    def return_params(self):
        return self.depth, self.width, self.ratio