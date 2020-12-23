class Rule:
    def __init__(self, p1, p2, side):
        self.p1 = p1
        self.p2 = p2
        if side == "up" or side == "down":
            self.side = side
        if self.p1.x - self.p2.x == 0:
            self.parallel_to_y = True
            self.m = 0
            self.n = 0
        else:
            self.parallel_to_y = False
            self.m = (p1.y - p2.y) / (p1.x - p2.x)
            self.n = p1.y - (self.m * p1.x)  # n = y - mx

    def classify(self, p):
        if self.side == "up":
            return self.classify_first_direction(p)
        elif self.side == "down":
            return self.classify_second_direction(p)

    def classify_is_correct(self, p):
        if self.side == "up":
            return self.first_classify_is_correct(p)
        elif self.side == "down":
            return self.second_classify_is_correct(p)

    def classify_first_direction(self, p):
        # if the rule is an ordinary line
        #
        if not self.parallel_to_y:
            y = p.y
            mxn = (self.m * p.x) + self.n
            if y >= mxn:
                return -1
            else:
                return 1

        # if the rule is a line parallel to the Y axis
        else:
            if p.x >= self.p1.x:
                return 1
            else:
                return -1

    def classify_second_direction(self, p):
        return self.classify_first_direction(p) * -1

    def first_classify_is_correct(self, p):
        if self.classify_first_direction(p) == p.type:
            return True
        else:
            return False

    def second_classify_is_correct(self, p):
        if self.classify_second_direction(p) == p.type:
            return True
        else:
            return False

    def print_r(self):
        if self.p1.x - self.p2.x == 0:
            print("x =", self.p1.x, "side:", self.side, "(p1:", self.p1.to_string(), ", p2:", self.p2.to_string())
        else:
            print("y =", self.m, "x +", self.n, "side:", self.side, "(p1:", self.p1.to_string(), ", p2:",
                  self.p2.to_string())

