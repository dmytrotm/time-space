
import json

class TapeDeviationDetector:
    def __init__(self,x,w,dx,dw):
        self.x = x
        self.w = w
        self.dx = dx
        self.dw = dw
    
    def is_tape_correct(self,x,w)->int:
        '''
            -1 - too far
            1 - too long or too short
            0 - correct
        '''
        if abs(self.x-x) > self.dx:
            return -1
        elif abs(self.w-w) > self.dw:
            return 1
        return 0 

def tape_deviation_detector_factory(index:int)->TapeDeviationDetector:
    if(index>7 or index<0):
        raise Exception("Wrong tape index")
    try:

        with open("positions.json", "r") as file:
            data = json.load(file)

        x = data[index*2]["Mean"]
        dx = 0.3
        w = data[index*2+1]["Mean"]
        dw = data[index*2+1]["IQR"] * 2

    except FileNotFoundError:

        x = 0.4
        dx = 0.25
        w = 0.25
        dw = 0.06 

    return TapeDeviationDetector(x,w,dx,dw)