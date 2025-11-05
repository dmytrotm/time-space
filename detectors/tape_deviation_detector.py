class TapeDeviationDetector:
    def __init__(self, positions_json):
        self.positions_json = positions_json

    def is_tape_correct(self, index, new_x, new_w) -> int:
        """
        -1 - too far
        1 - too long or too short
        0 - correct
        """

        x = self.positions_json[index * 2]["Mean"]
        dx = 0.3
        w = self.positions_json[index * 2 + 1]["Mean"]
        dw = self.positions_json[index * 2 + 1]["IQR"] * 2

        if abs(x - new_x) > dx:
            return -1
        elif abs(w - new_w) > dw:
            return 1
        return 0
