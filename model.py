class Model:
    def __init__(self, thetas):
        if thetas is None or type(thetas) != list or len(thetas) < 2:
            raise(ValueError("Thetas must be an array of at least 3 matrices."))
