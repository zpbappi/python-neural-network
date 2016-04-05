class Model:
    def __init__(self, thetas):
        def has_valid_dimensions(matrics):
            collection1 = matrics[:-1]
            collection2 = matrics[1:]
            for (x, y) in zip(collection1, collection2):
                rx, _ = x.shape
                _, ry = y.shape
                if rx + 1 != ry:
                    return False
            return True

        if thetas is None or type(thetas) != list or len(thetas) < 2:
            raise(ValueError("Thetas must be an array of at least 3 matrices."))

        if not has_valid_dimensions(thetas):
            raise(ValueError("Thetas should have dimensions like: (a,b), (b+1,c), (c+1,d)..."))
