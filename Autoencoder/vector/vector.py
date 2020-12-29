import numpy as np


def angle(vec1, vec2, deg=False):
    _angle = np.arctan2(np.abs(np.cross(vec1, vec2)), np.dot(vec1, vec2))
    if deg:
        _angle = np.rad2deg(_angle)
    return _angle


if __name__ == "__main__":
    vec1 = np.array((1, 0))
    vec2 = np.array((-1, 0))
    print(angle(vec1,vec2))