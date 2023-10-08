import typing
import numpy as np
import random

HEIGHTFIELD_ARENA_GOALS = {
    "full": {
        "points": [
            (np.array([0.05, 0.05]), (0.0, 2 * np.pi)),
            (np.array([0.33, -0.15]), (0.0, 2 * np.pi)),
            (np.array([0.3, -0.45]), (0.0, 2 * np.pi)),
            (np.array([0.65, -0.1]), (0.0, 2 * np.pi)),
            (np.array([0.525, 0.2]), (0.0, 2 * np.pi)),
            (np.array([0.6, 0.75]), (0.0, 2 * np.pi)),
            (np.array([0.3, 0.6]), (0.0, 2 * np.pi)),
            (np.array([-0.2, 0.6]), (0.0, 2 * np.pi)),
            (np.array([-0.5, 0.57]), (0.0, 2 * np.pi)),
            (np.array([-0.6, 0.05]), (0.0, 2 * np.pi)),
            (np.array([-0.7, -0.6]), (0.0, 2 * np.pi)),
            (np.array([0.0, -0.8]), (0.0, 2 * np.pi)),
        ],
        "connections": {
            0: [1, 6, 7, 9, 10, 11],
            1: [0, 2, 3, 4],
            2: [1, 3, 11],
            3: [1, 2, 4],
            4: [1, 3, 6],
            5: [4, 6],
            6: [0, 4, 5, 7],
            7: [0, 6, 8],
            8: [7, 9],
            9: [0, 8, 10],
            10: [0, 9, 11],
            11: [0, 2, 10],
        }
    },
    "edges": {
        "points": [
            (np.array([0.9, 0.0]), (np.deg2rad(90) - 0.1, np.deg2rad(90) + 0.1)),
            (np.array([0.0, 0.9]), (np.deg2rad(180) - 0.1, np.deg2rad(180) + 0.1)),
            (np.array([-0.9, 0.0]), (np.deg2rad(-90) - 0.1, np.deg2rad(-90) + 0.1)),
            (np.array([0.0, -0.9]), (np.deg2rad(0) - 0.1, np.deg2rad(0) + 0.1)),
        ],
        "connections": {
            0: [1],
            1: [2],
            2: [3],
            3: [0],
        }
    },
    "corners": {
        "points": [
            (np.array([0.9, 0.9]), (np.deg2rad(135) - 0.1, np.deg2rad(135) + 0.1)),
            (np.array([-0.9, 0.9]), (np.deg2rad(-135) - 0.1, np.deg2rad(-135) + 0.1)),
            (np.array([-0.9, -0.9]), (np.deg2rad(-45) - 0.1, np.deg2rad(-45) + 0.1)),
            (np.array([0.9, -0.9]), (np.deg2rad(45) - 0.1, np.deg2rad(45) + 0.1)),
        ],
        "connections": {
            0: [1],
            1: [2],
            2: [3],
            3: [0],
        }
    },
    "ring_dense": {
        "points": [
            (np.array([0.9, 0.9]), (np.deg2rad(135) - 0.1, np.deg2rad(135) + 0.1)),
            (np.array([0.0, 0.9]), (np.deg2rad(-180) - 0.1, np.deg2rad(-180) + 0.1)),
            (np.array([-0.9, 0.9]), (np.deg2rad(-135) - 0.1, np.deg2rad(-135) + 0.1)),
            (np.array([-0.9, 0.0]), (np.deg2rad(-90) - 0.1, np.deg2rad(-90) + 0.1)),
            (np.array([-0.9, -0.9]), (np.deg2rad(-45) - 0.1, np.deg2rad(-45) + 0.1)),
            (np.array([0.0, -0.9]), (np.deg2rad(0) - 0.1, np.deg2rad(0) + 0.1)),
            (np.array([0.9, -0.9]), (np.deg2rad(-45) - 0.1, np.deg2rad(-45) + 0.1)),
            (np.array([0.9, 0.0]), (np.deg2rad(90) - 0.1, np.deg2rad(90) + 0.1)),
        ],
        "connections": {
            0: [1, 2],
            1: [2, 3],
            2: [3, 4],
            3: [4, 5],
            4: [5, 6],
            5: [6, 7],
            6: [7, 0],
            7: [0, 1],
        }
    },
    "ring_small_inner": {
        "points": [
            (np.array([-0.10888671875, 0.609375]), (np.deg2rad(-45) - 0.1, np.deg2rad(-45) + 0.1)),
            (np.array([0.07421875, 0.072265625]), (np.deg2rad(-110) - 0.1, np.deg2rad(-110) + 0.1)),
            (np.array([0.0, 0.0]), (np.deg2rad(-110) - 0.1, np.deg2rad(-110) + 0.1)),
            (np.array([-.1, -0.25]), (np.deg2rad(-120) - 0.1, np.deg2rad(-120) + 0.1)),
            (np.array([-0.45, 0.1]), (np.deg2rad(110) - 0.1, np.deg2rad(110) + 0.1)),
            (np.array([-0.5849609375, 0.072265625]), (np.deg2rad(110) - 0.1, np.deg2rad(110) + 0.1)),
            (np.array([-0.49951171875, 0.67041015625]), (np.deg2rad(45) - 0.1, np.deg2rad(45) + 0.1)),
        ],
        "connections": {
            0: [1],
            1: [2],
            2: [3],
            3: [4],
            4: [5],
            5: [6],
            6: [0],
        }
    },
    "small_inner_1fork": {
        "points": [
            (np.array([-0.10888671875, 0.609375]), (np.deg2rad(-45) - 0.1, np.deg2rad(-45) + 0.1)),
            (np.array([0.07421875, 0.072265625]), (np.deg2rad(-110) - 0.1, np.deg2rad(-110) + 0.1)),
            (np.array([-0.072265625, -0.220703125]), (np.deg2rad(-120) - 0.1, np.deg2rad(-120) + 0.1)),
            (np.array([-0.5849609375, 0.072265625]), (np.deg2rad(110) - 0.1, np.deg2rad(110) + 0.1)),
            (np.array([-0.49951171875, 0.67041015625]), (np.deg2rad(45) - 0.1, np.deg2rad(45) + 0.1)),
            (np.array([-0.2919921875, -0.5380859375]), (np.deg2rad(-160) - 0.1, np.deg2rad(-160) + 0.1)),
            (np.array([-0.6826171875, -0.46484375]), (np.deg2rad(90) - 0.1, np.deg2rad(90) + 0.1)),
        ],
        "connections": {
            0: [1],
            1: [2],
            2: [3, 5],
            3: [4],
            4: [0],
            5: [6],
            6: [3],
        }
    },
    "hike": {
        "points": [
            (np.array([-0.10888671875, 0.609375]), (np.deg2rad(-45) - 0.1, np.deg2rad(-45) + 0.1)),
            (np.array([0.07421875, 0.072265625]), (np.deg2rad(-110) - 0.1, np.deg2rad(-110) + 0.1)),
            (np.array([-0.5849609375, 0.072265625]), (np.deg2rad(110) - 0.1, np.deg2rad(110) + 0.1)),
            (np.array([-0.49951171875, 0.67041015625]), (np.deg2rad(45) - 0.1, np.deg2rad(45) + 0.1)),
            # (np.array([-0.45, -0.35]), (np.deg2rad(90) - 0.1, np.deg2rad(90) + 0.1)),
            (np.array([-.35, -.05]), (np.deg2rad(90) - 0.1, np.deg2rad(90) + 0.1)),
            # (np.array([-.3, 0.3]), (np.deg2rad(90) - 0.1, np.deg2rad(90) + 0.1)),
        ],
        "connections": {
            0: [1],
            1: [4],
            # 5: [4],
            4: [2],
            2: [3],
            3: [0],

        }
    },
    "dense_inner": {
        "points": [
            # our original ring
            (np.array([-0.1, 0.58]), (np.deg2rad(-45) - 0.1, np.deg2rad(-45) + 0.1)), # 0
            (np.array([0.1, 0.072265625]), (np.deg2rad(-110) - 0.1, np.deg2rad(-110) + 0.1)), # 1
            (np.array([0.0, 0.0]), (np.deg2rad(-110) - 0.1, np.deg2rad(-110) + 0.1)), # 2
            (np.array([-.1, -0.25]), (np.deg2rad(-120) - 0.1, np.deg2rad(-120) + 0.1)), # 3
            (np.array([-0.45, 0.1]), (np.deg2rad(110) - 0.1, np.deg2rad(110) + 0.1)),  # 4  
            (np.array([-0.57, 0.05]), (np.deg2rad(110) - 0.1, np.deg2rad(110) + 0.1)), # 5
            (np.array([-0.52, 0.62]), (np.deg2rad(45) - 0.1, np.deg2rad(45) + 0.1)), # 6
            (np.array([-0.45, 0.7]), (np.deg2rad(45) - 0.1, np.deg2rad(45) + 0.1)), # 7
        ],
        "connections": {
            0: [1, 7],
            1: [0, 2],
            2: [1, 3],
            3: [2, 4],
            4: [5, 3],
            5: [6, 4],
            6: [5, 7],
            7: [6, 0],
        }
    },
    "full_dense": {
        "points": [
            # our original ring
            (np.array([-0.1, 0.58]), (np.deg2rad(-45) - 0.1, np.deg2rad(-45) + 0.1)),
            (np.array([0.1, 0.072265625]), (np.deg2rad(-110) - 0.1, np.deg2rad(-110) + 0.1)),
            (np.array([0.0, 0.0]), (np.deg2rad(-110) - 0.1, np.deg2rad(-110) + 0.1)),
            (np.array([-.1, -0.25]), (np.deg2rad(-120) - 0.1, np.deg2rad(-120) + 0.1)),
            (np.array([-0.3, -.05]), (np.deg2rad(110) - 0.1, np.deg2rad(110) + 0.1)), # 4
            (np.array([-0.57, 0.05]), (np.deg2rad(110) - 0.1, np.deg2rad(110) + 0.1)), # 5
            (np.array([-0.52, 0.62]), (np.deg2rad(45) - 0.1, np.deg2rad(45) + 0.1)), # 6
            # (np.array([-0.45, 0.7]), (np.deg2rad(45) - 0.1, np.deg2rad(45) + 0.1)), # 
            # now expand the left bottom corner
            (np.array([-0.7, -0.6]), (np.deg2rad(45) - 0.1, np.deg2rad(45) + 0.1)),
            (np.array([-.3, -.55]), (np.deg2rad(45) - 0.1, np.deg2rad(45) + 0.1)),
            # from bottom left corner back to center
            (np.array([0.1, -0.85]), (np.deg2rad(90) - 0.1, np.deg2rad(90) + 0.1)),
            (np.array([0.3, -0.5]), (np.deg2rad(90) - 0.1, np.deg2rad(90) + 0.1)),
            (np.array([0.325, -0.07]), (np.deg2rad(90) - 0.1, np.deg2rad(90) + 0.1)),
            (np.array([0.17, -0.05]), (np.deg2rad(90) - 0.1, np.deg2rad(90) + 0.1)),
            (np.array([0.27, 0.57]), (np.deg2rad(90) - 0.1, np.deg2rad(90) + 0.1)),
            (np.array([0.1, 0.75]), (np.deg2rad(90) - 0.1, np.deg2rad(90) + 0.1)),
            (np.array([0.6, 0.05]), (np.deg2rad(90) - 0.1, np.deg2rad(90) + 0.1)),
            (np.array([0.5, 0]), (np.deg2rad(90) - 0.1, np.deg2rad(90) + 0.1)),
            (np.array([0.6, -.3]), (np.deg2rad(90) - 0.1, np.deg2rad(90) + 0.1)),
            (np.array([-0.45, 0.7]), (np.deg2rad(45) - 0.1, np.deg2rad(45) + 0.1)), # 7
        ],
        "connections": {
            0: [1, 18, 14],
            1: [0, 2, 12],
            2: [1, 3, 4],
            3: [2, 4, 8],
            4: [5, 3, 2, 7],
            5: [6, 7],
            6: [18, 5],
            7: [5, 8],
            8: [3, 7, 9],
            9: [8, 10],
            10: [9, 11, 17],
            11: [10, 12, 16],
            12: [11, 1, 13],
            13: [14, 12, 15],
            14: [13, 0],
            15: [16, 13],
            16: [15, 11, 17],
            17: [16, 10],
            18: [6, 0],
        }
    },
    # "hike": {
    #     "points": [
    #         (np.array([-0.10888671875, 0.609375]), (np.deg2rad(-45) - 0.1, np.deg2rad(-45) + 0.1)),
    #         (np.array([0.07421875, 0.072265625]), (np.deg2rad(-110) - 0.1, np.deg2rad(-110) + 0.1)),
    #         (np.array([-0.5849609375, 0.072265625]), (np.deg2rad(110) - 0.1, np.deg2rad(110) + 0.1)),
    #         (np.array([-0.49951171875, 0.67041015625]), (np.deg2rad(45) - 0.1, np.deg2rad(45) + 0.1)),
    #         (np.array([-0.45, -0.35]), (np.deg2rad(90) - 0.1, np.deg2rad(90) + 0.1)),
    #         (np.array([-.35, -.05]), (np.deg2rad(90) - 0.1, np.deg2rad(90) + 0.1)),
    #         # (np.array([-.3, 0.3]), (np.deg2rad(90) - 0.1, np.deg2rad(90) + 0.1)),
    #     ],
    #     "connections": {
    #         0: [1],
    #         1: [5],
    #         5: [4],
    #         4: [2],
    #         2: [3],
    #         3: [0],

    #     }
    # },
    "small_inner_graph": {
        "points": [
            (np.array([-0.10888671875, 0.609375]), (np.deg2rad(-45) - 0.1, np.deg2rad(-45) + 0.1)),
            (np.array([0.0498046875, 0.1943359375]), (np.deg2rad(-90) - 0.1, np.deg2rad(-90) + 0.1)),
            (np.array([-0.072265625, -0.220703125]), (np.deg2rad(-120) - 0.1, np.deg2rad(-120) + 0.1)),
            (np.array([-0.5849609375, 0.072265625]), (np.deg2rad(110) - 0.1, np.deg2rad(110) + 0.1)),
            (np.array([-0.49951171875, 0.67041015625]), (np.deg2rad(45) - 0.1, np.deg2rad(45) + 0.1)),
            (np.array([-0.2919921875, -0.5380859375]), (np.deg2rad(-160) - 0.1, np.deg2rad(-160) + 0.1)),
            (np.array([-0.6826171875, -0.46484375]), (np.deg2rad(90) - 0.1, np.deg2rad(90) + 0.1)),
            (np.array([0.2939453125, -0.0498046875]), (np.deg2rad(-30) - 0.1, np.deg2rad(-30) + 0.1)),
            (np.array([0.2939453125, -0.4892578125]), (np.deg2rad(-110) - 0.1, np.deg2rad(-110) + 0.1)),
            (np.array([0.025390625, -0.4892578125]), (np.deg2rad(120) - 0.1, np.deg2rad(120) + 0.1)),
            (np.array([0.28, 0.53]), (np.deg2rad(90) - 0.1, np.deg2rad(90) + 0.1)),
        ],
        "connections": {
            0: [1],
            1: [2, 7, 10],
            2: [3, 5],
            3: [4],
            4: [0],
            5: [6],
            6: [3],
            7: [8, 10],
            8: [9],
            9: [5],
            10: [0],
        },
    },
    "small_inner_graph_reversed": {
        "points": [
            (np.array([-0.10888671875, 0.609375]), (np.deg2rad(-45) - 0.1, np.deg2rad(-45) + 0.1)),
            (np.array([0.0498046875, 0.1943359375]), (np.deg2rad(-90) - 0.1, np.deg2rad(-90) + 0.1)),
            (np.array([-0.072265625, -0.220703125]), (np.deg2rad(-120) - 0.1, np.deg2rad(-120) + 0.1)),
            (np.array([-0.5849609375, 0.072265625]), (np.deg2rad(110) - 0.1, np.deg2rad(110) + 0.1)),
            (np.array([-0.49951171875, 0.67041015625]), (np.deg2rad(45) - 0.1, np.deg2rad(45) + 0.1)),
            (np.array([-0.2919921875, -0.5380859375]), (np.deg2rad(-160) - 0.1, np.deg2rad(-160) + 0.1)),
            (np.array([-0.6826171875, -0.46484375]), (np.deg2rad(90) - 0.1, np.deg2rad(90) + 0.1)),
            (np.array([0.2939453125, -0.0498046875]), (np.deg2rad(-30) - 0.1, np.deg2rad(-30) + 0.1)),
            (np.array([0.2939453125, -0.4892578125]), (np.deg2rad(-110) - 0.1, np.deg2rad(-110) + 0.1)),
            (np.array([0.025390625, -0.4892578125]), (np.deg2rad(120) - 0.1, np.deg2rad(120) + 0.1)),
            (np.array([0.28, 0.53]), (np.deg2rad(90) - 0.1, np.deg2rad(90) + 0.1)),
        ],
        "connections": {
            0: [4, 10],
            1: [0],
            2: [1],
            3: [2, 6],
            4: [3],
            5: [2, 9],
            6: [5],
            7: [1],
            8: [7],
            9: [8],
            10: [1, 7],
        }
    }
}

