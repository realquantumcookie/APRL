import typing
import numpy as np
import random

GOAL_THRESHOLD = 1.0
def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

class GoalGraph():
    """
    A graph of goals that the car can drive to. Once the car arrives at a goal,
    the goal will be changed to one of its successors.
    """
    def __init__(self, scale, goal_config='corners', goal_threshold=GOAL_THRESHOLD):
        if goal_config == 'full':
            self.goals = [
                (0.05, 0.05),
                (0.33, -0.15),
                (0.3, -0.45),
                (0.65, -0.1),
                (0.525, 0.2),
                (0.6, 0.75),
                (0.3, 0.6),
                (-0.2, 0.6),
                (-0.5, 0.57),
                (-0.6, 0.05),
                (-0.7, -0.6),
                (0.0, -0.8),
            ]

            self.graph = {
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

            self.start_headings = [(0, 2 * np.pi)] * len(self.goals)
        elif goal_config == 'edges':
            self.goals = [
                (0.9, 0.0),
                (0.0, 0.9),
                (-0.9, 0.0),
                (0.0, -0.9),
            ]

            self.start_headings = [
                (a - 0.1, a + 0.1) for a in [
                    np.deg2rad(90),
                    np.deg2rad(180),
                    np.deg2rad(-90),
                    np.deg2rad(0),
                ]
            ]

            self.graph = {
                0: [1],
                1: [2],
                2: [3],
                3: [0],
            }
        elif goal_config == 'corners':
            self.goals = [
                (0.9, 0.9),
                (-0.9, 0.9),
                (-0.9, -0.9),
                (0.9, -0.9),
            ]

            self.start_headings = [
                (a - 0.1, a + 0.1) for a in [
                    np.deg2rad(135),
                    -np.deg2rad(135),
                    -np.deg2rad(45),
                    np.deg2rad(45),
                ]
            ]

            self.graph = {
                0: [1],
                1: [2],
                2: [3],
                3: [0],
            }
        elif goal_config == 'ring_dense':
            self.goals = [
                (0.9, 0.9),
                (0.0, 0.9),
                (-0.9, 0.9),
                (-0.9, 0.0),
                (-0.9, -0.9),
                (0.0, -0.9),
                (0.9, -0.9),
                (0.9, 0.0),
            ]

            self.start_headings = [
                (a - 0.1, a + 0.1) for a in [
                    np.deg2rad(135),
                    np.deg2rad(-180),
                    -np.deg2rad(135),
                    np.deg2rad(-90),
                    -np.deg2rad(45),
                    np.deg2rad(0),
                    -np.deg2rad(45),
                    np.deg2rad(90),
                ]
            ]

            self.graph = {
                0: [1, 2],
                1: [2, 3],
                2: [3, 4],
                3: [4, 5],
                4: [5, 6],
                5: [6, 7],
                6: [7, 0],
                7: [0, 1],
            }
        elif goal_config == 'ring_small_inner':
            self.goals = [
                (-0.10888671875, 0.609375),
                (0.07421875, 0.072265625),
                (-0.072265625, -0.220703125),
                (-0.5849609375, 0.072265625),
                (-0.49951171875, 0.67041015625),
            ]

            self.start_headings = [
                (a - 0.1, a + 0.1) for a in [
                    np.deg2rad(-45),
                    np.deg2rad(-110),
                    np.deg2rad(-120),
                    np.deg2rad(110),
                    np.deg2rad(45),
                ]
            ]

            self.graph = {
                0: [1],
                1: [2],
                2: [3],
                3: [4],
                4: [0],
            }
        elif goal_config == 'small_inner_1fork':
            self.goals = [
                (-0.10888671875, 0.609375),
                (0.07421875, 0.072265625),
                (-0.072265625, -0.220703125),
                (-0.5849609375, 0.072265625),
                (-0.49951171875, 0.67041015625),
                (-0.2919921875, -0.5380859375),
                (-0.6826171875, -0.46484375)
            ]

            self.start_headings = [
                (a - 0.1, a + 0.1) for a in [
                    np.deg2rad(-45),
                    np.deg2rad(-110),
                    np.deg2rad(-120),
                    np.deg2rad(110),
                    np.deg2rad(45),
                    np.deg2rad(-160),
                    np.deg2rad(90),
                ]
            ]

            self.graph = {
                0: [1],
                1: [2],
                2: [3, 5],
                3: [4],
                4: [0],
                5: [6],
                6: [3],
            }
        elif goal_config == 'small_inner_graph':
            self.goals = [
                (-0.10888671875, 0.609375),
                (0.0498046875, 0.1943359375),
                (-0.072265625, -0.220703125),
                (-0.5849609375, 0.072265625),
                (-0.49951171875, 0.67041015625),
                (-0.2919921875, -0.5380859375),
                (-0.6826171875, -0.46484375),
                (0.2939453125, -0.0498046875),
                (0.2939453125, -0.4892578125),
                (0.025390625, -0.806640625),
                (0.28, 0.53),
            ]

            self.start_headings = [
                (a - 0.3, a + 0.3) for a in [
                    np.deg2rad(-45),
                    np.deg2rad(-90),
                    np.deg2rad(-120),
                    np.deg2rad(110),
                    np.deg2rad(45),
                    np.deg2rad(-160),
                    np.deg2rad(90),
                    np.deg2rad(-30),
                    np.deg2rad(-110),
                    np.deg2rad(120),
                    np.deg2rad(90),
                ]
            ]

            self.graph = {
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
            }
        elif goal_config == 'small_inner_graph_reversed':
            self.goals = [
                (-0.10888671875, 0.609375),
                (0.0498046875, 0.1943359375),
                (-0.072265625, -0.220703125),
                (-0.5849609375, 0.072265625),
                (-0.49951171875, 0.67041015625),
                (-0.2919921875, -0.5380859375),
                (-0.6826171875, -0.46484375),
                (0.2939453125, -0.0498046875),
                (0.2939453125, -0.4892578125),
                (0.025390625, -0.806640625),
                (0.28, 0.53),
            ]

            self.start_headings = [
                (a - 0.3 + np.pi, a + 0.3 + np.pi) for a in [
                    np.deg2rad(-45),
                    np.deg2rad(-90),
                    np.deg2rad(-120),
                    np.deg2rad(110),
                    np.deg2rad(45),
                    np.deg2rad(-160),
                    np.deg2rad(90),
                    np.deg2rad(-30),
                    np.deg2rad(-110),
                    np.deg2rad(120),
                    np.deg2rad(90),
                ]
            ]

            self.graph = {
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
        else:
            raise ValueError(f"No such goal graph for {goal_config}")

        self.goal_reprs = []
        self.edge_reprs = []

        self.current_start_idx = 0
        self.current_goal_idx = 0

        self.goal_threshold = goal_threshold
        self.scale = scale
    
    @staticmethod
    def from_circle(origin : np.ndarray, radius : float, counterclockwise : bool, count : int, scale : float, goal_threshold = GOAL_THRESHOLD):
        """
        Generate a goal graph from a circle.
        """
        goals = []
        start_headings = []
        for i in range(count):
            angle = i * 2 * np.pi / count
            if not counterclockwise:
                angle = -angle
            pos = origin + radius * np.array([np.cos(angle), np.sin(angle)])
            goals.append(pos)
            start_heading = normalize_angle(angle + np.pi / 2 if counterclockwise else angle - np.pi / 2)
            start_headings.append((start_heading - 0.1, start_heading + 0.1))

        graph = {}
        for i in range(count):
            graph[i] = [(i + 1) % count]

        goal_graph = GoalGraph(scale,goal_threshold=goal_threshold)
        goal_graph.goals = goals
        goal_graph.start_headings = start_headings
        goal_graph.graph = graph
        return goal_graph
    

    @property
    def current_start(self):
        return np.array(self.goals[self.current_start_idx]) * self.scale, self.start_headings[self.current_start_idx]

    @property
    def current_goal(self):
        return np.array(self.goals[self.current_goal_idx]) * self.scale

    def is_complete(self):
        return self._ticks_at_current_goal > 0

    def set_goal(self, goal_idx, physics):
        """
        Set a new goal and update the renderables to match.
        """
        for idx, repr in enumerate(self.goal_reprs):
            # opacity = 1.0 if idx == goal_idx else 0.0
            opacity = 1 if idx < 7 else 0.0
            if physics:
                physics.bind(repr).rgba = (*repr.rgba[:3], opacity)
            else:
                repr.rgba = (*repr.rgba[:3], opacity)

        self.current_goal_idx = goal_idx
        self._ticks_at_current_goal = 0

    def tick(self, car_pos, physics):
        """
        Update the goal if the car was at the current goal for at least one tick.
        We need the delay so that the car can get the high reward for reaching
        the goal before the goal changes.
        """
        if self.is_complete():
            self.current_start_idx = self.current_goal_idx
            # self.set_goal(random.choice(self.graph[self.current_start_idx]), physics)
            self.set_goal(self.graph[self.current_start_idx][0], physics)
            return True

        if np.linalg.norm(np.array(car_pos)[:2] - self.current_goal) < self.goal_threshold:
            self._ticks_at_current_goal += 1
        else:
            self._ticks_at_current_goal = 0

        return False

    def reset(self, physics):
        # self.current_start_idx = random.randint(0, len(self.goals) - 1)
        self.current_start_idx = 0
        self.set_goal(random.choice(self.graph[self.current_start_idx]), physics)
        self._ticks_at_current_goal = 0

    def add_renderables(self, mjcf_root, height_lookup, show_edges=False):
        """
        Add renderables to the mjcf root to visualize the goals and (optionally) edges.
        """
        self.clear_renderables()
        RENDER_HEIGHT_OFFSET = 5.0

        self.goal_reprs = [
            mjcf_root.worldbody.add('site',
                                    type="sphere",
                                    size="0.1",
                                    rgba=(0.0, 1.0, 0.0, 0.5),
                                    group=0,
                                    pos=(g[0] * self.scale, g[1] * self.scale, height_lookup((g[0] * self.scale, g[1] * self.scale)) + RENDER_HEIGHT_OFFSET))
            for g in self.goals
        ]

        self.edge_reprs = [
            mjcf_root.worldbody.add('site',
                                    type="cylinder",
                                    size="0.05",
                                    rgba=(1, 1, 1, 0.5),
                                    fromto=(self.goals[s][0] * self.scale, self.goals[s][1] * self.scale, height_lookup((self.goals[s][0] * self.scale, self.goals[s][1] * self.scale)) + RENDER_HEIGHT_OFFSET,
                                            self.goals[g][0] * self.scale, self.goals[g][1] * self.scale, height_lookup((self.goals[g][0] * self.scale, self.goals[g][1] * self.scale)) + RENDER_HEIGHT_OFFSET))
            for s in self.graph for g in self.graph[s]
            if show_edges
        ]
    
    def clear_renderables(self):
        if self.goal_reprs is not None and len(self.goal_reprs) > 0:
            for r in self.goal_reprs:
                r.remove()
            self.goal_reprs = []
        
        if self.edge_reprs is not None and len(self.edge_reprs) > 0:
            for r in self.edge_reprs:
                r.remove()
            self.edge_reprs = []
        

def dfs_longest(edge_graph, start, visited = []):
    if start not in visited:
        visited.append(start)
    
    longest_e = []
    for e in edge_graph[start]:
        if e in visited:
            continue
        visited.append(e)

        c = dfs_longest(edge_graph, e, visited)
        if len(c) > len(longest_e):
            longest_e = c
    
    ret = [start]
    ret.extend(longest_e)
    return ret
    
class GoalRoute():
    def __init__(self, goal_graph : GoalGraph, start_idx : int = 0):
        self.route : typing.List = dfs_longest(goal_graph.graph,start_idx,[])
        self.graph = goal_graph.graph
        self.goal_threshold = goal_graph.goal_threshold
        self.goals = goal_graph.goals
        self.goal_start_headings = goal_graph.start_headings
        self.scale = goal_graph.scale
        self.goal_reprs = None
        self.edge_reprs = None
        self.route_reprs = None
        self.start_idx = 0
        self.current_line_segment_length = np.linalg.norm(self.current_start[0] - self.current_goal)
        self._compute_line_segment_lengths()

    def _compute_line_segment_lengths(self):
        self.line_segment_lengths = []
        for i in range(len(self.route) - 1):
            self.line_segment_lengths.append(
                np.linalg.norm(
                    np.array(self.goals[self.route[i]]) * self.scale - np.array(self.goals[self.route[i+1]]) * self.scale
                )
            )

    def reset(self, physics = None):
        self.set_current_start(0, physics)
    
    def set_current_start(self, start_idx, physics = None):
        if self.goal_reprs is not None:
            for idx, repr in enumerate(self.goal_reprs):
                is_start = idx == self.route[start_idx]
                is_goal = idx == self.route[start_idx + 1] if start_idx < len(self.route) - 1 else False

                if is_start:
                    rgba = (0.0, 0.0, 1.0, 1.0) #blue
                elif is_goal:
                    rgba = (1.0, 0.0, 0.0, 1.0) #red
                else:
                    rgba = (0.0, 1.0, 0.0, 0.5)


                if physics:
                    physics.bind(repr).rgba = rgba
                else:
                    repr.rgba = rgba
        if self.route_reprs is not None:
            for idx, repr in enumerate(self.route_reprs):
                is_current = idx == start_idx

                if is_current:
                    rgba = (1.0, 0.0, 0.0, 1.0) #solid red
                else:
                    rgba = (1.0, 1.0, 0.0, 0.5) #purple


                if physics:
                    physics.bind(repr).rgba = rgba
                else:
                    repr.rgba = rgba

        print("New Goal Route start idx: ", start_idx)
        self.start_idx = start_idx
        if(start_idx < len(self.route) - 1 and start_idx >= 0):
            self.current_line_segment_length = self.line_segment_lengths[start_idx]
            print("current line segment length: ", self.current_line_segment_length)
        else:
            self.current_line_segment_length = 0.0
        

    def add_renderables(self, mjcf_root, height_lookup, show_edges = False, render_height_offset = 5.0):
        self.clear_renderables()
        self.goal_reprs = [
             mjcf_root.worldbody.add('site',
                                    type="sphere",
                                    size="0.1",
                                    rgba=(0.0, 1.0, 0.0, 0.5), #green
                                    group=0,
                                    pos=(g[0] * self.scale, g[1] * self.scale, height_lookup((g[0] * self.scale, g[1] * self.scale)) + render_height_offset))
            for g in self.goals
        ]
        if show_edges:
            self.edge_reprs = []
            self.route_reprs = []
            for s in self.graph:
                try:
                    indexOfS = self.route.index(s)
                except ValueError:
                    indexOfS = -1

                for g in self.graph[s]:
                    
                    try:
                        indexOfG = self.route.index(g)
                    except ValueError:
                        indexOfG = -1
                    
                    if indexOfS == indexOfG - 1 and indexOfS != -1:
                        pass
                        """
                        In-route edges are added here in previous versions, but now they are added later in the code to align with the self.route layout
                        """
                    else:
                        self.edge_reprs.append(mjcf_root.worldbody.add(
                            'site',
                            type="cylinder",
                            size="0.04",
                            rgba=(1, 1, 1, 0.5), #white
                            fromto=(
                                self.goals[s][0] * self.scale, self.goals[s][1] * self.scale, height_lookup((self.goals[s][0] * self.scale, self.goals[s][1] * self.scale)) + RENDER_HEIGHT_OFFSET,
                                self.goals[g][0] * self.scale, self.goals[g][1] * self.scale, height_lookup((self.goals[g][0] * self.scale, self.goals[g][1] * self.scale)) + RENDER_HEIGHT_OFFSET
                            )
                        ))
            for s_idx,g_idx  in zip(self.route[:-1], self.route[1:]):
                s,g = self.goals[s_idx], self.goals[g_idx]
                self.route_reprs.append(mjcf_root.worldbody.add(
                    'site',
                    type="cylinder",
                    size="0.04",
                    rgba=(1, 1, 0, 0.5), #purple
                    fromto=(
                        s[0] * self.scale, s[1] * self.scale, height_lookup((s[0] * self.scale, s[1] * self.scale)) + render_height_offset,
                        g[0] * self.scale, g[1] * self.scale, height_lookup((g[0] * self.scale, g[1] * self.scale)) + render_height_offset
                    )
                ))

        self.set_current_start(self.start_idx)
    
    @property
    def route_length(self):
        return len(self.route) - 1

    def get_closest_point_on_current_segment(self, loc: np.ndarray) -> np.ndarray:
        assert loc.ndim == 1 and loc.shape[0] >= 2

        loc = loc[:2]
        line_normal = self.current_line_normal_vector
        current_start = self.current_start[0]
        current_goal = self.current_goal
        projected_point = loc + np.dot(self.current_start[0] - loc, line_normal) * line_normal
        
        dist_to_start = np.linalg.norm(projected_point - current_start)
        dist_to_goal = np.linalg.norm(projected_point - current_goal)
        line_segment_length = self.current_line_segment_length
        if dist_to_start > line_segment_length or dist_to_goal > line_segment_length:
            return current_start if dist_to_start < dist_to_goal else current_goal
        else:
            return projected_point
    
    @property
    def current_line_normal_vector(self) -> np.ndarray:
        # get the vector perpendicular to the current line segment
        # if self.is_finished:
        #     raise ValueError("Route finished, call reset() to make it valid again")
        
        start_vec = self.current_start[0]
        end_vec = self.current_goal
        unnormalized = np.array([end_vec[1] - start_vec[1], start_vec[0] - end_vec[0]])
        return unnormalized / np.linalg.norm(unnormalized)

    @property
    def is_finished(self) -> bool:
        return self.start_idx + 1 >= len(self.route) or self.start_idx < 0

    def current_progress_on_route(self, pos) -> float:
        if self.is_finished:
            return 1.0

        pos_on_line = self.get_closest_point_on_current_segment(pos)
        dist_to_start = np.linalg.norm(pos_on_line - self.current_start[0])
        total_dist_to_start = np.sum(self.line_segment_lengths[:self.start_idx])
        p = (total_dist_to_start + dist_to_start) / np.sum(self.line_segment_lengths)
        #assert 0 <= p <= 1
        #print(p)
        return p

    def tick(self, car_pos, physics):
        if self.is_finished:
            raise ValueError("Route finished, call reset() to make it valid again")
        if np.linalg.norm(np.array(car_pos)[:2] - self.current_goal) < self.goal_threshold:
            print("Segment " + str(self.start_idx+1) + "/" + str(len(self.route) - 1) + " finished")
            self.set_current_start(self.start_idx + 1, physics)
            if self.is_finished:
                print("Route Finished")
            return True

        return False

    @property
    def current_start(self) -> typing.Tuple[np.ndarray, typing.Tuple[float,float]]:
        return self.get_start_for_idx(self.start_idx), self.get_start_headings_for_idx(self.start_idx)

    @property
    def current_goal(self) -> np.ndarray:
        return self.get_goal_for_idx(self.start_idx)
    
    def get_goal_for_idx(self, idx) -> np.ndarray:
        if idx >= self.route_length:
            return np.array(self.goals[self.route[-1]]) * self.scale
        else:
            return np.array(self.goals[self.route[idx + 1]]) * self.scale

    def get_start_for_idx(self, idx) -> np.ndarray:
        if idx >= self.route_length:
            return np.array(self.goals[self.route[-2]]) * self.scale
        else:
            return np.array(self.goals[self.route[idx]]) * self.scale
    
    def get_start_headings_for_idx(self, idx) -> typing.Tuple[float,float]:
        if idx >= self.route_length:
            return self.get_start_headings_for_idx(self.route_length - 1)
        else:
            return self.goal_start_headings[idx]

    def clear_renderables(self):
        if self.goal_reprs is not None:
            for goal_repr in self.goal_reprs:
                goal_repr.remove()
        
        if self.edge_reprs is not None:
            for edge_repr in self.edge_reprs:
                edge_repr.remove()
        
        if self.route_reprs is not None:
            for route_repr in self.route_reprs:
                route_repr.remove()