import numpy as np
import random
import cv2
import gym


class DisjointSet:
    '''
    Disjoint Set : Utility class that helps implement Kruskal MST algorithm
        Allows to check whether to keys belong to the same set and to union
        sets together
    '''
    class Element:
        def __init__(self, key):
            self.key = key
            self.parent = self
            self.rank = 0

        def __eq__(self, other):
            return self.key == other.key

        def __ne__(self, other):
            return self.key != other.key

    def __init__(self):
        '''
        Tree = element map where each node is a (key, parent, rank) 
        Sets are represented as subtrees whose root is identified with
        a self referential parent
        '''
        self.tree = {}

    def make_set(self, key):
        '''
        Creates a new singleton set.
        @params 
            key : id of the element
        @return
            None
        '''
        # Create and add a new element to the tree
        e = self.Element(key)
        if not key in self.tree.keys():
            self.tree[key] = e

    def find(self, key):
        '''
        Finds a given element in the tree by the key.
        @params 
            key(hashable) : id of the element
        @return
            Element : root of the set which contains element with the key
        '''
        if key in self.tree.keys():
            element = self.tree[key]
            # root is element with itself as parent
            # if not root continue
            if element.parent != element:
                element.parent = self.find(element.parent.key)
            return element.parent

    def union(self, element_a, element_b):
        '''
        Creates a new set that contains all elements in both element_a and element_b's sets
        Pass into union the Elements returned by the find operation
        @params 
            element_a(Element) : Element or key of set a
            element_b(Element) : Element of set b
        @return
            None
        '''
        root_a = self.find(element_a.key)
        root_b = self.find(element_b.key)
        # if not in the same subtree (set)
        if root_a != root_b:
            # merge the sets
            if root_a.rank < root_b.rank:
                root_a.parent = root_b
            elif root_a.rank > root_b.rank:
                root_b.parent = root_a
            else:
                # same rank, set and increment arbitrary root as parent
                root_b.parent = root_a
                root_a.rank += 1


class Maze:
    # static variables
    # Directions to move the player.
    # Note, up and down are reversed (visual up and down not grid coordinates)
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

    def __init__(self, width, height, seed, symbols, scaling):
        '''
        Default constructor to create an widthXheight maze
        @params 
            width(int)    : number of columns
            height(int)    : number of rows
            seed(float)    : number to seed RNG
            symbols(dict)    : used to modify maze symbols and colors
                            settings{
                                start, end, start_color, end_color, : start and end symbols and colors
                                wall_v, wall_h, wall_c, wall_color, : vertical,horizontal and corner wall symbols and colors 
                                head, tail, head_color, tail_color   : player head and trail symbols and colors
                                *_bg_color, : substitute _color with bg_color to set background colors 
        @return                                                
            Maze    : constructed object
        '''
        assert width > 0
        assert height > 0
        self.init_symbols(symbols)
        self.time_taken = False
        self.timer_thread = None
        self.is_moving = True  # used as a semaphore for the update time thread
        self.width = width
        self.height = height
        self.seed = seed
        self.scaling = scaling
        self.path = []  # current path taken
        self.player = (0, 0)  # players position
        # self.items = [(x,y)] #TODO?? Add a list of possible items to collect for points?
        # Creates 2-D array of cells(unique keys)
        # Grid is 2-D, and the unique ids are sequential, i
        # so uses a 2-D to 1-D mapping
        # to get the key since row+col is not unique for all rows and columns
        #    E.g.
        #    width = 5
        #            1-D Mapping    vs      Naive
        #    grid[2][3] =     5*2+3 = 13     vs      2+3 = 6
        #    grid[3][2] =    5*3+2 = 17    vs     3+2 = 6 X Not unique!
        # use 2D list comprehensions to avoid iterating twice
        self.grid = [[(width * row + col)
                      for row in range(0, height)]
                     for col in range(0, width)]
        # portals[key] = {keys of neighbors}
        self.portals = {}
        # generate the maze by using a kruskals algorithm
        self.kruskalize()

    def __repr__(self):
        '''
        Allows for print(maze)
        @params
            None
        @return
            String : Ascii representation of the Maze
        '''
        return self.to_str()

    def to_str(self):
        '''
        Defines the string representation of the maze.
        @return
            Maze    : constructed object
        '''
        s = ''
        for col in range(0, self.width):
            s += self.wall_c + self.wall_h

        s += self.wall_c + '\n'
        # wall if region not the same
        for row in range(0, self.height):
            # draw S for start if at (0,0)
            if row == 0:
                s += self.wall_v + self.start
            else:
                s += self.wall_v + self.empty

            # draw | if no portal between [row][col] and [row][col-1]
            for col in range(1, self.width):
                # if  theres a portal between cell and left cell
                if self.grid[col - 1][row] in self.portals[self.grid[col][row]]:
                    # if portal remove wall
                    c = self.empty
                else:
                    # if not portal draw vertical wall
                    c = self.wall_v
                # if at [width-1][height-1] draw end marker or cell
                if row == self.height - 1 and col == self.width - 1:
                    c += self.end
                else:  # draw cell
                    c += self.empty
                s += c
            s += self.wall_v + '\n'
            # draw - if not portal between [row][col] and [row+1][col]
            for col in range(0, self.width):
                # if edge above (visually below)
                c = self.wall_h
                key = self.grid[col][row]
                # if not at last row, and theres a portal between cell and
                # above cell
                if row + 1 < self.height and self.grid[col][row + 1] in self.portals[key]:
                    c = self.empty
                s += self.wall_c + c
            s += self.wall_c + '\n'
        s += self.empty
        return s

    def to_np(self):
        s = np.zeros((2 * self.height + 1, 2 * self.width + 1), dtype=np.int)
        # print(s.shape)
        for col in range(0, 2 * self.width + 1):
            s[0][col] = 1
        for row in range(0, self.height):
            s[2 * row + 1][0] = 1
            s[2 * row + 1][1] = 0
            for col in range(1, self.width):
                if self.grid[col - 1][row] in self.portals[self.grid[col][row]]:
                    # if portal remove wall
                    s[2 * row + 1][2 * col] = 0
                else:
                    # if not portal draw vertical wall
                    s[2 * row + 1][2 * col] = 1
                # if at [width-1][height-1] draw end marker or cell
                if row == self.height - 1 and col == self.width - 1:
                    s[2 * row + 1][2 * col + 1] = 0
            s[2 * row + 1][2 * self.width] = 1
            for col in range(0, self.width):
                # if edge above (visually below)
                c = 1
                key = self.grid[col][row]
                # if not at last row, and theres a portal between cell and
                # above cell
                if row + 1 < self.height and self.grid[col][row + 1] in self.portals[key]:
                    c = 0
                s[2 * row + 2][2 * col] = 1
                if c == 1:
                    s[2 * row + 2][2 * col + 1] = 1
                else:
                    s[2 * row + 2][2 * col + 1] = 0
            s[2 * row + 2][-1] = 1
        return s

    def scale(self, map_np):
        new_map_np = np.zeros(
            (self.scaling * map_np.shape[0], self.scaling * map_np.shape[1]))
        for i in range(map_np.shape[0]):
            for j in range(map_np.shape[1]):
                if map_np[i][j] == 1:
                    for k in range(self.scaling):
                        for l in range(self.scaling):
                            new_map_np[self.scaling * i +
                                       k][self.scaling * j + l] = 1
                else:
                    for k in range(self.scaling):
                        for l in range(self.scaling):
                            new_map_np[self.scaling * i +
                                       k][self.scaling * j + l] = 0
        return new_map_np

    def portals_str(self):
        '''
        Returns a string containing a list of all portal coordinates
        '''
        i = 1
        s = 'Portal Coordinates\n'
        for key, portals in self.portals.items():
            for near in portals.keys():
                                # print the cell ids
                s += '%-015s' % (str((key, near)))
                # draw 5 portals coordinates per line
                if i % 5 == 0:
                    s += '\n'
                i += 1
        return s

    def init_symbols(self, symbols):
        # get symbol colors _color + bg_color

        start_color = symbols['start_color'] if 'start_color' in symbols else ''
        start_bg_color = symbols['start_bg_color'] if 'start_bg_color' in symbols else ''

        end_color = symbols['end_color'] if 'end_color' in symbols else ''
        end_bg_color = symbols['end_bg_color'] if 'end_bg_color' in symbols else ''

        wall_color = symbols['wall_color'] if 'wall_color' in symbols else ''
        wall_bg_color = symbols['wall_bg_color'] if 'wall_bg_color' in symbols else''

        head_color = symbols['head_color'] if 'head_color' in symbols else ''
        head_bg_color = symbols['head_bg_color'] if 'head_bg_color' in symbols else ''

        tail_color = symbols['tail_color'] if 'tail_color' in symbols else ''
        tail_bg_color = symbols['tail_bg_color'] if 'tail_bg_color' in symbols else ''

        empty_color = symbols['empty_color'] if 'empty_color' in symbols else ''

        # symbol colors
        self.start = start_bg_color + start_color + symbols['start']
        self.end = end_bg_color + end_color + symbols['end'] + empty_color
        self.wall_h = wall_bg_color + wall_color + symbols['wall_h']
        self.wall_v = wall_bg_color + wall_color + symbols['wall_v']
        self.wall_c = wall_bg_color + wall_color + symbols['wall_c']
        self.head = head_bg_color + head_color + symbols['head']
        self.tail = tail_bg_color + tail_color + symbols['tail']
        self.empty = empty_color + ' '

    def kruskalize(self):
        '''
        Kruskal's algorithm, except when grabbing the next available edge, 
        order is randomized. 
        Uses a disjoint set to create a set of keys. 
        Then for each edge seen, the key for each cell is used to determine 
        whether or not the the keys are in the same set.
        If they are not, then the two sets each key belongs to are unioned.
        Each set represents a region on the maze, this finishes until all
        keys are reachable (MST definition) or rather all keys are unioned into 
        single set. 
        @params
            None 
        @return
            None
        '''
        # edge = ((row1, col1), (row2, col2)) such that grid[row][col] = key
        edges_ordered = []
        # First add all neighboring edges into a list
        for row in range(0, self.height):
            for col in range(0, self.width):
                cell = (col, row)
                left_cell = (col - 1, row)
                down_cell = (col, row - 1)
                near = []
                # if not a boundary cell, add edge, else ignore
                if col > 0:
                    near.append((left_cell, cell))
                if row > 0:
                    near.append((down_cell, cell))
                edges_ordered.extend(near)

        # seed the random value
        random.seed(self.seed)
        edges = []
        # shuffle the ordered edges randomly into a new list
        while len(edges_ordered) > 0:
            # randomly pop an edge
            edges.append(edges_ordered.pop(
                random.randint(0, len(edges_ordered)) - 1))
        disjoint_set = DisjointSet()
        for row in range(0, self.height):
            for col in range(0, self.width):
                # the key is the cells unique id
                key = self.grid[col][row]
                # create the singleton
                disjoint_set.make_set(key)
                # intialize the keys portal dict
                self.portals[key] = {}
        edge_count = 0
        # eulers formula e = v-1, so the
        # minimum required edges is v for a connected graph!
        # each cell is identified by its key, and each key is a vertex on the
        # MST
        key_count = self.grid[self.width - 1][self.height - 1]  # last key
        while edge_count < key_count:
            # get next edge ((row1, col1), (row2,col2))
            edge = edges.pop()
            # get the sets for each vertex in the edge
            key_a = self.grid[edge[0][0]][edge[0][1]]
            key_b = self.grid[edge[1][0]][edge[1][1]]
            set_a = disjoint_set.find(key_a)
            set_b = disjoint_set.find(key_b)
            # if they are not in the same set they are not in the
            # same region in the maze
            if set_a != set_b:
                # add the portal between the cells,
                # graph is undirected and will search
                # [a][b] or [b][a]
                edge_count += 1
                self.portals[key_a][key_b] = True
                self.portals[key_b][key_a] = True
                disjoint_set.union(set_a, set_b)

    def move(self, direction):
        '''
        Used to indicate of the player has completed the maze
        @params
            direction((int, int)) : Direction to move player

        @return
            None
        '''
        assert(direction in [self.LEFT, self.RIGHT, self.UP, self.DOWN])
        # if new move is the same as last move pop from path onto player
        new_move = (self.player[0] + direction[0],
                    self.player[1] + direction[1])
        valid = False
        # if new move is not within grid
        if new_move[0] < 0 or new_move[0] >= self.width or\
                new_move[1] < 0 or new_move[1] >= self.height:
            return valid
        player_key = self.width * self.player[1] + self.player[0]
        move_key = self.width * new_move[1] + new_move[0]
        # if theres a portal between player and newmove
        if move_key in self.portals[player_key]:
            self.is_moving = True
            #'\033[%d;%dH' % (y x)# move cursor to y, x
            head = '\033[%d;%dH' % (
                new_move[1] * 2 + 2, new_move[0] * 2 + 2) + self.head
            # uncolor edge between (edge is between newmove and player)
            edge = '\033[%d;%dH' % (self.player[1] * 2 + (new_move[1] - self.player[1]) + 2,
                                    self.player[0] * 2 + (new_move[0] - self.player[0]) + 2)
            tail = '\033[%d;%dH' % (
                self.player[1] * 2 + 2, self.player[0] * 2 + 2)
            end = '\033[%d;%dH' % ((self.height) * 2 + 2, 0) + self.empty
            # if new move is backtracking to last move then sets player pos to
            # top of path and remove path top
            if len(self.path) > 0 and new_move == self.path[-1]:
                # move cursor to player and color tail, move cursor to player
                # and color empty
                self.player = self.path.pop()
                # move cursor to player and color tail, move cursor to player and color empty
                # uncolor edge between and remove tail
                edge += self.empty
                tail += self.empty
                valid = False  # moved back
            # else move progresses path, draws forward and adds move to path
            else:
                self.path.append(self.player)
                self.player = new_move
                # move cursor to position to draw if ANSI
                # color edge between and color tail
                edge += self.tail
                tail += self.tail
                valid = True  # successfully moved forward between portals

            # use write and flush to ensure buffer is emptied completely to
            # avoid flicker
            sys.stdout.write(head + edge + tail + end)
            sys.stdout.flush()
            self.is_moving = False
        return valid

    def solve(self, position=(0, 0)):
        ''' Uses backtracking to solve maze'''
        if self.is_done():
            return True
        for direction in [self.LEFT, self.RIGHT, self.UP, self.DOWN]:
                # try a move, move will return false if no portal of backward
                # progress
            if self.move(direction):
                    # after move, set new test position to be current player
                    # position
                if self.solve(self.player):
                    return True
            # if position changed
            if position != self.player:
                # move back from towards previos position
                self.move((position[0] - self.player[0],
                           position[1] - self.player[1]))

        return False

    def heuristic_solve(self, position=(0, 0), depth=0, lookahead=10):
        ''' Use backtracking with iterative deepening to solve maze with a distance or randomized choice heuristic'''
        if self.is_done():
            return True
        if depth > 0:
            directions = [self.LEFT, self.RIGHT, self.UP, self.DOWN]
            # sort by distance towards the end dist 0 is closest so ascending order
            # heuristic
            directions.sort(
                # get manhatten distance
                #key=lambda direction: (self.width-self.player[0]+direction[0]-1+self.height-self.player[1]+direction[1]-1)/2.0
                # random
                key=lambda direction: random.random()
            )
            for direction in directions:
                # try a move, move will return false if no portal of backward
                # progress
                if self.move(direction):
                    # after move, set new test position to be current player
                    # position
                    if self.heuristic_solve(self.player, depth - 1, lookahead):
                        return True
                # if position changed
                if position != self.player:
                    # move back from towards previos position
                    self.move((position[0] - self.player[0],
                               position[1] - self.player[1]))
            return False
        else:
            return self.heuristic_solve(self.player, lookahead, lookahead + 1)

    def start_timer(self):
        self.is_moving = False
        self.timer_thread = threading.Thread(target=self.timer_job)
        self.timer_thread.start()

    def kill_timer(self):
        self.player = (self.width - 1, self.height - 1)
        if self.timer_thread != None:
            self.timer_thread.join()

    def end_timer(self):
        self.kill_timer()
        return self.time_taken

    def timer_job(self):
        start_time = time.time()
        # your code
        # prints the current time at the bottom of the maze
        while not self.is_done():
            # if not currently writing move, print time at bottom
            if not self.is_moving:
                time_elapsed = time.time() - start_time
                # delay on the update rate (only update every 10th of a second)
                if time_elapsed - self.time_taken > 0.01:
                    self.time_taken = time_elapsed
                    # use write and flush to ensure buffer is emptied
                    # completely to avoid flicker
                    sys.stdout.write('\033[%d;%dHTime:%.2f' % (
                        self.height * 2 + 2, 0, self.time_taken))
                    sys.stdout.flush()

        self.time_taken = time.time() - start_time

    def is_done(self):
        '''
        Used to indicate of the player has completed the maze
        @params
            None 
        @return
            True if player has reached the end
        '''
        return self.player == (self.width - 1, self.height - 1)


class EnvCleaner(object):
    def __init__(self, N_agent, map_size, seed):
        self.map_size = map_size
        self.seed = seed
        self.occupancy = self.generate_maze(seed)
        self.N_agent = N_agent
        self.agt_pos_list = []
        for i in range(self.N_agent):
            self.agt_pos_list.append([1, 1])

    def generate_maze(self, seed):
        symbols = {
            # default symbols
            'start': 'S',
            'end': 'X',
            'wall_v': '|',
            'wall_h': '-',
            'wall_c': '+',
            'head': '#',
            'tail': 'o',
            'empty': ' '
        }
        maze_obj = Maze(int((self.map_size - 1) / 2),
                        int((self.map_size - 1) / 2), seed, symbols, 1)
        grid_map = maze_obj.to_np()
        for i in range(self.map_size):
            for j in range(self.map_size):
                if grid_map[i][j] == 0:
                    grid_map[i][j] = 2
        return grid_map

    def step(self, action_list):
        reward = 0
        for i in range(len(action_list)):
            if action_list[i] == 0:     # up
                # if can move
                if self.occupancy[self.agt_pos_list[i][0] - 1][self.agt_pos_list[i][1]] != 1:
                    self.agt_pos_list[i][0] = self.agt_pos_list[i][0] - 1
            if action_list[i] == 1:     # down
                # if can move
                if self.occupancy[self.agt_pos_list[i][0] + 1][self.agt_pos_list[i][1]] != 1:
                    self.agt_pos_list[i][0] = self.agt_pos_list[i][0] + 1
            if action_list[i] == 2:     # left
                # if can move
                if self.occupancy[self.agt_pos_list[i][0]][self.agt_pos_list[i][1] - 1] != 1:
                    self.agt_pos_list[i][1] = self.agt_pos_list[i][1] - 1
            if action_list[i] == 3:     # right
                # if can move
                if self.occupancy[self.agt_pos_list[i][0]][self.agt_pos_list[i][1] + 1] != 1:
                    self.agt_pos_list[i][1] = self.agt_pos_list[i][1] + 1
            # if the spot is dirty
            if self.occupancy[self.agt_pos_list[i][0]][self.agt_pos_list[i][1]] == 2:
                self.occupancy[self.agt_pos_list[i]
                               [0]][self.agt_pos_list[i][1]] = 0
                reward = reward + 1
        return reward

    def get_global_obs(self):
        done = True
        obs = np.zeros((self.map_size, self.map_size, 3))
        for i in range(self.map_size):
            for j in range(self.map_size):
                if self.occupancy[i, j] == 0:
                    obs[i, j, 0] = 1.0
                    obs[i, j, 1] = 1.0
                    obs[i, j, 2] = 1.0
                if self.occupancy[i, j] == 2:
                    obs[i, j, 0] = 0.0
                    obs[i, j, 1] = 1.0
                    obs[i, j, 2] = 0.0
                    done = False
        for i in range(self.N_agent):
            obs[self.agt_pos_list[i][0], self.agt_pos_list[i][1], 0] = 1.0
            obs[self.agt_pos_list[i][0], self.agt_pos_list[i][1], 1] = 0.0
            obs[self.agt_pos_list[i][0], self.agt_pos_list[i][1], 2] = 0.0
        return obs, done

    def reset(self):
        self.occupancy = self.generate_maze(self.seed)
        self.agt_pos_list = []
        for i in range(self.N_agent):
            self.agt_pos_list.append([1, 1])

    def render(self):
        obs, _ = self.get_global_obs()
        enlarge = 5
        new_obs = np.ones(
            (self.map_size * enlarge, self.map_size * enlarge, 3))
        for i in range(self.map_size):
            for j in range(self.map_size):
                if obs[i][j][0] == 0.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 0.0:
                    cv2.rectangle(new_obs, (i * enlarge, j * enlarge), (i *
                                                                        enlarge + enlarge, j * enlarge + enlarge), (0, 0, 0), -1)
                if obs[i][j][0] == 1.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 0.0:
                    cv2.rectangle(new_obs, (i * enlarge, j * enlarge), (i *
                                                                        enlarge + enlarge, j * enlarge + enlarge), (0, 0, 255), -1)
                if obs[i][j][0] == 0.0 and obs[i][j][1] == 1.0 and obs[i][j][2] == 0.0:
                    cv2.rectangle(new_obs, (i * enlarge, j * enlarge), (i *
                                                                        enlarge + enlarge, j * enlarge + enlarge), (0, 255, 0), -1)
        cv2.imshow('image', new_obs)
        cv2.waitKey(10)


class clean_env2(gym.Env):
    def __init__(self, agent=2, max_iter=1000, shape=None):
        """
        Args:
        """
        self.agent = agent
        self.max_iter = max_iter
        self.shape = shape

        self.env = EnvCleaner(agent, shape[0], 0)

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(shape[0] * shape[1] * shape[2],))

        self.update_flag = False

    def get_obs_state(self):
        return self.observation_space

    def get_action_state(self):
        return self.action_space

    def get_object(self):
        return self

    def set_update_flag(self, v):
        self.update_flag = v

    def get_update_flag(self):
        return self.update_flag

    def reset(self):
        self.reward_total = 0
        self.agt_pos_old = self.env.agt_pos_list
        self.env = EnvCleaner(self.agent, self.shape[0], 0)
        state_list = []
        self.step_idx = 0
        self.done = False
        for i in range(self.agent):
            state_list.append(
                self.env.agt_pos_list[i][0] + self.shape[0] * self.env.agt_pos_list[i][1])
        self.state, done = self.env.get_global_obs()
        # state = state.flatten()
        return self.state

    def render(self):
        self.env.render()

    def step(self, action):
        for i in range(self.agent):
            assert self.action_space.contains(action[i]), 'invalid action'

        self.reward = self.env.step(action)
        if self.reward > 0.9:
            self.reward_total += self.reward
            self.reward += self.reward_total * 0.1
        #
        self.reward -= 0.1
        #

        for idx, pos_old in enumerate(self.agt_pos_old):
            if pos_old == self.env.agt_pos_list[idx]:
                self.reward -= 0.1

        self.step_idx += 1

        self.state, self.done = self.env.get_global_obs()
        if self.done:
            print('finished!')

        info = self.state
        # info = info.flatten()

        state_list = []
        for i in range(self.agent):
            state_list.append(
                self.env.agt_pos_list[i][0] + self.shape[0] * self.env.agt_pos_list[i][1])

        if self.max_iter < self.step_idx:
            self.done = True
            self.reward -= 10

        return self.state, self.reward, self.done, info


if __name__ == '__main__':
    env = clean_env2(agent=2, shape=(11, 11, 3), max_iter=4000)
    for i in range(10):
        s = env.reset()
        time_step = 0
        while True:
            env.render()
            time_step += 1
            a_list = []
            a = [env.action_space.sample() for i in range(env.env.N_agent)]
            ns, r, done, _ = env.step(a)
            # print(s, a, r, ns, done)
            print(time_step)
            s = ns
            if done:
                break
