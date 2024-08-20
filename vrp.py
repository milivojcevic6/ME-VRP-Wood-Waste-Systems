import math
import random
from copy import deepcopy

sh = 6


def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def distance_between_furthest(Set):
    max_dist = 0
    for i in range(len(Set)):
        for j in range(i + 1, len(Set)):
            dist = euclidean_distance(Set[i], Set[j])
            if dist > max_dist:
                max_dist = dist

    return max_dist


class VRP:
    def __init__(self, D, v, cap, P, M, R, P_cap_limit, M_cap_limit, w1=None, w2=None, k=None, first_echelon=True,
                 left_right=True, s_initial=None):
        self.first_echelon = first_echelon
        self.left_right = left_right
        self.v = v
        self.depot = D
        self.cap = cap
        self.rout = [[] for _ in range(v)]
        self.r_cap = [0 for _ in range(v)]
        self.r_cost = [0 for _ in range(v)]
        self.P_index = P[0][4]  # starting index of P in nodes list
        self.M_index = M[0][4] - len(P) if (
                (first_echelon and left_right) or (not first_echelon and not left_right)) else P[0][
            4]  # starting index of M in nodes list
        self.set_S = P + M  # all options for the satellite
        self.s_cap = []  # checking satellite capacities
        self.s = []  # assigned satellites
        self.w1 = w1 if w1 else 10.0
        self.w2 = w2 if w2 else 100.0
        self.k = k if k else v // 2  # hard-coded
        self.alpha = len(R) * distance_between_furthest(R)
        self.nodes = [self.depot] + P + M + R
        self.P_used = 0
        self.M_used = 0
        self.P_limit = P_cap_limit
        self.M_limit = M_cap_limit

        self.virtual_satellite = (self.set_S[0][0] + self.alpha, self.set_S[0][1] + self.alpha,
                                  float('Inf'), 'combined', -1)

        for i in range(len(self.set_S)):
            self.s_cap.append(self.set_S[i][2])

        if not s_initial:
            for i in range(self.v):
                self.s.append((self.set_S[i % len(self.set_S)]))
        else:
            for i in range(len(s_initial)):
                for j in range(math.ceil(s_initial[i] / self.cap)):
                    self.s.append(self.set_S[i])

            while len(self.s) < self.v:
                rndm = random.choice(P)
                self.s.append(rndm)

        original_v = self.v

        # Best fit R assignment
        for r in R:
            i = 0

            while i < self.v and (r[2] + self.r_cap[i] > self.cap or
                                  r[2] > self.s_cap[self.s[i][4] -
                                                    (self.M_index if self.s[i][3] == 'combined' else self.P_index)]
                                  or r[2] > (self.M_limit if self.s[i][3] == 'combined' else self.P_limit)):
                i += 1

            if (i < self.v and r[2] + self.r_cap[i] <= self.cap and
                    r[2] <= self.s_cap[self.s[i][4] - (self.M_index if self.s[i][3] == 'combined' else self.P_index)]
                    and r[2] <= (self.M_limit if self.s[i][3] == 'combined' else self.P_limit)):

                self.add_r_to_rout(r, i)

            else:
                self.add_to_new_veh(r)

        updated = True
        while updated and self.v > original_v:
            updated = False
            if [] in self.rout:
                updated = True

                i = 0
                while self.rout[i]:
                    i += 1

                self.v -= 1
                self.rout = self.rout[:i] + self.rout[i + 1:]
                self.r_cap = self.r_cap[:i] + self.r_cap[i + 1:]
                self.r_cost = self.r_cost[:i] + self.r_cost[i + 1:]
                self.s = self.s[:i] + self.s[i + 1:]

        # update sh if needed
        global sh

        if len(R) <= sh:
            sh = len(R)

    def add_to_new_veh(self, r):

        self.v += 1
        self.rout.append([])
        self.r_cap.append(0)
        self.r_cost.append(0)

        i = 0

        while (i < len(self.set_S) and (r[2] >
                                        self.s_cap[self.set_S[i][4] -
                                                   (self.M_index if
                                                   self.set_S[i][3] == 'combined'
                                                   else self.P_index)]
                                        or r[2] >
                                        (self.M_limit if
                                        self.set_S[i][3] == 'combined'
                                        else self.P_limit))):
            i += 1

        if i >= len(self.set_S):
            self.s.append(self.virtual_satellite)
        else:
            self.s.append(self.set_S[i])

        self.add_r_to_rout(r, self.v - 1)

    def add_r_to_rout(self, r, rout_i, position=None):

        if (rout_i < 0 or rout_i >= self.v or (position is not None and position > len(self.rout[rout_i]))
                or r[2] > (self.M_limit if self.s[rout_i][3] == 'combined' else self.P_limit)):
            return None

        if position is None:
            position = len(self.rout[rout_i])

        self.rout[rout_i].insert(position, r)
        self.r_cap[rout_i] += r[2]
        self.s_cap[self.s[rout_i][4] - (self.M_index if self.s[rout_i][3] == 'combined' else self.P_index)] -= r[2]

        if self.first_echelon and self.s[rout_i][3] == 'combined':
            self.M_used += r[2]
        elif self.first_echelon:
            self.P_used += r[2]
        elif self.s[rout_i][3] == 'combined':
            self.M_limit -= r[2]

        else:
            self.P_limit -= r[2]

        self.r_cost[rout_i] += self.fix_cost(position, rout_i)

    def remove_r_from_rout(self, r, rout_i, position):

        if rout_i < 0 or rout_i >= self.v or r not in self.rout[rout_i]:
            return None

        self.r_cost[rout_i] -= self.fix_cost(position, rout_i)
        self.rout[rout_i].pop(position)
        self.r_cap[rout_i] -= r[2]
        self.s_cap[self.s[rout_i][4] - (self.M_index if self.s[rout_i][3] == 'combined' else self.P_index)] += r[2]

        if self.first_echelon and self.s[rout_i][3] == 'combined':
            self.M_used -= r[2]
        elif self.first_echelon:
            self.P_used -= r[2]
        elif self.s[rout_i][3] == 'combined':
            self.M_limit += r[2]

        else:
            self.P_limit += r[2]

    def fix_cost(self, position, rout_i):

        difference = 0

        if rout_i < 0 or rout_i >= self.v or position < 0 or position >= len(self.rout[rout_i]):
            return None

        current = (self.rout[rout_i][position][0], self.rout[rout_i][position][1])

        if position > 0:

            before = (self.rout[rout_i][position - 1][0], self.rout[rout_i][position - 1][1])

        else:

            before = (self.depot[0], self.depot[1]) if self.first_echelon else (self.s[rout_i][0], self.s[rout_i][1])

        if position < len(self.rout[rout_i]) - 1:

            after = (self.rout[rout_i][position + 1][0], self.rout[rout_i][position + 1][1])

        else:

            after = (self.depot[0], self.depot[1]) if not self.first_echelon else (self.s[rout_i][0], self.s[rout_i][1])

        difference += euclidean_distance(before, current)

        difference += euclidean_distance(current, after)

        if len(self.rout[rout_i]) > 1:
            difference -= euclidean_distance(before, after)

        else:
            x = (self.depot[0], self.depot[1]) if not self.first_echelon else (self.s[rout_i][0], self.s[rout_i][1])
            y = (self.depot[0], self.depot[1]) if self.first_echelon else (self.s[rout_i][0], self.s[rout_i][1])

            difference += euclidean_distance(x, y)

        return difference

    def swap_s_cost(self, k, i):

        if len(self.rout[k]) == 0:
            return

        difference = 0

        if self.first_echelon:

            before = (self.rout[k][-1][0], self.rout[k][-1][1])
            after = (self.depot[0], self.depot[1])

        else:

            before = (self.depot[0], self.depot[1])
            after = (self.rout[k][0][0], self.rout[k][0][1])

        new = (self.set_S[i][0], self.set_S[i][1])
        old = (self.s[k][0], self.s[k][1])

        difference += euclidean_distance(before, new)
        difference += euclidean_distance(new, after)

        difference -= euclidean_distance(before, old)
        difference -= euclidean_distance(old, after)

        self.r_cost[k] += difference

    def cost(self):

        distance = 0
        num_vehicles_used = 0

        for i in range(self.v):
            distance += self.r_cost[i] if len(self.rout[i]) > 0 else 0
            num_vehicles_used += 1 if len(self.rout[i]) > 0 else 0

        total_cost = self.w1 * distance + self.w2 * num_vehicles_used + self.alpha * max(0, num_vehicles_used - self.k)

        # Add extra cost if capacities are not distributed properly
        if not self.first_echelon and (self.M_limit != 0 or self.P_limit != 0):
            total_cost += self.alpha ** 5

        return total_cost

    def swap_s(self, k, i):

        if (k >= self.v or k < 0 or i < 0 or i >= len(self.set_S) or
                self.r_cap[k] > self.s_cap[i] or
                (self.s[k][3] != self.set_S[i][3] and
                 self.r_cap[k] > (self.M_limit if self.set_S[i][3] == 'combined' else self.P_limit))):

            return None

        self.swap_s_cost(k, i)

        self.s_cap[i] -= self.r_cap[k]

        self.s_cap[self.s[k][4] - (self.M_index if self.s[k][3] == 'combined' else self.P_index)] += self.r_cap[k]

        if self.first_echelon and self.s[k][3] == 'combined':
            self.M_used -= self.r_cap[k]
        elif self.first_echelon:
            self.P_used -= self.r_cap[k]
        elif self.s[k][3] == 'combined':
            self.M_limit += self.r_cap[k]
        else:
            self.P_limit += self.r_cap[k]

        if self.first_echelon and self.set_S[i][3] == 'combined':
            self.M_used += self.r_cap[k]
        elif self.first_echelon:
            self.P_used += self.r_cap[k]
        elif self.set_S[i][3] == 'combined':
            self.M_limit -= self.r_cap[k]
        else:
            self.P_limit -= self.r_cap[k]

        self.s[k] = self.set_S[i]

        return self

    def swap_r2(self, i1, j1, i2, j2):

        if (i1 < 0 or i1 >= len(self.rout) or i2 < 0 or i2 >= len(self.rout) or
                j1 < 0 or j1 >= len(self.rout[i1]) or j2 < 0 or j2 >= len(self.rout[i2])):
            return None

        element1 = self.rout[i1][j1]
        element2 = self.rout[i2][j2]

        current_v_load1 = self.r_cap[i1]
        current_v_load2 = self.r_cap[i2]

        sat1_limit = self.s_cap[self.s[i1][4] - (self.M_index if self.s[i1][3] == 'combined' else self.P_index)]
        sat2_limit = self.s_cap[self.s[i2][4] - (self.M_index if self.s[i2][3] == 'combined' else self.P_index)]

        if (element1[2] + current_v_load2 - element2[2] < 0 or
                element1[2] + current_v_load2 - element2[2] > self.cap or
                element2[2] + current_v_load1 - element1[2] < 0 or
                element2[2] + current_v_load1 - element1[2] > self.cap or
                element1[2] - element2[2] > sat2_limit or
                element2[2] - element1[2] > sat1_limit or
                (self.M_limit if self.s[i1][3] == 'combined' else self.P_limit) + element1[2] - element2[2] < 0 or
                (self.M_limit if self.s[i2][3] == 'combined' else self.P_limit) + element2[2] - element1[2] < 0):
            return None

        self.remove_r_from_rout(element1, i1, j1)

        self.add_r_to_rout(element2, i1, j1)

        self.remove_r_from_rout(element2, i2, j2)

        self.add_r_to_rout(element1, i2, j2)

        return self

    # Insert the Ri element from position (i1, j1) into position (i2, j2) and remove it from (i1, j1).
    def insert(self, i1, j1, i2, j2):

        if (i1 < 0 or i1 >= len(self.rout) or i2 < 0 or i2 >= len(self.rout) or
                j1 < 0 or j1 >= len(self.rout[i1]) or j2 < 0 or j2 > len(self.rout[i2]) or
                (i1 != i2 and
                 (self.rout[i1][j1][2] > self.s_cap[self.s[i2][4] -
                                                    (self.M_index if self.s[i2][3] == 'combined' else self.P_index)] or
                  self.rout[i1][j1][2] + self.r_cap[i2] > self.cap))):
            return None

        # Decrease index j2 if j1 is before it in the same rout
        if i1 == i2 and j1 < j2:
            j2 -= 1

        element = self.rout[i1][j1]
        if element[2] > (self.M_limit if self.s[i2][3] == 'combined' else self.P_limit):
            return None

        self.remove_r_from_rout(element, i1, j1)
        self.add_r_to_rout(element, i2, j2)

        return self

    # Reinserting removed objects RANDOMly
    def reinsert(self, removed_objects):

        for obj in removed_objects:
            added = False
            tried = []

            while not added:

                vehicle_index = random.choice(list(range(len(self.v))))

                if (self.r_cap[vehicle_index] + obj[2] <= self.cap and obj[2] <= self.s_cap[
                    self.s[vehicle_index][4] - (self.M_index if self.s[vehicle_index][
                                                                    3] == 'combined' else self.P_index)] or
                        obj[2] <= (self.M_limit if self.s[vehicle_index][3] == 'combined' else self.P_limit)):

                    index = random.choice(list(range(len(self.rout[vehicle_index]) + 1)))
                    self.add_r_to_rout(obj, vehicle_index, index)
                    added = True

                elif vehicle_index not in tried:
                    tried.append(vehicle_index)

                    if len(tried) == self.v:
                        self.add_to_new_veh(obj)
                        added = True

    # Shake operation
    def shake(self):

        # Initialize list to store the top sh differences and their indices
        max_diff = [float('-inf')] * sh
        max_diff_index = [None] * sh
        removed_objects = []

        for i in range(len(self.rout)):
            current_cost = self.r_cost[i]
            for j in range(len(self.rout[i])):
                without = deepcopy(self)
                without.remove_r_from_rout(without.rout[i][j], i, j)
                difference = current_cost - without.cost()

                # Check where this difference fits in the top sh list
                for k in range(sh):
                    if difference > max_diff[k]:
                        # Shift down the existing values
                        max_diff[k + 1:sh] = max_diff[k:sh - 1]
                        max_diff_index[k + 1:sh] = max_diff_index[k:sh - 1]

                        # Insert the new difference and its index
                        max_diff[k] = difference
                        max_diff_index[k] = (i, j)
                        break

        for obj in sorted(removed_objects, reverse=True):
            if obj is not None:
                removed_objects.append(self.rout[obj[0]][obj[1]])
                self.remove_r_from_rout(self.rout[obj[0]][obj[1]], [obj[0]], [obj[1]])

        # reinsert them randomly
        self.reinsert(removed_objects)

        return self

    # update parameters from the neighbor
    def update_from_neighbor(self, neighbor):
        self.v = neighbor.v
        self.depot = neighbor.depot
        self.s = neighbor.s
        self.s_cap = neighbor.s_cap
        self.rout = neighbor.rout
        self.r_cap = neighbor.r_cap
        self.r_cost = neighbor.r_cost
        self.P_limit = neighbor.P_limit
        self.M_limit = neighbor.M_limit
        self.P_used = neighbor.P_used
        self.M_used = neighbor.M_used

    # improve function is running everything (:
    def improve(self):

        updated = True
        best_neighbor = deepcopy(self)

        while updated:

            updated = False

            current_cost = self.cost()

            # try to swap all R(i1,j1)-R(i2,j2) pairs
            for i1 in range(self.v):
                if self.rout[i1]:
                    for j1 in range(len(self.rout[i1])):
                        for i2 in range(i1, self.v):
                            if self.rout[i2]:
                                for j2 in range(len(self.rout[i2])):
                                    if i1 == i2 and j1 <= j2:
                                        continue
                                    neighbor = deepcopy(self).swap_r2(i1, j1, i2, j2)
                                    if neighbor is not None and neighbor.cost() < current_cost:
                                        current_cost = neighbor.cost()
                                        best_neighbor.update_from_neighbor(neighbor)
                                        updated = True

            # swap Si element
            for k in range(self.v):
                if self.rout[k]:
                    cur_s = self.s[k]
                    for i in range(len(self.set_S)):
                        if cur_s != self.set_S[i]:
                            neighbor = deepcopy(self)
                            neighbor.swap_s(k, i)
                            if (neighbor is not None) and neighbor.cost() < current_cost:
                                current_cost = neighbor.cost()
                                best_neighbor.update_from_neighbor(neighbor)
                                updated = True

            # insert Ri on some other position
            for i1 in range(self.v):
                if self.rout[i1]:
                    for j1 in range(len(self.rout[i1])):
                        for i2 in range(self.v):
                            for j2 in range(len(self.rout[i1]) + 1):
                                if i1 == i2 and j1 == j2:
                                    continue
                                neighbor = deepcopy(self).insert(i1, j1, i2, j2)
                                if neighbor is not None and neighbor.cost() < current_cost:
                                    current_cost = neighbor.cost()
                                    best_neighbor.update_from_neighbor(neighbor)
                                    updated = True

            neighbor = deepcopy(self).shake()
            if neighbor is not None and neighbor.cost() < current_cost:
                best_neighbor.update_from_neighbor(neighbor)
                updated = True

            # check if improved
            if updated:
                self.update_from_neighbor(best_neighbor)

    #
    def get_edges(self):

        edges = []

        leftVRP = (self.left_right and self.first_echelon) or (not self.left_right and not self.first_echelon)

        for veh in range(self.v):
            for i in range(1, len(self.rout[veh])):
                edges.append((self.rout[veh][i - 1][4], self.rout[veh][i][4], veh, leftVRP))

            if self.rout[veh] and leftVRP:
                edges.append((0, self.rout[veh][0][4], veh, True))
                edges.append((self.rout[veh][-1][4], self.s[veh][4], veh, True))
                edges.append((self.s[veh][4], 0, veh, True))

            elif self.rout[veh]:
                edges.append((self.rout[veh][-1][4], 1, veh, False))
                edges.append((self.s[veh][4], self.rout[veh][0][4], veh, False))
                edges.append((1, self.s[veh][4], veh, False))

        return edges
