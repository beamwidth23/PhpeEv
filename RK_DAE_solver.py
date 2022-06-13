import numpy as np


class RK_DAE:

    def __init__(self, example, method, simulation_duration, step_size):
        self.name = method.name
        self.A = method.A
        self.b = method.b
        self.s = len(method.A)
        self.ex = example
        self.h = step_size
        self.ts = self._get_time_steps(simulation_duration, step_size)
        self.ys = self._initialize_array(example.y_0, len(self.ts))
        self.zs = self._initialize_array(example.z_0, len(self.ts))

    def solve(self):
        for i, _ in enumerate(self.ts[1:], 1):
            self.ys[i], self.zs[i] = self.find_next_y_z(self.ys[i-1], self.zs[i-1])
        return self

    def find_next_y_z(self, y, z):
        raise NotImplementedError

    @staticmethod
    def _initialize_array(initial_value, length):
        x = np.zeros((length, len(initial_value)))
        x[0] = initial_value
        return x

    @staticmethod
    def _get_time_steps(total_time, step_size):
        nr_of_steps = int(total_time / step_size) + 1
        return np.linspace(0, total_time, nr_of_steps)


class ERK_DAE1(RK_DAE):

    def find_next_y_z(self, y, z):

        Y = np.zeros((self.s, len(y)))
        k = np.zeros((self.s, len(y)))
        Y[0] = y
        for i in range(1, self.s):
            k[i-1] = self.ex.get_dy(Y[i-1])
            Y[i] = y + self.h * np.sum(self.A[i, j] * k[j] for j in range(i))

        k[-1] = self.ex.get_dy(Y[-1])
        new_y = y + self.h * np.sum((self.b[i] * k[i] for i in range(self.s)), axis=0)
        new_z = self.ex.get_z(new_y)
        return new_y, new_z


def __run_basic_example():
    from pendulum import Pendulum, DoublePendulum
    from methods import RK4

    p1 = Pendulum(m=5, x=1.5, y=-2, u=0, v=0)
    p2 = Pendulum(m=15, x=5.5, y=-5, u=0, v=0)
    dp = DoublePendulum(p1, p2)
    rk = ERK_DAE1(dp, RK4, 5, 0.1).solve()
    l1_start = (rk.ys[0, 0]**2 + rk.ys[0, 2]**2)**0.5
    l2_start = (rk.ys[0, 1]**2 + rk.ys[0, 3]**2)**0.5
    l1_end = (rk.ys[-1, 0]**2 + rk.ys[-1, 2]**2)**0.5
    l2_end = (rk.ys[-1, 1]**2 + rk.ys[-1, 3]**2)**0.5
    print(l1_start, l2_start)
    print(l1_end, l2_end)


if __name__ == '__main__':
    __run_basic_example()
