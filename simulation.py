import random

from pendulum import Pendulum, DoublePendulum
from RK_DAE_solver import ERK_DAE1
from methods import Euler, ExplicitMidpoint, RK4, DOPRI5

#각 막대의 질량중심은    위치다

def random_initial_values(p_mag, m_mag=5):

    m = round(random.random() * m_mag + 1, 2)
    x = round(random.random() * p_mag * random.choice([-1, 1]), 2)
    y = round((random.random() - 0.2) * p_mag, 2)
    return (m, x, y)


def create_random_example(p1_mag=4, p2_mag=6, m1_mag=5, m2_mag=5):

    p1_0 = random_initial_values(p1_mag, m1_mag)
    p2_0 = random_initial_values(p2_mag, m2_mag)

    p1 = Pendulum(*p1_0, u=0, v=0)
    p2 = Pendulum(*p2_0, u=0, v=0)
    dp = DoublePendulum(p1, p2)
    return dp


def create_perturbations(number, ex=None, amount=1e-6):

    if ex is None:
        ex = create_random_example()
    p1 = ex._b1
    p2 = ex._b2

    examples = []
    for n in range(number):
        p1_per = Pendulum(p1.m, (p1.x - n*amount), (p1.y + n*amount), p1.u, p1.v)
        p2_per = Pendulum(p2.m, (p2.x + n*amount), (p2.y - n*amount), p2.u, p2.v)
        examples.append(DoublePendulum(p1_per, p2_per))
    return examples


def simulate(example, method=RK4, duration=30, step_size=0.001):

    return ERK_DAE1(example, method, duration, step_size).solve()


def simulate_multiple_methods(example, methods, duration=30, step_size=0.001):

    return [simulate(example, method, duration, step_size) for method in methods]


def simulate_multiple_examples(examples, method=RK4, duration=30, step_size=0.001):
    return [simulate(ex, method, duration, step_size) for ex in examples]


def __run_basic_example():
    rsys = create_random_example()
    r1 = simulate(rsys, method=Euler, duration=15)
    print(r1.ys[:3])
    print(r1)

    exes = create_perturbations(5, amount=1e-3)
    for e in exes:
        print(e.y_0)
        print(e.z_0)

    mtd = [Euler, DOPRI5, ExplicitMidpoint]
    mms = simulate_multiple_methods(rsys, mtd, duration=10, step_size=0.01)
    for mm in mms:
        print(mm.name)
        print(mm.ys[-1])


if __name__ == '__main__':
    __run_basic_example()
