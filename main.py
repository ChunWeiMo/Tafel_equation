from matplotlib import pyplot as plt
import numpy as np


class T:
    def __init__(self, duration=3600, delta_t=1):
        self.duration = duration
        self.delta_t = delta_t
        self.steps = int(self.duration/self.delta_t)
        self.t = np.linspace(0, self.duration, self.steps + 1)


class Voltage:
    def __init__(self, vmin, vmax, t: T):
        self.vmax = vmax
        self.vmin = vmin
        self.potential = None
        self.voltage(t)

    def voltage(self, t: T):
        self.potential = np.linspace(self.vmin, self.vmax, t.steps + 1)


class Resistance:
    @staticmethod
    def poly(a, b, c, t: T):
        return a * t.t**2.0+b*t.t+c

    @staticmethod
    def exp(a, b, t: T):
        return a * np.exp(b * t.t)

    @staticmethod
    def ln(a, t: T):
        return a * np.log(t.t+1)+1


def main():
    t = T(100, 1)

    v = Voltage(1, 2.5, t)

    resist = np.zeros((11, len(t.t)))
    i = np.zeros((11, len(t.t)))

    coefficient = np.linspace(0.10, 0.20, 5)  # ln
    # coefficient = np.linspace(0.001, 0.005, 5) # exp

    fig, ax1 = plt.subplots()
    ax2 = plt.twinx()
    for n, a in enumerate(coefficient):
        # resist[n] = Resistance.poly(0, a, 1, t)
        # resist[n] = Resistance.exp(0.994, a, t)
        resist[n] = Resistance.ln(a, t)
        i[n] = v.potential/resist[n]

        ax1.plot(v.potential, i[n], label=f"r = {resist[n][-1]}", linestyle="solid")
        ax2.plot(v.potential, resist[n], label=f"r = {resist[n][-1]}", linestyle="dashed")

    ax1.set_xlabel("Volts (%)")
    ax1.set_ylabel("Current (%)")
    ax2.legend(loc='upper right')
    ax2.set_ylabel("Resistance (%)")
    plt.show()


if __name__ == "__main__":
    main()
