from matplotlib import pyplot as plt
import numpy as np
import lmfit


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
    def poly(t, c_0, c_1, c_2):
        return c_0 + c_1 * t + c_2 * (t ** 2)

    @staticmethod
    def exp(t, c_0, c_1, k):
        return c_0 + c_1 * np.exp(k * t)

    @staticmethod
    def ln(t, c_0, c_1, c_2, k):
        return c_0 + c_1 * np.log(k*t + c_2)

    @staticmethod
    def sigmoid(t, c_0, c_1, k, t0):
        return c_0 + c_1 / (1 + np.exp(-k * (t - t0)))


def main():
    t = T(100, 1)

    v1 = Voltage(0.1, 2.0, t)
    resist_1 = np.zeros((11, len(t.t)))
    i_1 = np.zeros((11, len(t.t)))

    c_0 = 1
    c_2_values = np.linspace(5e-7, 5e-5, 5)
    t_final = t.t[-1]

    fig, ax1 = plt.subplots()
    ax2 = plt.twinx()
    v_normal = 1 + v1.potential / (v1.potential[-1]-v1.potential[0])
    for n, c_2 in enumerate(c_2_values):
        c_1 = (0.6 - c_2 * (t_final ** 2))/t_final
        resist_1[n] = Resistance.poly(t.t, c_0, c_1, c_2)

        i_1[n] = v1.potential/resist_1[n]
        i_normal = 1 + i_1[n] / (i_1[n][-1] - i_1[n][0])

        ax1.plot(v_normal, i_normal, label=f"c_2 = {c_2:0.2e}", linestyle="solid")
        ax2.plot(v_normal, resist_1[n], label=f"r = {resist_1[n][-1]:0.2f}", linestyle="dashed")

    ax1.set_xlabel("Normalized volt")
    ax1.set_ylabel("Normalized Current (—)")
    ax1.legend(loc='upper left')
    ax2.set_ylabel("Normalized resistance (--)")
    plt.show()


if __name__ == "__main__":
    main()
