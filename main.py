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
    def poly(t, c_0, c_1, c_2, c_3):
        return c_0 + c_1 * t + c_2 * t**2 + c_3 * t**3

    @staticmethod
    def exp(a, b, t: T):
        return a * np.exp(b * t.t)

    @staticmethod
    def ln(a, t: T):
        return a * np.log(t.t+1)+1

    @staticmethod
    def sigmoid(t, c_0, c_1, k, t0):
        return c_0 + c_1 / (1 + np.exp(-k * (t - t0)))


def main():
    t = T(100, 1)

    v1 = Voltage(0.1, 2.0, t)
    v2 = Voltage(3.0, 5.0, t)

    resist_1 = np.zeros((11, len(t.t)))
    resist_2 = np.zeros((11, len(t.t)))

    i_1 = np.zeros((11, len(t.t)))
    i_2 = np.zeros((11, len(t.t)))

    c_0 = 1.0
    c_1_base = 0.02
    c_2 = np.linspace(-1e-5, 1e-5, 5)
    c_3 = 1e-6

    c_0 = 1
    c_1_base = 0.5
    k = 0.3
    # k_values = np.linspace(0.1, 0.5, 5)
    # t0 = 20.0
    t0_values = np.linspace(20, 50, 5)

    fig, ax1 = plt.subplots()
    ax2 = plt.twinx()
    v_normal = 1 + v1.potential / (v1.potential[-1]-v1.potential[0])
    for n, t0 in enumerate(t0_values):
        c_1 = c_1_base
        # resist_1[n] = Resistance.poly(t.t, c_0, c_1, c, c_3)
        resist_1[n] = Resistance.sigmoid(t.t, c_0, c_1, k, t0)
        
        i_1[n] = v1.potential/resist_1[n]
        i_normal = 1 + i_1[n] / (i_1[n][-1] - i_1[n][0])

        ax1.plot(v_normal, i_normal, label=f"k = {k:0.2f}", linestyle="solid")
        ax2.plot(v_normal, resist_1[n], label=f"r = {resist_1[n][-1]:0.2f}", linestyle="dashed")

    ax1.set_xlabel("Normalized volt")
    ax1.set_ylabel("Normalized Current (—)")
    # ax1.legend(loc='upper left')
    # ax2.legend(loc='upper right')
    ax2.set_ylabel("Normalized resistance (--)")
    plt.show()


if __name__ == "__main__":
    main()
