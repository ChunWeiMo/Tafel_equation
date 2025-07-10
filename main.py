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


def tafel_equation(v, a, b):
    """
    Tafel equation: v = a + b * log10(i)
    where:
    - v is the overpotential
    - a is the intercept
    - b is the slope
    - i is the current density
    """
    return a + b * np.log10(v)


class Current:
    @staticmethod
    def ohm(v: Voltage, t: T):
        resist_1 = np.zeros((11, len(t.t)))
        i = np.zeros((11, len(t.t)))
        i_normal = np.zeros((11, len(t.t)))

        c_0 = 1
        c_2_values = np.linspace(5e-7, 5e-5, 5)
        t_final = t.t[-1]

        for n, c_2 in enumerate(c_2_values):
            c_1 = (0.6 - c_2 * (t_final ** 2))/t_final
            resist_1[n] = Resistance.poly(t.t, c_0, c_1, c_2)

            i[n] = v.potential/resist_1[n]
            i_normal[n] = 1 + i[n] / (i[n][-1] - i[n][0])

        return i_normal, resist_1, c_2_values


def main():
    t = T(100, 1)
    v1 = Voltage(0.3, 1.0, t)
    i_normal_all, resist_all, coefficients = Current.ohm(v1, t)

    fig, ax1 = plt.subplots()
    ax2 = plt.twinx()

    for i, r, k in zip(i_normal_all, resist_all, coefficients):
        ax1.plot(v1.potential*1000, i, label=f"c_2 = {k:0.2e}", linestyle="solid")
        ax2.plot(v1.potential*1000, r, label=f"r = {r[-1]:0.2f}", linestyle="dashed")

    ax1.set_xlabel("overpatential (mV)")
    ax1.set_ylabel("Normalized Current (—)")
    ax1.legend(loc='upper left')
    ax2.set_ylabel("Normalized resistance (--)")
    plt.show()


if __name__ == "__main__":
    main()
