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
        c_0 = 1
        c_1 = 0.6  # final resistance is 1.6
        t_0 = 40
        k_values = np.linspace(0.01, 0.1, 5)
        t_final = t.t[-1]
        r_final = 1.6
        num_k = len(k_values)
        num_t = len(t.t)

        resist_1 = np.zeros((num_k, num_t))
        i = np.zeros((num_k, num_t))
        i_normal = np.zeros((num_k, num_t))

        for n, k in enumerate(k_values):
            resist_1[n] = Resistance.sigmoid(t.t, c_0, c_1, k, t_0)

            i[n] = v.potential/resist_1[n]

        return i, resist_1, k_values

    @staticmethod
    def tafel(v, i0, a):
        """
        Description
        ---
        Tafel equation: v = a * log10(i/i0)

        Parameters
        ---
        - v: overpotential  
        - i0: exchange current density
        - a: Tafel slope

        Returns
        - i: current density
        """
        return i0 * 10 ** (v/a)


def plot_current_resistance(v1: Voltage, i_normal_all, i_mineral_normal_all, i_tafel_normal):
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    fig1, ax1 = plt.subplots()
    for i, color in zip(i_normal_all, colors):
        ax1.plot(v1.potential*1000, i,  linestyle="solid", color=color)
    ax1.set_title("Overpotential vs Normalized Current")
    ax1.set_xlabel("overpatential (mV)")
    ax1.set_ylabel("Current (—)")

    fig2, ax2 = plt.subplots()
    for i_mineral, color in zip(i_mineral_normal_all, colors):
        ax2.plot(v1.potential*1000, i_mineral, linestyle="dashdot", color=color)
    ax2.set_title("Overpotential vs Mineral Current")
    ax2.set_xlabel("overpatential (mV)")
    ax2.set_ylabel("Current (—)")

    fig3, ax3 = plt.subplots()
    for i_tafel, color in zip(i_tafel_normal, colors):
        ax3.plot(v1.potential*1000, i_tafel,  linestyle="dotted", color=color)
    ax3.set_title("Overpotential vs electrochemical current")
    ax3.set_xlabel("overpatential (mV)")
    ax3.set_ylabel("Current (—)")
    plt.show()


def main():
    t = T(100, 1)
    v1 = Voltage(0.3, 1.0, t)
    i_all, resist_all, coefficients = Current.ohm(v1, t)
    i_all *= 0.0025  # adjust current to 10^-4mA/cm^2 order

    i_normal_all = np.zeros_like(i_all)
    i_tafel = Current.tafel(v1.potential, 1e-4, 1)
    
    i_tafel_normal = np.zeros_like(i_all)
    i_mineral_all = np.zeros_like(i_all)
    i_mineral_normal_all = np.zeros_like(i_all)
    for n, i in enumerate(i_all):
        i_mineral_all[n] = i - i_tafel
        i_normal_all[n] = (i - i[0]) / (i[-1] - i[0])
        i_mineral_normal_all[n] = (i_mineral_all[n] - i_mineral_all[n][0]) / (i[-1] - i[0])
        i_tafel_normal[n] = i_normal_all[n] - i_mineral_normal_all[n]

    colors = ['blue', 'orange', 'green', 'red', 'purple']

    fig1, ax1 = plt.subplots()
    ax3 = ax1.twinx()
    for i, r, k, color in zip(i_normal_all, resist_all, coefficients, colors):
        ax1.plot(v1.potential*1000, i, label=f"k = {k:0.2e}", linestyle="solid", color=color)
        ax3.plot(v1.potential*1000, r, label=f"r = {r[-1]:0.2f}", linestyle="dashed")
    ax1.set_title("Current and Resistance vs Overpotential")
    ax1.set_xlabel("overpatential (mV)")
    ax1.set_ylabel("Normalized Current (—)")
    ax1.legend(loc='upper left')
    ax3.set_ylabel("Normalized resistance (--)")

    plot_current_resistance(v1, i_normal_all, i_mineral_normal_all, i_tafel_normal)


if __name__ == "__main__":
    main()
