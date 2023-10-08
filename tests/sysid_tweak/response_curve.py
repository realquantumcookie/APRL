import numpy as np
import matplotlib.pyplot as plt

joint_ranges = {
    "abduction": (-1.047, 1.047),
    "hip": (-0.663, 2.966),
    "knee": (-2.721, -0.837),
}

def square_wave(
    duration: float,
    freq: float,
    period: float,
    off_value: float,
    on_value: float,
) -> np.ndarray:
    dt = 1.0 / freq
    on_time = period / 2
    on_steps = int(on_time / dt)
    t = np.linspace(0, duration, int(duration / dt))
    y = np.ones(len(t)) * off_value
    for i in range(len(t)):
        if i % (2 * on_steps) >= on_steps:
            y[i] = on_value
    return y


duration = 5  # seconds
freq = 500 # Hz
period = 2 # seconds

t = np.linspace(0, duration, int(duration * freq))
y_abduction = square_wave(duration, freq, period, joint_ranges["abduction"][0] + 0.1, joint_ranges["abduction"][1] - 0.1)
y_hip = square_wave(duration, freq, period, joint_ranges["hip"][0] + 0.2, joint_ranges["hip"][1] - 0.2)
y_knee = square_wave(duration, freq, period, joint_ranges["knee"][0] + 0.2, joint_ranges["knee"][1] - 0.2)

# plt.plot(t, y_abduction, label="abduction")
# plt.plot(y_hip, label="hip")
# plt.plot(y_knee, label="knee")
# plt.legend()
# plt.show()