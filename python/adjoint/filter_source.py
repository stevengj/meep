import numpy as np
from scipy import signal, linalg
from meep import CustomSource


class FilteredSource(CustomSource):
    def __init__(self,
                 center_frequency,
                 frequencies,
                 frequency_response,
                 dt,
                 time_src=None):
        dt = dt / 2  # divide by two to compensate for staggered E,H time interval
        self.dt = dt
        self.frequencies = frequencies
        self.center_frequencies = frequencies

        # For now, the basis functions cannot overlap much in the frequency domain. Otherwise, the
        # resulting nodes are wildly large and induce numerical precision errors. We can always
        # produce a safe simulation by forcing the length of each basis function to meet the minimum
        # frequency requirements. This method still minimizes storage requirements.
        self.T = np.max(np.abs(1 / np.diff(frequencies)))
        self.N = np.rint(self.T / self.dt)
        self.t = np.arange(0, dt * (self.N), dt)
        self.n = np.arange(self.N)
        f = self.func()
        self.bf = [
            lambda t, i=i: 0
            if t > self.T else (self.nuttall(t, self.center_frequencies) /
                                (self.dt / np.sqrt(2 * np.pi)))[i]
            for i in range(len(self.center_frequencies))
        ]
        self.time_src_bf = [
            CustomSource(src_func=bfi,
                         center_frequency=center_frequency,
                         is_integrated=False,
                         end_time=self.T) for bfi in self.bf
        ]

        if time_src:
            # get the cutoff of the input signal
            signal_t = np.array([
                time_src.swigobj.current(ti, dt) for ti in self.t
            ])  # time domain signal
            signal_dtft = self.dtft(signal_t, self.frequencies)
        else:
            signal_dtft = 1

        # multiply sampled dft of input signal with filter transfer function
        H = signal_dtft * frequency_response

        # estimate the impulse response using a sinc function RBN
        self.nodes, self.err = self.estimate_impulse_response(H)

        # initialize super
        super(FilteredSource, self).__init__(src_func=f,
                                             center_frequency=center_frequency,
                                             is_integrated=False,
                                             end_time=self.T)

    def cos_window_td(self, a, t, f0):
        cos_sum = np.sum([(-1)**k * a[k] * np.cos(2 * np.pi * t * k / self.T)
                          for k in range(len(a))],
                         axis=0)
        return np.exp(-1j * 2 * np.pi * f0 * t) * (cos_sum)

    def cos_window_fd(self, a, f, f0):
        df = 1 / (self.N * self.dt)
        cos_sum = a[0] * self.sinc(f, f0)
        for k in range(1, len(a)):
            cos_sum += (-1)**k * a[k] / 2 * self.sinc(f, f0 - k * df) + (
                -1)**k * a[k] / 2 * self.sinc(f, f0 + k * df)
        return cos_sum

    def sinc(self, f, f0):
        num = np.where(f == f0, self.N + 1,
                       (1 - np.exp(1j * (self.N + 1) * (2 * np.pi) *
                                   (f - f0) * self.dt)))
        den = np.where(f == f0, 1,
                       (1 - np.exp(1j * (2 * np.pi) * (f - f0) * self.dt)))
        return num / den

    def rect(self, t, f0):
        n = np.rint((t) / self.dt)
        return np.where(n.any() < 0.0 or n.any() > self.N, 0,
                        np.exp(-1j * 2 * np.pi * f0 * t))

    def hann(self, t, f0):
        a = [0.5, 0.5]
        return self.cos_window_td(a, t, f0)

    def hann_dtft(self, f, f0):
        a = [0.5, 0.5]
        return self.cos_window_fd(a, f, f0)

    def nuttall(self, t, f0):
        a = [0.355768, 0.4873960, 0.144232, 0.012604]
        return self.cos_window_td(a, t, f0)

    def nuttall_dtft(self, f, f0):
        a = [0.355768, 0.4873960, 0.144232, 0.012604]
        return self.cos_window_fd(a, f, f0)

    def dtft(self, y, f):
        return np.matmul(
            np.exp(1j * 2 * np.pi * f[:, np.newaxis] * np.arange(y.size) *
                   self.dt), y) * self.dt / np.sqrt(2 * np.pi)

    def __call__(self, t):
        if t > self.T:
            return 0
        vec = self.nuttall(t, self.center_frequencies) / (
            self.dt / np.sqrt(2 * np.pi))  # compensate for meep dtft
        return np.inner(vec, self.nodes)

    def func(self):
        def _f(t):
            return self(t)

        return _f

    def estimate_impulse_response(self, H):
        # Use vandermonde matrix to calculate weights of each gaussian. Each window is centered at each frequency point.
        # TODO: come up with a more sophisticated way to choose temporal window size and basis locations
        # that will minimize l2 estimation error and the node weights (since matrix is ill-conditioned)
        vandermonde = self.nuttall_dtft(self.frequencies[:, np.newaxis],
                                        self.center_frequencies[np.newaxis, :])
        nodes = np.matmul(linalg.pinv(vandermonde), H.T)
        H_hat = np.matmul(vandermonde, nodes)
        l2_err = np.sum(np.abs(H - H_hat.T)**2 / np.abs(H)**2)
        return nodes, l2_err
