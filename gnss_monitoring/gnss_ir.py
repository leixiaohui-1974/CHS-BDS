import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend
from scipy.signal import lombscargle

class GNSSIRWaterLevel:
    """
    Calculates water level using GNSS-IR (Interference and Reflectometry).
    """

    def __init__(self, satellite_elevation, satellite_azimuth, antenna_height, wavelength):
        """
        Initializes the GNSS-IR water level calculator.

        Args:
            satellite_elevation (np.array): Satellite elevation angles (in degrees).
            satellite_azimuth (np.array): Satellite azimuth angles (in degrees).
            antenna_height (float): The known height of the antenna phase center above a reference point.
            wavelength (float): Wavelength of the GNSS signal (e.g., GPS L1 is approx. 0.1903m).
        """
        self.satellite_elevation = satellite_elevation
        self.satellite_azimuth = satellite_azimuth
        self.antenna_height = antenna_height
        self.wavelength = wavelength
        self.snr_data = None
        self.snr_residual = None
        self.reflector_height = None

    def _simulate_snr_data(self, true_reflector_height, noise_level=0.5, a=10, b=-20, c=30):
        """
        Simulates realistic SNR data for testing purposes.

        The model is SNR = A*sin(2*pi*f*sin(e)) + B*sin(e)^2 + C*sin(e) + D + noise
        where 'e' is the satellite elevation.

        Args:
            true_reflector_height (float): The "true" height to simulate.
            noise_level (float): The standard deviation of the Gaussian noise to add.
            a, b, c (float): Coefficients for the polynomial trend.
        """
        # Calculate the expected frequency of the multipath oscillations
        # f = 2 * h / lambda
        frequency = 2 * true_reflector_height / self.wavelength

        # sin(elevation) is the independent variable for the interference pattern
        sin_elevation = np.sin(np.deg2rad(self.satellite_elevation))

        # Generate the multipath signal (the sine wave)
        multipath_signal = a * np.sin(2 * np.pi * frequency * sin_elevation)

        # Generate the direct signal trend (a simple polynomial)
        direct_signal_trend = b * sin_elevation**2 + c * sin_elevation

        # Combine them and add some random noise
        self.snr_data = direct_signal_trend + multipath_signal + np.random.normal(0, noise_level, len(sin_elevation))
        print("✅ Simulated SNR data generated.")

    def preprocess_snr(self, detrend_type='linear'):
        """
        Preprocesses the SNR data to remove the direct signal trend.
        """
        if self.snr_data is None:
            raise ValueError("SNR data not loaded or simulated yet.")

        self.snr_residual = detrend(self.snr_data, type=detrend_type)
        print("✅ SNR data detrended.")

    def analyze_frequency(self, min_h=0.5, max_h=10.0, oversample_factor=5):
        """
        Analyzes the frequency of the SNR residual using Lomb-Scargle Periodogram.
        """
        if self.snr_residual is None:
            raise ValueError("SNR residual not calculated yet. Run preprocess_snr() first.")

        # The independent variable is sin(elevation)
        x = np.sin(np.deg2rad(self.satellite_elevation))
        y = self.snr_residual

        # The frequency in LSP corresponds to (2*h/lambda). We need to find h.
        # So we define a grid of frequencies to search over, corresponding to a grid of possible heights.
        f_min = 2 * min_h / self.wavelength
        f_max = 2 * max_h / self.wavelength

        # Setup the frequency grid for Lomb-Scargle
        freqs = np.linspace(f_min, f_max, int((f_max - f_min) * len(x) * oversample_factor))

        # Perform Lomb-Scargle Periodogram
        power = lombscargle(x, y, freqs * 2 * np.pi, normalize=True) # freqs must be in angular frequency

        # Find the frequency with the highest power
        dominant_freq = freqs[np.argmax(power)]

        # Convert frequency back to reflector height
        self.reflector_height = (dominant_freq * self.wavelength) / 2
        print(f"✅ Frequency analysis complete. Dominant frequency corresponds to height: {self.reflector_height:.2f} m")

        # For plotting
        self.lsp_freqs = freqs
        self.lsp_power = power


    def run_analysis(self, true_reflector_height, plot=True):
        """
        Runs the full analysis chain.
        """
        # 1. Simulate data
        self._simulate_snr_data(true_reflector_height=true_reflector_height)

        # 2. Preprocess
        self.preprocess_snr()

        # 3. Analyze
        self.analyze_frequency()

        if plot:
            self.plot_results()

        return self.reflector_height

    def plot_results(self):
        """
        Plots the results of the analysis.
        """
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))

        sin_elevation = np.sin(np.deg2rad(self.satellite_elevation))

        # Plot 1: Raw SNR data
        axs[0].plot(sin_elevation, self.snr_data, 'k.', markersize=3)
        axs[0].set_title('Simulated SNR Data vs. sin(Elevation)')
        axs[0].set_xlabel('sin(Satellite Elevation)')
        axs[0].set_ylabel('SNR (dB)')
        axs[0].grid(True)

        # Plot 2: Detrended SNR data
        axs[1].plot(sin_elevation, self.snr_residual, 'b.', markersize=3)
        axs[1].set_title('Detrended SNR Residual')
        axs[1].set_xlabel('sin(Satellite Elevation)')
        axs[1].set_ylabel('SNR Residual (dB)')
        axs[1].grid(True)

        # Plot 3: Lomb-Scargle Periodogram
        # Convert frequency axis back to reflector height for interpretability
        height_axis = (self.lsp_freqs * self.wavelength) / 2
        axs[2].plot(height_axis, self.lsp_power)
        axs[2].set_title('Lomb-Scargle Periodogram of SNR Residual')
        axs[2].set_xlabel('Reflector Height (m)')
        axs[2].set_ylabel('LSP Power')
        axs[2].axvline(x=self.reflector_height, color='r', linestyle='--', label=f'Detected Height: {self.reflector_height:.2f} m')
        axs[2].legend()
        axs[2].grid(True)

        plt.tight_layout()
        plt.savefig("gnss_ir_analysis_results.png")
        print("✅ Results plotted and saved to 'gnss_ir_analysis_results.png'")


if __name__ == '__main__':
    # --- Configuration ---
    # A typical satellite pass might go from 5 to 30 degrees in elevation
    ELEVATION_ANGLES = np.linspace(5, 30, 200) # degrees
    AZIMUTH_ANGLES = np.linspace(120, 150, 200) # Not used in this simplified model, but important for real data
    ANTENNA_HEIGHT = 1.5 # meters above a reference point
    WAVELENGTH = 0.1903 # meters (GPS L1 frequency)
    TRUE_REFLECTOR_HEIGHT = 4.5 # The "ground truth" height we want to recover

    print("--- Starting GNSS-IR Water Level Simulation ---")

    # Initialize the analyzer
    analyzer = GNSSIRWaterLevel(
        satellite_elevation=ELEVATION_ANGLES,
        satellite_azimuth=AZIMUTH_ANGLES,
        antenna_height=ANTENNA_HEIGHT,
        wavelength=WAVELENGTH
    )

    # Run the full analysis
    estimated_height = analyzer.run_analysis(true_reflector_height=TRUE_REFLECTOR_HEIGHT, plot=True)

    print(f"\n--- Results ---")
    print(f"Known (true) reflector height: {TRUE_REFLECTOR_HEIGHT:.2f} m")
    print(f"Estimated reflector height: {estimated_height:.2f} m")
    print(f"Difference: {abs(TRUE_REFLECTOR_HEIGHT - estimated_height):.3f} m")
    print("---------------------------------------------")
