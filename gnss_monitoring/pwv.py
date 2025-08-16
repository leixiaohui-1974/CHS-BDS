import numpy as np
import matplotlib.pyplot as plt

class PWVEstimator:
    """
    Estimates Precipitable Water Vapor (PWV) from GNSS-derived Zenith Total Delay (ZTD).
    """

    def __init__(self, latitude, height):
        """
        Initializes the PWV Estimator.

        Args:
            latitude (float): Latitude of the GNSS station in degrees.
            height (float): Height of the GNSS station above sea level in meters.
        """
        self.latitude = np.deg2rad(latitude) # Convert to radians for calculation
        self.height = height
        self.timestamps = None
        self.ztd = None
        self.pressure = None
        self.temperature_c = None
        print("✅ PWVEstimator initialized.")

    def simulate_inputs(self, num_epochs=288, interval_minutes=5):
        """
        Simulates realistic input data for a 24-hour period.
        - ZTD: Zenith Total Delay, which would come from a PPP solution.
        - Pressure: Surface atmospheric pressure.
        - Temperature: Surface temperature.
        """
        print("--- Simulating Input Data (ZTD, Pressure, Temp) ---")
        self.timestamps = np.arange(num_epochs) * interval_minutes

        # Simulate ZTD (in meters) with a diurnal pattern and some noise
        # A base delay of 2.4m is typical.
        base_ztd = 2.4
        diurnal_variation_ztd = 0.05 * np.sin(2 * np.pi * self.timestamps / (24 * 60))
        noise_ztd = np.random.normal(0, 0.005, num_epochs)
        # Add a sudden increase to simulate a pre-cipitation event
        event_start = int(num_epochs * 0.6)
        event_duration = int(num_epochs * 0.2)
        event_magnitude = 0.08
        event_ztd = np.zeros(num_epochs)
        event_ztd[event_start:event_start+event_duration] = event_magnitude * np.sin(np.pi * np.arange(event_duration) / event_duration)
        self.ztd = base_ztd + diurnal_variation_ztd + noise_ztd + event_ztd
        print(f"✅ Simulated ZTD values (min: {self.ztd.min():.3f}m, max: {self.ztd.max():.3f}m)")

        # Simulate surface pressure (in hPa) with some noise
        base_pressure = 1010
        noise_pressure = np.random.normal(0, 0.5, num_epochs)
        self.pressure = base_pressure + noise_pressure
        print(f"✅ Simulated Pressure values (min: {self.pressure.min():.1f} hPa, max: {self.pressure.max():.1f} hPa)")

        # Simulate surface temperature (in Celsius) with a diurnal pattern
        base_temp = 15
        diurnal_variation_temp = 8 * np.sin(2 * np.pi * (self.timestamps - 6*60) / (24 * 60)) # Peak in afternoon
        self.temperature_c = base_temp + diurnal_variation_temp + np.random.normal(0, 0.5, num_epochs)
        print(f"✅ Simulated Temperature values (min: {self.temperature_c.min():.1f}°C, max: {self.temperature_c.max():.1f}°C)")

        return self.timestamps, self.ztd, self.pressure, self.temperature_c


        return self.timestamps, self.ztd, self.pressure, self.temperature_c

    def calculate_zhd(self):
        """
        Calculates the Zenith Hydrostatic Delay (ZHD) using the Saastamoinen model.
        ZHD is returned in meters.
        """
        if self.pressure is None:
            raise ValueError("Pressure data is not available. Run simulate_inputs() first.")

        print("--- Calculating Zenith Hydrostatic Delay (ZHD) ---")
        # Saastamoinen ZHD model
        f_lat_h = 1 - 0.00266 * np.cos(2 * self.latitude) - 0.00028 * (self.height / 1000.0)
        self.zhd = (0.0022768 * self.pressure) / f_lat_h
        print(f"✅ Calculated ZHD values (mean: {np.mean(self.zhd):.3f}m)")
        return self.zhd

    def calculate_pwv(self):
        """
        Calculates the Precipitable Water Vapor (PWV) from ZTD and ZHD.
        PWV is returned in millimeters.
        """
        if self.ztd is None or self.zhd is None:
            raise ValueError("ZTD or ZHD not available. Run required calculations first.")

        print("--- Calculating Zenith Wet Delay (ZWD) and PWV ---")
        # Calculate Zenith Wet Delay (ZWD) in meters
        self.zwd = self.ztd - self.zhd
        print(f"✅ Calculated ZWD values (mean: {np.mean(self.zwd):.3f}m)")

        # Convert ZWD to PWV using the Bevis et al. (1992) formula
        # First, calculate the mean atmospheric temperature (Tm) from surface temperature (Ts)
        temp_kelvin = self.temperature_c + 273.15
        tm = 70.2 + 0.72 * temp_kelvin  # Tm in Kelvin, a common approximation

        # Atmospheric constants from Bevis et al., converted to SI units (Pascals)
        rho_w = 1000.0    # Density of liquid water (kg/m^3)
        Rv = 461.5        # Specific gas constant for water vapor (J/kg/K)
        # k-values are converted from hPa^-1 to Pa^-1 by dividing by 100
        k2_prime = 16.52 / 100  # K/Pa
        k3 = 3.776e5 / 100      # K^2/Pa

        # Calculate the dimensionless conversion factor Pi
        Pi = 10**6 / (rho_w * Rv * ((k3 / tm) + k2_prime))

        self.pwv = Pi * self.zwd * 1000 # Convert ZWD from m to mm for PWV
        print(f"✅ Calculated PWV values (mean: {np.mean(self.pwv):.2f}mm)")
        return self.pwv

    def plot_results(self):
        """
        Plots the simulated data and the final PWV results.
        """
        print("--- Plotting Results ---")
        fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

        # Plot 1: ZTD and ZHD
        axs[0].plot(self.timestamps / 60, self.ztd, 'b-', label='ZTD (Total)')
        axs[0].plot(self.timestamps / 60, self.zhd, 'g-', label='ZHD (Hydrostatic)')
        axs[0].set_ylabel('Delay (m)')
        axs[0].set_title('GNSS Zenith Delays')
        axs[0].legend()
        axs[0].grid(True)

        # Plot 2: ZWD
        axs[1].plot(self.timestamps / 60, self.zwd, 'c-', label='ZWD (Wet)')
        axs[1].set_ylabel('Delay (m)')
        axs[1].set_title('Zenith Wet Delay')
        axs[1].legend()
        axs[1].grid(True)

        # Plot 3: Temperature and Pressure
        ax3_twin = axs[2].twinx()
        p1, = axs[2].plot(self.timestamps / 60, self.temperature_c, 'r-', label='Temperature')
        p2, = ax3_twin.plot(self.timestamps / 60, self.pressure, 'm-', label='Pressure')
        axs[2].set_ylabel('Temperature (°C)', color='r')
        ax3_twin.set_ylabel('Pressure (hPa)', color='m')
        axs[2].set_xlabel('Time (hours)')
        axs[2].set_title('Surface Meteorological Data')
        axs[2].legend(handles=[p1, p2], loc='best')
        axs[2].grid(True)

        # Plot 4: Final PWV
        axs[3].plot(self.timestamps / 60, self.pwv, 'k-', label='PWV')
        axs[3].set_ylabel('PWV (mm)')
        axs[3].set_title('Estimated Precipitable Water Vapor')
        axs[3].set_xlabel('Time (hours)')
        axs[3].legend()
        axs[3].grid(True)

        plt.tight_layout()
        plt.savefig("pwv_estimation_results.png")
        print("✅ Results plotted and saved to 'pwv_estimation_results.png'")

if __name__ == '__main__':
    # --- Configuration ---
    STATION_LATITUDE = 34.0  # degrees
    STATION_HEIGHT = 150.0   # meters

    # Initialize the estimator
    estimator = PWVEstimator(latitude=STATION_LATITUDE, height=STATION_HEIGHT)

    # 1. Simulate input data
    estimator.simulate_inputs()

    # 2. Calculate Zenith Hydrostatic Delay (ZHD)
    estimator.calculate_zhd()

    # 3. Calculate PWV from ZWD
    estimator.calculate_pwv()

    # 4. Plot the results
    estimator.plot_results()

    print("\n--- PWV Estimation Complete ---")
