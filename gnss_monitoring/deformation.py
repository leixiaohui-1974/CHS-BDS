import numpy as np

class DeformationMonitor:
    """
    Performs high-precision deformation monitoring using GNSS double-differencing.
    """

    def __init__(self, ref_pos, mon_pos, satellite_positions):
        """
        Initializes the Deformation Monitor.

        Args:
            ref_pos (np.array): ECEF coordinates of the reference station [X, Y, Z].
            mon_pos (np.array): Approximate ECEF coordinates of the monitoring station [X, Y, Z].
            satellite_positions (dict): A dictionary where keys are satellite IDs and values
                                        are their ECEF coordinates [X, Y, Z] at a given epoch.
        """
        self.ref_pos = np.array(ref_pos)
        self.mon_pos = np.array(mon_pos)
        self.satellite_positions = satellite_positions
        self.wavelength = 0.1903  # GPS L1 wavelength in meters

        # True baseline vector (for simulation and verification)
        self.true_baseline = self.mon_pos - self.ref_pos

        print("DeformationMonitor initialized.")
        print(f"  - True Baseline (dX, dY, dZ): {self.true_baseline}")


    def _simulate_phase_observations(self):
        """
        Simulates carrier phase observations for both stations from all satellites.
        This is a simplified model for demonstration.
        """
        print("\n--- Simulating Observations ---")
        self.observations = {}

        # These represent common error sources that will be eliminated
        # in the double-differencing process.
        satellite_clock_error = {sat_id: np.random.uniform(0, 1e-6) for sat_id in self.satellite_positions}
        receiver_clock_error_ref = np.random.uniform(0, 5e-7)
        receiver_clock_error_mon = np.random.uniform(0, 6e-7)
        atmospheric_delay_ref = {sat_id: np.random.uniform(0, 5) for sat_id in self.satellite_positions}
        atmospheric_delay_mon = {sat_id: np.random.uniform(0, 5.1) for sat_id in self.satellite_positions} # Slightly different for monitor

        for station_name, station_pos in [('ref', self.ref_pos), ('mon', self.mon_pos)]:
            self.observations[station_name] = {}
            for sat_id, sat_pos in self.satellite_positions.items():
                # Calculate true geometric range
                true_range = np.linalg.norm(sat_pos - station_pos)

                # Phase observation = (range / wavelength) + clock errors + atmospheric delay + ambiguity + noise
                phase = (true_range / self.wavelength) \
                        - satellite_clock_error[sat_id] * (299792458 / self.wavelength) \
                        + receiver_clock_error_ref * (299792458 / self.wavelength) \
                        + atmospheric_delay_ref[sat_id] / self.wavelength \
                        + 1e7 # A large, constant integer ambiguity

                self.observations[station_name][sat_id] = phase
                print(f"  - Simulated phase for Station '{station_name}' to Satellite '{sat_id}': {phase:.3f} cycles")

        return self.observations

    def perform_double_differencing(self):
        """
        Performs double differencing on the phase observations.
        """
        print("\n--- Performing Double Differencing ---")
        self.double_diffs = []

        # Select a reference satellite (e.g., the first one in the list)
        sat_ids = list(self.satellite_positions.keys())
        ref_sat_id = sat_ids[0]
        print(f"  - Reference Satellite: {ref_sat_id}")

        # Single difference (between stations) for the reference satellite
        sd_ref_sat = self.observations['mon'][ref_sat_id] - self.observations['ref'][ref_sat_id]

        # Create double differences for all other satellites
        for i in range(1, len(sat_ids)):
            sat_id = sat_ids[i]

            # Single difference for the current satellite
            sd_sat = self.observations['mon'][sat_id] - self.observations['ref'][sat_id]

            # Double difference
            dd = sd_sat - sd_ref_sat
            self.double_diffs.append({'dd_value': dd, 'sat1': ref_sat_id, 'sat2': sat_id})
            print(f"  - DD ( {ref_sat_id} - {sat_id} ): {dd:.3f} cycles")

        return self.double_diffs

    def solve_baseline(self, approx_mon_pos):
        """
        Solves for the baseline vector using least-squares adjustment.

        Args:
            approx_mon_pos (np.array): An initial, approximate position for the monitoring station.
        """
        print("\n--- Solving for Baseline Vector ---")

        # Ensure inputs are NumPy arrays for vector operations
        approx_mon_pos = np.array(approx_mon_pos)
        approx_baseline = approx_mon_pos - self.ref_pos
        num_obs = len(self.double_diffs)

        # The design matrix H will have shape (num_obs, 3)
        H = np.zeros((num_obs, 3))
        # The observation vector b will have shape (num_obs,)
        b = np.zeros(num_obs)

        for i, dd_obs in enumerate(self.double_diffs):
            ref_sat_id = dd_obs['sat1']
            sat_id = dd_obs['sat2']

            # Get satellite positions and ensure they are NumPy arrays
            ref_sat_pos = np.array(self.satellite_positions[ref_sat_id])
            sat_pos = np.array(self.satellite_positions[sat_id])

            # Calculate line-of-sight (LOS) unit vectors from the approx monitor position
            los_ref_sat = (ref_sat_pos - approx_mon_pos) / np.linalg.norm(ref_sat_pos - approx_mon_pos)
            los_sat = (sat_pos - approx_mon_pos) / np.linalg.norm(sat_pos - approx_mon_pos)

            # The corresponding row in the design matrix H
            # The correct formulation is (reference LOS - other LOS)
            H[i, :] = los_ref_sat - los_sat

            # The "observed minus computed" value for the b vector
            # 1. Compute the geometric range differences based on the approximate baseline
            range_ref_ref = np.linalg.norm(ref_sat_pos - self.ref_pos)
            range_ref_sat = np.linalg.norm(sat_pos - self.ref_pos)
            range_mon_ref_approx = np.linalg.norm(ref_sat_pos - approx_mon_pos)
            range_mon_sat_approx = np.linalg.norm(sat_pos - approx_mon_pos)

            computed_dd = ((range_mon_sat_approx - range_ref_sat) - (range_mon_ref_approx - range_ref_ref)) / self.wavelength

            # For this simulation, we ignore the integer ambiguity term. In reality, this is the hardest part.
            # The b vector is the difference between the "measured" DD and the "computed" DD from the approx position.
            b[i] = dd_obs['dd_value'] - computed_dd

        # Scale b vector by wavelength to get units of meters
        b = b * self.wavelength

        # Solve the linear system H * dX = b for dX (the correction vector)
        # dX contains the corrections [dx, dy, dz] to the approximate baseline.
        dX, residuals, rank, s = np.linalg.lstsq(H, b, rcond=None)

        print(f"  - Solved correction (dx, dy, dz): {dX}")

        # Calculate the final estimated baseline
        self.estimated_baseline = approx_baseline + dX
        print(f"  - Estimated Baseline: {self.estimated_baseline}")
        return self.estimated_baseline


if __name__ == '__main__':
    # --- Configuration ---
    # Define ECEF coordinates for a reference station and a monitoring station
    # Let's assume the monitoring station is ~100m East of the reference.
    REF_STATION_POS = [3275650.0, 553640.0, 5201550.0]
    # The "true" position of the monitoring station, used to generate simulated data
    TRUE_MON_POS = [3275650.0 + 100.0, 553640.0 + 10.0, 5201550.0 - 5.0]

    # Define positions for a few simulated satellites
    SATELLITE_POSITIONS = {
        'G01': [20000000, 10000000, 15000000],
        'G05': [22000000, -5000000, 18000000],
        'G12': [15000000, 15000000, 20000000],
        'G21': [18000000, -12000000, 16000000],
    }

    # Use an approximate position for the monitor station in the solver.
    # This simulates a real-world scenario where the position is not known perfectly.
    APPROX_MON_POS = [TRUE_MON_POS[0] + 0.1, TRUE_MON_POS[1] - 0.1, TRUE_MON_POS[2] + 0.05]

    print("--- Starting Deformation Monitoring Simulation ---")

    # Initialize the monitor
    monitor = DeformationMonitor(
        ref_pos=REF_STATION_POS,
        mon_pos=TRUE_MON_POS, # Use the true position to generate realistic data
        satellite_positions=SATELLITE_POSITIONS
    )

    # 1. Simulate the raw phase observations
    monitor._simulate_phase_observations()

    # 2. Perform double differencing
    monitor.perform_double_differencing()

    # 3. Solve for the baseline vector
    estimated_baseline = monitor.solve_baseline(approx_mon_pos=APPROX_MON_POS)

    # 4. Compare results
    print("\n--- Final Results ---")
    print(f"True Baseline      (dX, dY, dZ): {monitor.true_baseline}")
    print(f"Estimated Baseline (dX, dY, dZ): {estimated_baseline}")
    error = monitor.true_baseline - estimated_baseline
    print(f"Error              (dX, dY, dZ): [{error[0]:.4f}, {error[1]:.4f}, {error[2]:.4f}] m")
    print(f"Error Magnitude: {np.linalg.norm(error):.4f} m")
    print("-----------------------------------------------------")
