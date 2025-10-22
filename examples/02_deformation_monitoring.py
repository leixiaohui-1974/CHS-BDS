"""
Example 2: High-Precision Deformation Monitoring
=================================================

This example demonstrates deformation monitoring using
GNSS double-differencing technique.
"""

import numpy as np
from gnss_monitoring.deformation import DeformationMonitor

print("=" * 70)
print("Example 2: GNSS Deformation Monitoring")
print("=" * 70)

# ===== Configuration =====
print("\nConfiguration:")

# Station coordinates in ECEF (Earth-Centered Earth-Fixed) format
REF_STATION_POS = [3275650.0, 553640.0, 5201550.0]  # meters
TRUE_MON_POS = [3275750.0, 553650.0, 5201545.0]     # meters

# Satellite positions (ECEF)
SATELLITE_POSITIONS = {
    'G01': [20000000, 10000000, 15000000],
    'G05': [22000000, -5000000, 18000000],
    'G12': [15000000, 15000000, 20000000],
    'G21': [18000000, -12000000, 16000000],
}

print(f"  - Reference station: {REF_STATION_POS}")
print(f"  - Monitoring station: {TRUE_MON_POS}")
print(f"  - Number of satellites: {len(SATELLITE_POSITIONS)}")

# ===== Step 1: Calculate True Baseline =====
print("\n" + "-" * 70)
print("Step 1: Calculate True Baseline")
print("-" * 70)

true_baseline = np.array(TRUE_MON_POS) - np.array(REF_STATION_POS)
baseline_length = np.linalg.norm(true_baseline)

print(f"\nTrue baseline vector:")
print(f"  dX = {true_baseline[0]:+.3f} m")
print(f"  dY = {true_baseline[1]:+.3f} m")
print(f"  dZ = {true_baseline[2]:+.3f} m")
print(f"  Length = {baseline_length:.3f} m")

# ===== Step 2: Initialize Monitor =====
print("\nStep 2: Initialize Deformation Monitor")
print("-" * 70)

monitor = DeformationMonitor(
    ref_pos=REF_STATION_POS,
    mon_pos=TRUE_MON_POS,
    satellite_positions=SATELLITE_POSITIONS
)

# ===== Step 3: Simulate Observations =====
print("\nStep 3: Simulate Phase Observations")
print("-" * 70)

observations = monitor._simulate_phase_observations()

print(f"✅ Simulated phase observations for:")
print(f"   - {len(observations['ref'])} measurements at reference station")
print(f"   - {len(observations['mon'])} measurements at monitoring station")

# ===== Step 4: Double Differencing =====
print("\nStep 4: Perform Double Differencing")
print("-" * 70)

double_diffs = monitor.perform_double_differencing()

print(f"✅ Computed {len(double_diffs)} double differences")
print(f"   Reference satellite: {double_diffs[0]['sat1']}")
print(f"\n   Double differences:")
for dd in double_diffs:
    print(f"     {dd['sat1']}-{dd['sat2']}: {dd['dd_value']:.3f} cycles")

# ===== Step 5: Solve for Baseline =====
print("\nStep 5: Solve for Baseline Vector")
print("-" * 70)

# Use approximate position (slightly offset from true position)
APPROX_MON_POS = [
    TRUE_MON_POS[0] + 0.1,  # 10 cm offset in X
    TRUE_MON_POS[1] - 0.1,  # 10 cm offset in Y
    TRUE_MON_POS[2] + 0.05  # 5 cm offset in Z
]

print(f"Starting from approximate position:")
print(f"  X = {APPROX_MON_POS[0]:.3f} m")
print(f"  Y = {APPROX_MON_POS[1]:.3f} m")
print(f"  Z = {APPROX_MON_POS[2]:.3f} m")

estimated_baseline = monitor.solve_baseline(approx_mon_pos=APPROX_MON_POS)

# ===== Step 6: Display Results =====
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print(f"\n📊 Baseline Estimation:")
print(f"\n   Component  |  True (m)  | Estimated (m) |  Error (mm)")
print(f"   " + "-" * 55)
print(f"   dX         | {true_baseline[0]:+10.4f} | {estimated_baseline[0]:+13.4f} | {(true_baseline[0] - estimated_baseline[0])*1000:+10.2f}")
print(f"   dY         | {true_baseline[1]:+10.4f} | {estimated_baseline[1]:+13.4f} | {(true_baseline[1] - estimated_baseline[1])*1000:+10.2f}")
print(f"   dZ         | {true_baseline[2]:+10.4f} | {estimated_baseline[2]:+13.4f} | {(true_baseline[2] - estimated_baseline[2])*1000:+10.2f}")

error_vector = true_baseline - estimated_baseline
error_magnitude = np.linalg.norm(error_vector)

print(f"\n🎯 Accuracy Metrics:")
print(f"   3D error magnitude: {error_magnitude * 1000:.2f} mm")
print(f"   Horizontal error:   {np.linalg.norm(error_vector[:2]) * 1000:.2f} mm")
print(f"   Vertical error:     {abs(error_vector[2]) * 1000:.2f} mm")

if error_magnitude < 0.001:  # < 1mm
    print(f"   ✅ Excellent! Sub-millimeter accuracy achieved")
elif error_magnitude < 0.005:  # < 5mm
    print(f"   ✅ Very good! High-precision monitoring capability")
elif error_magnitude < 0.01:  # < 1cm
    print(f"   ✅ Good accuracy for most monitoring applications")
else:
    print(f"   ⚠️  Moderate accuracy - check observation quality")

# ===== Deformation Analysis =====
print(f"\n💡 Deformation Analysis:")
print(f"   If this were real monitoring data:")
print(f"   - 3D displacement: {error_magnitude * 1000:.2f} mm")
print(f"   - Horizontal displacement: {np.linalg.norm(error_vector[:2]) * 1000:.2f} mm")
print(f"   - Vertical displacement: {error_vector[2] * 1000:.2f} mm")

# Warning thresholds
WARNING_THRESHOLD = 10  # mm
CRITICAL_THRESHOLD = 50  # mm

if error_magnitude * 1000 > CRITICAL_THRESHOLD:
    print(f"   🚨 CRITICAL: Displacement exceeds {CRITICAL_THRESHOLD} mm!")
elif error_magnitude * 1000 > WARNING_THRESHOLD:
    print(f"   ⚠️  WARNING: Displacement exceeds {WARNING_THRESHOLD} mm")
else:
    print(f"   ✅ Displacement within normal range")

print("\n" + "=" * 70)
print("✅ Example completed successfully!")
print("=" * 70)
