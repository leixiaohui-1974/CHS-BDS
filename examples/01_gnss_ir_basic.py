"""
Example 1: Basic GNSS-IR Water Level Monitoring
================================================

This example demonstrates how to use the GNSS-IR module
for water level monitoring using simulated data.
"""

import numpy as np
import matplotlib.pyplot as plt
from gnss_monitoring.gnss_ir import GNSSIRWaterLevel

# ===== Configuration =====
print("=" * 70)
print("Example 1: GNSS-IR Water Level Monitoring")
print("=" * 70)

# Define satellite pass parameters
ELEVATION_ANGLES = np.linspace(5, 30, 250)  # degrees
AZIMUTH_ANGLES = np.linspace(120, 150, 250)  # degrees

# GNSS station parameters
ANTENNA_HEIGHT = 1.5  # meters above reference point
WAVELENGTH = 0.1903  # meters (GPS L1 frequency)

# True water level height (to be estimated)
TRUE_REFLECTOR_HEIGHT = 4.5  # meters

print(f"\nConfiguration:")
print(f"  - Antenna height: {ANTENNA_HEIGHT} m")
print(f"  - Signal wavelength: {WAVELENGTH} m (GPS L1)")
print(f"  - True reflector height: {TRUE_REFLECTOR_HEIGHT} m")
print(f"  - Elevation range: {ELEVATION_ANGLES.min():.1f}° - {ELEVATION_ANGLES.max():.1f}°")

# ===== Step 1: Initialize Analyzer =====
print("\n" + "-" * 70)
print("Step 1: Initializing GNSS-IR Analyzer")
print("-" * 70)

analyzer = GNSSIRWaterLevel(
    satellite_elevation=ELEVATION_ANGLES,
    satellite_azimuth=AZIMUTH_ANGLES,
    antenna_height=ANTENNA_HEIGHT,
    wavelength=WAVELENGTH
)

# ===== Step 2: Simulate SNR Data =====
print("\nStep 2: Simulating SNR Data")
print("-" * 70)

analyzer._simulate_snr_data(
    true_reflector_height=TRUE_REFLECTOR_HEIGHT,
    noise_level=0.5,  # dB
    a=10,  # Multipath amplitude
    b=-20,  # Polynomial coefficient
    c=30   # Polynomial coefficient
)

print(f"✅ Generated {len(analyzer.snr_data)} SNR observations")
print(f"   SNR range: {analyzer.snr_data.min():.2f} - {analyzer.snr_data.max():.2f} dB")

# ===== Step 3: Preprocess SNR Data =====
print("\nStep 3: Preprocessing SNR Data")
print("-" * 70)

analyzer.preprocess_snr(detrend_type='linear')

print(f"✅ SNR data detrended")
print(f"   Residual range: {analyzer.snr_residual.min():.2f} - {analyzer.snr_residual.max():.2f} dB")

# ===== Step 4: Analyze Frequency =====
print("\nStep 4: Performing Frequency Analysis")
print("-" * 70)

analyzer.analyze_frequency(
    min_h=0.5,  # Minimum height to search (meters)
    max_h=10.0,  # Maximum height to search (meters)
    oversample_factor=5
)

# ===== Step 5: Display Results =====
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

estimated_height = analyzer.reflector_height
error = abs(TRUE_REFLECTOR_HEIGHT - estimated_height)
error_percent = (error / TRUE_REFLECTOR_HEIGHT) * 100

print(f"\n📊 Height Estimation:")
print(f"   True height:      {TRUE_REFLECTOR_HEIGHT:.3f} m")
print(f"   Estimated height: {estimated_height:.3f} m")
print(f"   Absolute error:   {error:.3f} m ({error_percent:.2f}%)")

if error < 0.1:
    print(f"   ✅ Excellent accuracy!")
elif error < 0.5:
    print(f"   ✅ Good accuracy")
else:
    print(f"   ⚠️  Moderate accuracy - consider improving data quality")

# ===== Step 6: Generate Plots =====
print("\nStep 6: Generating Visualization")
print("-" * 70)

analyzer.plot_results()
print(f"✅ Plot saved as 'gnss_ir_analysis_results.png'")

# ===== Additional Analysis =====
print("\n" + "=" * 70)
print("ADDITIONAL INFORMATION")
print("=" * 70)

water_level = ANTENNA_HEIGHT - estimated_height
print(f"\n💧 Water Level:")
print(f"   Antenna height:   {ANTENNA_HEIGHT} m")
print(f"   Reflector height: {estimated_height:.3f} m")
print(f"   Water level:      {water_level:.3f} m (below antenna)")

# Estimate precision
print(f"\n🎯 Precision Metrics:")
print(f"   Wavelength:       {WAVELENGTH * 1000:.1f} mm")
print(f"   Expected precision: ±{WAVELENGTH * 1000 / 4:.1f} mm (λ/4)")
print(f"   Actual error:     ±{error * 1000:.1f} mm")

print("\n" + "=" * 70)
print("✅ Example completed successfully!")
print("=" * 70)

# Keep plot window open
plt.show()
