"""
Example 3: Precipitable Water Vapor (PWV) Estimation
=====================================================

This example demonstrates PWV estimation from GNSS
zenith total delay measurements.
"""

import numpy as np
import matplotlib.pyplot as plt
from gnss_monitoring.pwv import PWVEstimator

print("=" * 70)
print("Example 3: PWV Estimation for Weather Monitoring")
print("=" * 70)

# ===== Configuration =====
print("\nConfiguration:")

STATION_LATITUDE = 34.0  # degrees
STATION_HEIGHT = 150.0   # meters above sea level
NUM_EPOCHS = 288         # 24 hours at 5-minute intervals
INTERVAL_MINUTES = 5

print(f"  - Station latitude: {STATION_LATITUDE}°")
print(f"  - Station height: {STATION_HEIGHT} m ASL")
print(f"  - Time resolution: {INTERVAL_MINUTES} minutes")
print(f"  - Total epochs: {NUM_EPOCHS} ({NUM_EPOCHS * INTERVAL_MINUTES / 60:.1f} hours)")

# ===== Step 1: Initialize Estimator =====
print("\n" + "-" * 70)
print("Step 1: Initialize PWV Estimator")
print("-" * 70)

estimator = PWVEstimator(
    latitude=STATION_LATITUDE,
    height=STATION_HEIGHT
)

# ===== Step 2: Simulate Input Data =====
print("\nStep 2: Simulate GNSS and Meteorological Data")
print("-" * 70)

timestamps, ztd, pressure, temperature = estimator.simulate_inputs(
    num_epochs=NUM_EPOCHS,
    interval_minutes=INTERVAL_MINUTES
)

print(f"✅ Generated input data:")
print(f"   ZTD (Zenith Total Delay):")
print(f"     - Range: {ztd.min():.4f} - {ztd.max():.4f} m")
print(f"     - Mean: {ztd.mean():.4f} m")
print(f"   Surface Pressure:")
print(f"     - Range: {pressure.min():.1f} - {pressure.max():.1f} hPa")
print(f"     - Mean: {pressure.mean():.1f} hPa")
print(f"   Surface Temperature:")
print(f"     - Range: {temperature.min():.1f} - {temperature.max():.1f} °C")
print(f"     - Mean: {temperature.mean():.1f} °C")

# ===== Step 3: Calculate ZHD =====
print("\nStep 3: Calculate Zenith Hydrostatic Delay (ZHD)")
print("-" * 70)

zhd = estimator.calculate_zhd()

print(f"✅ ZHD calculated using Saastamoinen model:")
print(f"   - Range: {zhd.min():.4f} - {zhd.max():.4f} m")
print(f"   - Mean: {zhd.mean():.4f} m")
print(f"   - Std: {zhd.std():.4f} m")

# ===== Step 4: Calculate PWV =====
print("\nStep 4: Calculate Precipitable Water Vapor (PWV)")
print("-" * 70)

pwv = estimator.calculate_pwv()

print(f"✅ PWV calculated using Bevis et al. (1992) formula:")
print(f"   - Range: {pwv.min():.2f} - {pwv.max():.2f} mm")
print(f"   - Mean: {pwv.mean():.2f} mm")
print(f"   - Std: {pwv.std():.2f} mm")

# ===== Step 5: Analyze Results =====
print("\n" + "=" * 70)
print("RESULTS ANALYSIS")
print("=" * 70)

# Find peak PWV
max_pwv_idx = np.argmax(pwv)
max_pwv_time = timestamps[max_pwv_idx] / 60  # Convert to hours

print(f"\n💧 PWV Statistics:")
print(f"   Minimum PWV: {pwv.min():.2f} mm (at {timestamps[np.argmin(pwv)]/60:.1f} hours)")
print(f"   Maximum PWV: {pwv.max():.2f} mm (at {max_pwv_time:.1f} hours)")
print(f"   Mean PWV: {pwv.mean():.2f} mm")
print(f"   Standard deviation: {pwv.std():.2f} mm")
print(f"   Range: {pwv.max() - pwv.min():.2f} mm")

# PWV interpretation
print(f"\n🌤️  Weather Interpretation:")
if pwv.max() < 15:
    print(f"   ✅ Dry atmosphere - low precipitation probability")
elif pwv.max() < 30:
    print(f"   ⚠️  Moderate moisture - watch for precipitation")
elif pwv.max() < 45:
    print(f"   ⚠️  High moisture - precipitation likely")
else:
    print(f"   🚨 Very high moisture - heavy precipitation possible")

# ZWD analysis
zwd = estimator.zwd
print(f"\n📊 Component Analysis:")
print(f"   ZTD (Total):       {ztd.mean():.4f} m ({ztd.mean()/ztd.mean()*100:.1f}%)")
print(f"   ZHD (Hydrostatic): {zhd.mean():.4f} m ({zhd.mean()/ztd.mean()*100:.1f}%)")
print(f"   ZWD (Wet):         {zwd.mean():.4f} m ({zwd.mean()/ztd.mean()*100:.1f}%)")
print(f"\n   Wet delay represents ~{zwd.mean()/ztd.mean()*100:.1f}% of total delay")

# Rainfall prediction
pwv_rate = np.diff(pwv) / (INTERVAL_MINUTES / 60)  # mm/hour
rapid_increase = np.where(pwv_rate > 2.0)[0]  # PWV increasing > 2 mm/hour

print(f"\n🌧️  Rainfall Indicators:")
if len(rapid_increase) > 0:
    print(f"   ⚠️  {len(rapid_increase)} periods with rapid PWV increase detected")
    print(f"   Maximum PWV rate: {pwv_rate.max():.2f} mm/hour")
    print(f"   → Potential rainfall events identified")
else:
    print(f"   ✅ No rapid PWV increases detected")
    print(f"   → Stable atmospheric conditions")

# ===== Step 6: Generate Visualization =====
print("\nStep 6: Generate Visualization")
print("-" * 70)

estimator.plot_results()
print(f"✅ Plots saved as 'pwv_estimation_results.png'")

print("\n" + "=" * 70)
print("✅ Example completed successfully!")
print("=" * 70)

# Keep plot window open
plt.show()
