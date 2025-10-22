"""
Example 4: Integrated Multi-Module System
==========================================

This example demonstrates running all CHS-BDS modules
together and generating a comprehensive report.
"""

from gnss_monitoring.main import CHSBDSSystem
from gnss_monitoring.data_export import DataExporter
import json

print("=" * 70)
print("Example 4: Integrated CHS-BDS System")
print("=" * 70)

# ===== Step 1: Initialize System =====
print("\n" + "-" * 70)
print("Step 1: Initialize CHS-BDS System")
print("-" * 70)

system = CHSBDSSystem(config_path='config.yaml')

print(f"✅ System initialized successfully")
print(f"   Modules loaded: GNSS-IR, Deformation, PWV, Rainfall")

# ===== Step 2: Run All Modules =====
print("\nStep 2: Run All Monitoring Modules")
print("-" * 70)

# Run complete analysis
results = system.run_all()

# ===== Step 3: Display Summary =====
print("\n" + "=" * 70)
print("INTEGRATED RESULTS SUMMARY")
print("=" * 70)

if results['status'] == 'success':
    print(f"\n✅ All modules completed successfully!\n")

    # GNSS-IR Results
    if 'gnss_ir' in results['modules']:
        gnss_ir = results['modules']['gnss_ir']
        print(f"📡 GNSS-IR Water Level:")
        print(f"   Estimated height: {gnss_ir['estimated_height']:.3f} m")
        print(f"   Accuracy: ±{gnss_ir['error']*1000:.1f} mm")

    # Deformation Results
    if 'deformation' in results['modules']:
        deform = results['modules']['deformation']
        print(f"\n🏔️  Deformation Monitoring:")
        print(f"   3D error: {deform['error_magnitude']*1000:.2f} mm")
        print(f"   Baseline: {deform['estimated_baseline']}")

    # PWV Results
    if 'pwv' in results['modules']:
        pwv = results['modules']['pwv']
        print(f"\n💧 Precipitable Water Vapor:")
        print(f"   Mean PWV: {pwv['mean_pwv']:.2f} mm")
        print(f"   Range: {pwv['min_pwv']:.2f} - {pwv['max_pwv']:.2f} mm")

    # Rainfall Results
    if 'rainfall' in results['modules']:
        rainfall = results['modules']['rainfall']
        print(f"\n🌧️  Rainfall Prediction:")
        print(f"   Model: {rainfall['model_type']}")
        print(f"   Training samples: {rainfall['training_samples']}")
        print(f"   Status: {rainfall['status']}")

else:
    print(f"\n❌ Analysis failed: {results.get('error', 'Unknown error')}")

# ===== Step 4: Export Results =====
print("\n" + "-" * 70)
print("Step 4: Export Results")
print("-" * 70)

exporter = DataExporter(output_dir='./output/examples')

# Export to multiple formats
print(f"\nExporting results to multiple formats...")

# JSON export
json_path = exporter.export_json(results, 'integrated_results')
print(f"✅ JSON: {json_path}")

# Markdown report
md_path = exporter.export_markdown(results, 'integrated_report')
print(f"✅ Markdown: {md_path}")

# ===== Step 5: Generate Summary Statistics =====
print("\n" + "-" * 70)
print("Step 5: System Performance Statistics")
print("-" * 70)

print(f"\n📊 Analysis Summary:")
print(f"   System version: {results['version']}")
print(f"   Modules executed: {len(results['modules'])}")
print(f"   Overall status: {results['status']}")

# Calculate processing metrics
if 'modules' in results:
    total_data_points = 0
    for module_name, module_data in results['modules'].items():
        if 'num_epochs' in module_data:
            total_data_points += module_data['num_epochs']

    print(f"   Total data points processed: {total_data_points}")

# ===== Step 6: Application Scenarios =====
print("\n" + "=" * 70)
print("APPLICATION SCENARIOS")
print("=" * 70)

print(f"\n🎯 This integrated system can be used for:")
print(f"\n1. Coastal Monitoring:")
print(f"   - GNSS-IR → Tide level monitoring")
print(f"   - Deformation → Coastal erosion detection")
print(f"   - PWV/Rainfall → Storm surge prediction")

print(f"\n2. Landslide Warning:")
print(f"   - Deformation → Ground displacement detection")
print(f"   - PWV/Rainfall → Precipitation monitoring")
print(f"   - Integrated → Early warning system")

print(f"\n3. Dam Safety:")
print(f"   - GNSS-IR → Reservoir level monitoring")
print(f"   - Deformation → Dam structure monitoring")
print(f"   - PWV/Rainfall → Flood prediction")

print(f"\n4. Weather Forecasting:")
print(f"   - PWV → Water vapor monitoring")
print(f"   - Rainfall → Short-term precipitation prediction")
print(f"   - Integrated → Nowcasting system")

# ===== Step 7: Next Steps =====
print("\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70)

print(f"\n📚 To use with real data:")
print(f"   1. Load RINEX files using DataLoader")
print(f"   2. Configure parameters in config.yaml")
print(f"   3. Run specific modules: chs-bds run --module <name>")
print(f"   4. Set up automated monitoring with cron/systemd")

print(f"\n🔧 For production deployment:")
print(f"   1. Use Docker: docker-compose up -d")
print(f"   2. Configure database for data storage")
print(f"   3. Set up alert system for threshold exceedances")
print(f"   4. Deploy web dashboard for visualization")

print("\n" + "=" * 70)
print("✅ Integrated system example completed!")
print("=" * 70)
