# CHS-BDS Examples

This directory contains practical examples demonstrating how to use the CHS-BDS system for GNSS monitoring applications.

## 📚 Available Examples

### Example 1: Basic GNSS-IR Water Level Monitoring
**File**: `01_gnss_ir_basic.py`

Learn how to use GNSS Interference and Reflectometry (GNSS-IR) for water level monitoring.

```bash
python examples/01_gnss_ir_basic.py
```

**What you'll learn:**
- Initialize GNSS-IR analyzer
- Simulate SNR data
- Perform frequency analysis
- Estimate reflector heights
- Interpret results and accuracy

**Output**: `gnss_ir_analysis_results.png`

---

### Example 2: High-Precision Deformation Monitoring
**File**: `02_deformation_monitoring.py`

Demonstrate millimeter-level deformation monitoring using GNSS double-differencing.

```bash
python examples/02_deformation_monitoring.py
```

**What you'll learn:**
- Set up reference and monitoring stations
- Process phase observations
- Perform double differencing
- Solve for baseline vectors
- Assess deformation accuracy

---

### Example 3: PWV Estimation for Weather Monitoring
**File**: `03_pwv_estimation.py`

Extract atmospheric water vapor content from GNSS observations.

```bash
python examples/03_pwv_estimation.py
```

**What you'll learn:**
- Calculate zenith delays (ZTD, ZHD, ZWD)
- Convert delays to precipitable water vapor
- Analyze diurnal PWV variations
- Identify rainfall indicators
- Generate time series plots

**Output**: `pwv_estimation_results.png`

---

### Example 4: Integrated Multi-Module System
**File**: `04_integrated_system.py`

Run all CHS-BDS modules together for comprehensive monitoring.

```bash
python examples/04_integrated_system.py
```

**What you'll learn:**
- Initialize the full CHS-BDS system
- Run all modules in sequence
- Export results in multiple formats
- Generate comprehensive reports
- Understand integration workflows

**Output**:
- `integrated_results.json`
- `integrated_report.md`

---

## 🚀 Quick Start

### Run All Examples

```bash
# Run individual examples
for i in {1..4}; do
    python examples/0${i}_*.py
done
```

### Using the CLI

```bash
# Run all modules
chs-bds run --all

# Run specific module
chs-bds run --module gnss_ir

# With custom configuration
chs-bds run --all --config my_config.yaml
```

---

## 📖 Example Structure

Each example follows this structure:

1. **Configuration** - Set parameters and constants
2. **Initialization** - Create analyzer/monitor objects
3. **Data Processing** - Load or simulate data
4. **Analysis** - Run algorithms
5. **Results** - Display and interpret results
6. **Visualization** - Generate plots
7. **Export** - Save results to files

---

## 🎯 Use Cases

### Coastal and Tidal Monitoring
- Example 1: Water level changes
- Example 4: Integrated coastal monitoring

### Landslide and Ground Deformation
- Example 2: Displacement detection
- Example 4: Combined deformation and rainfall monitoring

### Weather Forecasting
- Example 3: Atmospheric water vapor
- Example 4: Short-term rainfall prediction

### Dam and Infrastructure Safety
- Example 1: Reservoir level
- Example 2: Structure displacement
- Example 3: Flood risk assessment

---

## 📊 Expected Outputs

### Plots
- SNR analysis and frequency spectrum
- Time series of delays and PWV
- Deformation vectors and baselines

### Data Files
- JSON results with detailed metrics
- CSV exports for further analysis
- Markdown reports for documentation

---

## 🔧 Customization

### Modify Parameters

Edit the configuration section in each example:

```python
# Example: Adjust GNSS-IR parameters
ELEVATION_ANGLES = np.linspace(5, 35, 300)  # More data points
TRUE_REFLECTOR_HEIGHT = 3.5  # Different water level
```

### Use Real Data

Replace simulation with data loading:

```python
from gnss_monitoring.data_loader import DataLoader

loader = DataLoader()
snr_data = loader.load_snr_data('path/to/data.csv')
```

---

## 🐛 Troubleshooting

### Import Errors
```bash
# Make sure CHS-BDS is installed
pip install -e .
```

### Missing Dependencies
```bash
# Install all requirements
pip install -r requirements.txt
```

### Plot Not Showing
```python
# Add at the end of script
import matplotlib.pyplot as plt
plt.show()
```

---

## 📚 Further Learning

### Documentation
- [README.md](../README.md) - Project overview
- [IMPROVEMENTS.md](../IMPROVEMENTS.md) - Recent updates
- [config.yaml](../config.yaml) - Configuration reference

### Advanced Topics
- RINEX file processing
- Real-time data streaming
- Multi-station networks
- Machine learning integration

---

## 💡 Tips

1. **Start Simple**: Begin with Example 1, then progress to more complex examples
2. **Experiment**: Modify parameters to understand their effects
3. **Compare**: Run examples multiple times with different configurations
4. **Visualize**: Always check the generated plots
5. **Document**: Save your custom configurations and results

---

## 🤝 Contributing

Have a useful example? Please share!

1. Create your example following the existing structure
2. Add documentation in this README
3. Submit a pull request

---

## 📮 Questions?

- Open an issue: https://github.com/leixiaohui-1974/CHS-BDS/issues
- Check the docs: https://github.com/leixiaohui-1974/CHS-BDS#readme

---

**Happy Monitoring!** 📡🌊🏔️💧
