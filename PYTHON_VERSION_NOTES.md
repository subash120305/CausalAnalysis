# Python Version Compatibility Notes

## Python 3.13 Compatibility

This project supports Python 3.10-3.13 with conditional dependencies:

### Package Versions by Python Version

#### Python 3.13
- **DoWhy**: 0.8.x (latest compatible version)
- **EconML**: 0.13.0+
- **NOTEARS**: Not available (will be skipped gracefully)

#### Python 3.10-3.12
- **DoWhy**: 0.11.0+ (latest stable)
- **EconML**: 0.14.0+ (latest stable)
- **NOTEARS**: 0.3.0+ (available)

### Installation

The `requirements.txt` file automatically selects the correct versions based on your Python version:

```bash
pip install -r requirements.txt
```

### Feature Availability

Some features may be limited on Python 3.13:
- ✅ DoWhy estimation methods (IPW, PSM, DR)
- ✅ EconML estimators (DML, DRLearner)
- ✅ PC and FCI discovery algorithms
- ❌ NOTEARS discovery (not available for Python 3.13)

### Recommendations

For best compatibility and feature availability:
- **Recommended**: Python 3.10, 3.11, or 3.12
- **Supported**: Python 3.13 (with limitations noted above)

If you encounter issues with Python 3.13, consider using Python 3.10-3.12 instead.
