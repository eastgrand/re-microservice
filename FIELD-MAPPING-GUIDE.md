# Field Mapping for Nesto Mortgage Data

This document provides an overview of the field mapping system used to convert descriptive field names from the Nesto dataset to standardized model feature names.

## How Field Mapping Works

The Nesto mortgage analytics service uses a field mapping system to convert between descriptive field names (like "2024 Total Population") and standardized model feature names (like "Population"). This provides several benefits:

1. **Readability**: More descriptive field names in the source data
2. **Standardization**: Consistent feature names across the system
3. **Model Compatibility**: Names work well with machine learning libraries
4. **Documentation**: Clear mapping between business concepts and model features

## Field Mapping Process

1. Source data uses descriptive field names from `nesto_merge_0.csv`
2. `map_nesto_data.py` contains the `FIELD_MAPPINGS` dictionary defining the mappings
3. During processing, field names are converted to their standardized versions
4. The model is trained using these standardized feature names
5. API responses use the standardized feature names for consistency

## Key Field Categories

The mapping includes these key categories:

1. **Geographic Fields**: Location identifiers (FSA codes, postal codes)
2. **Basic Demographics**: Population, age, gender distribution
3. **Housing Characteristics**: Ownership rates, property types
4. **Economic Indicators**: Income levels, employment rates
5. **Mortgage Data**: Application and approval counts

## Important Mapped Fields

| Original Field Name | Mapped Name | Description |
|---------------------|-------------|-------------|
| `Forward Sortation Area` | `zip_code` | Canadian Forward Sortation Area postal code |
| `2024 Total Population` | `Population` | Total population in the area |
| `2024 Household Average Income` | `Income` | Average household income |
| `Mortgage Applicationns` | `Mortgage_Applications` | Number of mortgage applications submitted |
| `Funded Applications` | `Mortgage_Approvals` | Number of approved mortgage applications (target variable) |

## Complete Documentation

For the complete field mapping reference, see:
- `data/NESTO_FIELD_MAPPING.md`: Complete documentation of all field mappings
- `map_nesto_data.py`: Implementation of the mapping system

## Adding New Fields

To add new fields to the mapping:

1. Identify the original field name in nesto_merge_0.csv
2. Choose an appropriate standardized name:
   - For percentage fields, add a `_Pct` suffix
   - For absolute counts, use clear descriptive names
3. Add to the mapping dictionary in `map_nesto_data.py`
4. Update the documentation in `data/NESTO_FIELD_MAPPING.md`
