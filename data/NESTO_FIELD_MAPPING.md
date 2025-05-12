# Nesto Data Field Mapping

This document maps the field names in the Nesto dataset (`nesto_merge_0.csv`) to model feature names. The file already contains descriptive field names, making it easier to understand the data and create interpretable model results.

## Field Mapping Table

| Original Field Name | Mapped Name | Description | Type | Category |
|---------------------|-------------|-------------|------|----------|
| `Forward Sortation Area` | `zip_code` | Forward Sortation Area (FSA) postal code | string | Geographic |
| `2024 Total Population` | `Population` | Total population in the area | numeric | Demographic |
| `2024 Household Average Income (Current Year $)` | `Income` | Average household income in CAD | numeric | Economic |
| `2024 Maintainers - Median Age` | `Age` | Median age of primary household maintainers | numeric | Demographic |
| `2024 Tenure: Owned` | `Homeownership` | Number of owner-occupied dwellings | numeric | Housing |
| `2024 Tenure: Rented` | `Rental_Units` | Number of renter-occupied dwellings | numeric | Housing |
| `2024 Visible Minority Black` | `African_American_Population` | Black/African population count | numeric | Demographic |
| `2024 Visible Minority Chinese` | `Asian_Population` | Chinese population count | numeric | Demographic |
| `2024 Visible Minority Latin American` | `Latin_American_Population` | Latin American population count | numeric | Demographic |
| `2024 Visible Minority Total Population` | `Total_Minority_Population` | Total visible minority population | numeric | Demographic |
| `2024 Labour Force - Labour Participation Rate` | `Labor_Participation_Rate` | Labor force participation rate percentage | numeric | Economic |
| `2024 Labour Force - Labour Employment Rate` | `Employment_Rate` | Employment rate percentage | numeric | Economic |
| `2024 Tenure: Total Households` | `Households` | Total number of households | numeric | Housing |
| `Funded Applications` | `Mortgage_Approvals` | Number of approved mortgage applications (target variable) | numeric | Mortgage |
| `ID` | `Province_Code` | Province identification code | string | Geographic |
| `LANDAREA` | `Area_Size` | Land area in square kilometers | numeric | Geographic |
| `SUM_ECYMARNMCL` | `Single_Status` | Number of single/never married adults | numeric | Demographic |
| `SUM_ECYSTYSING` | `Single_Family_Homes` | Number of single-family homes | numeric | Housing |
| `SUM_ECYMARM` | `Married_Population` | Number of married adults | numeric | Demographic |
| `SUM_HSHNIAGG` | `Aggregate_Income` | Aggregate income for the area | numeric | Economic |
| `Sum_Weight` | `Market_Weight` | Market significance weight | numeric | Business |

## Feature Importance for Nike Sales

Based on analysis, the most important features affecting Nike sales are:

1. **Income** - Strong positive correlation with sales
2. **Age** - Demographic factor with moderate influence
3. **Hispanic_Population** - Demographic with notable correlation
4. **Education_Level** - Social factor with significant impact
5. **Households** - Housing density correlation with sales

## Usage in Model

These mapped field names are used in the model training and prediction processes to provide more interpretable results. When accessing SHAP values or feature importance, the mapped names will be used.

## Data Preprocessing Notes

During preprocessing, we:
1. Convert string types to appropriate numeric values
2. Handle missing values with mean imputation
3. Scale features to normalize ranges
4. Exclude non-predictive geographic identifiers from model features

## Adding New Fields

To add new fields to the mapping:
1. Identify the original field name in nesto_training_data_0.csv
2. Choose an intuitive, descriptive mapped name
3. Add to this document and to the field_mapping dictionary in map_nesto_data.py
4. Update the train_model.py script to account for the new field