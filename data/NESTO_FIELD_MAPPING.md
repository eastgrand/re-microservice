# Nesto Data Field Mapping

This document maps the field names in the Nesto dataset (`nesto_merge_0.csv`) to model feature names. The file already contains descriptive field names, making it easier to understand the data and create interpretable model results.

## Field Mapping Table

### Geographic Fields

| Original Field Name | Mapped Name | Description | Type | Category |
|---------------------|-------------|-------------|------|----------|
| `Forward Sortation Area` | `zip_code` | Forward Sortation Area (FSA) postal code | string | Geographic |
| `ID` | `Province_Code` | Province identification code | string | Geographic |
| `Object ID` | `Object_ID` | Unique identifier for geographic object | string | Geographic |
| `Shape__Area` | `Geographic_Area` | Geographic area measurement | numeric | Geographic |
| `Shape__Length` | `Geographic_Length` | Geographic perimeter measurement | numeric | Geographic |
| `LANDAREA` | `Area_Size` | Land area in square kilometers | numeric | Geographic |

### Basic Demographic Fields

| Original Field Name | Mapped Name | Description | Type | Category |
|---------------------|-------------|-------------|------|----------|
| `2024 Total Population` | `Population` | Total population in the area | numeric | Demographic |
| `2024 Household Type - Total Households` | `Total_Household_Types` | Total count of all household types | numeric | Demographic |
| `2024 Maintainers - Median Age` | `Age` | Median age of primary household maintainers | numeric | Demographic |
| `2024 Female Household Population` | `Female_Population` | Count of female population | numeric | Demographic |
| `2024 Female Household Population (%)` | `Female_Population_Pct` | Percentage of female population | numeric | Demographic |
| `2024 Male Household Population` | `Male_Population` | Count of male population | numeric | Demographic |
| `2024 Male Household Population (%)` | `Male_Population_Pct` | Percentage of male population | numeric | Demographic |

### Age Groups and Household Maintainers

| Original Field Name | Mapped Name | Description | Type | Category |
|---------------------|-------------|-------------|------|----------|
| `2024 Maintainers - 25 to 34` | `Young_Adult_Maintainers` | Count of household maintainers aged 25-34 | numeric | Demographic |
| `2024 Maintainers - 25 to 34 (%)` | `Young_Adult_Maintainers_Pct` | Percentage of household maintainers aged 25-34 | numeric | Demographic |
| `2024 Maintainers - 35 to 44` | `Middle_Age_Maintainers` | Count of household maintainers aged 35-44 | numeric | Demographic |
| `2024 Maintainers - 35 to 44 (%)` | `Middle_Age_Maintainers_Pct` | Percentage of household maintainers aged 35-44 | numeric | Demographic |
| `2024 Maintainers - 45 to 54` | `Mature_Maintainers` | Count of household maintainers aged 45-54 | numeric | Demographic |
| `2024 Maintainers - 45 to 54 (%)` | `Mature_Maintainers_Pct` | Percentage of household maintainers aged 45-54 | numeric | Demographic |
| `2024 Maintainers - 55 to 64` | `Senior_Maintainers` | Count of household maintainers aged 55-64 | numeric | Demographic |
| `2024 Maintainers - 55 to 64 (%)` | `Senior_Maintainers_Pct` | Percentage of household maintainers aged 55-64 | numeric | Demographic |

### Housing Tenure Fields

| Original Field Name | Mapped Name | Description | Type | Category |
|---------------------|-------------|-------------|------|----------|
| `2024 Tenure: Owned` | `Homeownership` | Number of owner-occupied dwellings | numeric | Housing |
| `2024 Tenure: Owned (%)` | `Homeownership_Pct` | Percentage of owner-occupied dwellings | numeric | Housing |
| `2024 Tenure: Rented` | `Rental_Units` | Number of renter-occupied dwellings | numeric | Housing |
| `2024 Tenure: Rented (%)` | `Rental_Units_Pct` | Percentage of renter-occupied dwellings | numeric | Housing |
| `2024 Tenure: Band Housing` | `Band_Housing` | Number of band housing units | numeric | Housing |
| `2024 Tenure: Band Housing (%)` | `Band_Housing_Pct` | Percentage of band housing units | numeric | Housing |
| `2024 Tenure: Total Households` | `Households` | Total number of households | numeric | Housing |

### Visible Minority Demographics

| Original Field Name | Mapped Name | Description | Type | Category |
|---------------------|-------------|-------------|------|----------|
| `2024 Visible Minority Black` | `African_American_Population` | Black/African population count | numeric | Demographic |
| `2024 Visible Minority Black (%)` | `African_American_Population_Pct` | Percentage of Black/African population | numeric | Demographic |
| `2024 Visible Minority Chinese` | `Asian_Population` | Chinese population count | numeric | Demographic |
| `2024 Visible Minority Chinese (%)` | `Asian_Population_Pct` | Percentage of Chinese population | numeric | Demographic |
| `2024 Visible Minority Latin American` | `Latin_American_Population` | Latin American population count | numeric | Demographic |
| `2024 Visible Minority Latin American (%)` | `Latin_American_Population_Pct` | Percentage of Latin American population | numeric | Demographic |
| `2024 Visible Minority Arab` | `Arab_Population` | Arab population count | numeric | Demographic |
| `2024 Visible Minority Arab (%)` | `Arab_Population_Pct` | Percentage of Arab population | numeric | Demographic |
| `2024 Visible Minority Filipino` | `Filipino_Population` | Filipino population count | numeric | Demographic |
| `2024 Visible Minority Filipino (%)` | `Filipino_Population_Pct` | Percentage of Filipino population | numeric | Demographic |
| `2024 Visible Minority South Asian` | `South_Asian_Population` | South Asian population count | numeric | Demographic |
| `2024 Visible Minority South Asian (%)` | `South_Asian_Population_Pct` | Percentage of South Asian population | numeric | Demographic |
| `2024 Visible Minority Southeast Asian` | `Southeast_Asian_Population` | Southeast Asian population count | numeric | Demographic |
| `2024 Visible Minority Southeast Asian (%)` | `Southeast_Asian_Population_Pct` | Percentage of Southeast Asian population | numeric | Demographic |
| `2024 Visible Minority Japanese` | `Japanese_Population` | Japanese population count | numeric | Demographic |
| `2024 Visible Minority Japanese (%)` | `Japanese_Population_Pct` | Percentage of Japanese population | numeric | Demographic |
| `2024 Visible Minority Korean` | `Korean_Population` | Korean population count | numeric | Demographic |
| `2024 Visible Minority Korean (%)` | `Korean_Population_Pct` | Percentage of Korean population | numeric | Demographic |
| `2024 Visible Minority West Asian` | `West_Asian_Population` | West Asian population count | numeric | Demographic |
| `2024 Visible Minority West Asian (%)` | `West_Asian_Population_Pct` | Percentage of West Asian population | numeric | Demographic |
| `2024 Visible Minority Total Population` | `Total_Minority_Population` | Total visible minority population | numeric | Demographic |
| `2024 Visible Minority Total Population (%)` | `Total_Minority_Population_Pct` | Percentage of visible minority population | numeric | Demographic |

### Labor Market Indicators

| Original Field Name | Mapped Name | Description | Type | Category |
|---------------------|-------------|-------------|------|----------|
| `2024 Labour Force - Labour Participation Rate` | `Labor_Participation_Rate` | Labor force participation rate percentage | numeric | Economic |
| `2024 Labour Force - Labour Employment Rate` | `Employment_Rate` | Employment rate percentage | numeric | Economic |
| `2024 Labour Force - Labour Unemployment Rate` | `Unemployment_Rate` | Unemployment rate percentage | numeric | Economic |

### Housing Type Fields

| Original Field Name | Mapped Name | Description | Type | Category |
|---------------------|-------------|-------------|------|----------|
| `2024 Condominium Status - In Condo` | `Condo_Units` | Number of condominium units | numeric | Housing |
| `2024 Condominium Status - In Condo (%)` | `Condo_Ownership_Pct` | Percentage of households in condominiums | numeric | Housing |
| `2024 Condominium Status - Not In Condo` | `Non_Condo_Units` | Number of non-condominium units | numeric | Housing |
| `2024 Condominium Status - Not In Condo (%)` | `Non_Condo_Pct` | Percentage of non-condo housing units | numeric | Housing |
| `2024 Condominium Status - Total Households` | `Total_Condo_Status_Households` | Total households with condominium status | numeric | Housing |
| `2024 Structure Type Single-Detached House` | `Single_Detached_Houses` | Number of single-detached houses | numeric | Housing |
| `2024 Structure Type Single-Detached House (%)` | `Single_Detached_House_Pct` | Percentage of single-detached houses | numeric | Housing |
| `2024 Structure Type Semi-Detached House` | `Semi_Detached_Houses` | Number of semi-detached houses | numeric | Housing |
| `2024 Structure Type Semi-Detached House (%)` | `Semi_Detached_House_Pct` | Percentage of semi-detached houses | numeric | Housing |
| `2024 Structure Type Row House` | `Row_Houses` | Number of row houses | numeric | Housing |
| `2024 Structure Type Row House (%)` | `Row_House_Pct` | Percentage of row houses | numeric | Housing |
| `2024 Structure Type Apartment, Building Five or More Story` | `Large_Apartments` | Number of apartments in buildings with 5+ stories | numeric | Housing |
| `2024 Structure Type Apartment, Building Five or More Story (%)` | `Large_Apartment_Pct` | Percentage of apartments in buildings with 5+ stories | numeric | Housing |
| `2024 Structure Type Apartment, Building Fewer Than Five Story` | `Small_Apartments` | Number of apartments in buildings with less than 5 stories | numeric | Housing |
| `2024 Structure Type Apartment, Building Fewer Than Five Story (%)` | `Small_Apartment_Pct` | Percentage of apartments in buildings with less than 5 stories | numeric | Housing |
| `2024 Structure Type Movable Dwelling` | `Movable_Dwellings` | Number of movable dwellings | numeric | Housing |
| `2024 Structure Type Movable Dwelling (%)` | `Movable_Dwelling_Pct` | Percentage of movable dwellings | numeric | Housing |
| `2024 Structure Type Other Single-Attached House` | `Other_Single_Attached_Houses` | Number of other single-attached houses | numeric | Housing |
| `2024 Structure Type Other Single-Attached House (%)` | `Other_Attached_House_Pct` | Percentage of other single-attached houses | numeric | Housing |
| `2021 Housing: Apartment or Flat in Duplex (Census)` | `Duplex_Apartments` | Number of apartments in duplexes | numeric | Housing |
| `2021 Housing: Apartment or Flat in Duplex (Census) (%)` | `Duplex_Apt_Pct` | Percentage of apartments in duplexes | numeric | Housing |

### Housing Age/Period Fields

| Original Field Name | Mapped Name | Description | Type | Category |
|---------------------|-------------|-------------|------|----------|
| `2021 Period of Construction - 1960 or Before (Census)` | `Old_Housing` | Count of housing built before 1960 | numeric | Housing |
| `2021 Period of Construction - 1960 or Before (Census) (%)` | `Old_Housing_Pct` | Percentage of housing built before 1960 | numeric | Housing |
| `2021 Period of Construction - 1961 to 1980 (Census)` | `Housing_1961_1980` | Count of housing built 1961-1980 | numeric | Housing |
| `2021 Period of Construction - 1961 to 1980 (Census) (%)` | `Housing_1961_1980_Pct` | Percentage of housing built 1961-1980 | numeric | Housing |
| `2021 Period of Construction - 1981 to 1990 (Census)` | `Housing_1981_1990` | Count of housing built 1981-1990 | numeric | Housing |
| `2021 Period of Construction - 1981 to 1990 (Census) (%)` | `Housing_1981_1990_Pct` | Percentage of housing built 1981-1990 | numeric | Housing |
| `2021 Period of Construction - 1991 to 2000 (Census)` | `Housing_1991_2000` | Count of housing built 1991-2000 | numeric | Housing |
| `2021 Period of Construction - 1991 to 2000 (Census) (%)` | `Housing_1991_2000_Pct` | Percentage of housing built 1991-2000 | numeric | Housing |
| `2021 Period of Construction - 2001 to 2005 (Census)` | `Housing_2001_2005` | Count of housing built 2001-2005 | numeric | Housing |
| `2021 Period of Construction - 2001 to 2005 (Census) (%)` | `Housing_2001_2005_Pct` | Percentage of housing built 2001-2005 | numeric | Housing |
| `2021 Period of Construction - 2006 to 2010 (Census)` | `Housing_2006_2010` | Count of housing built 2006-2010 | numeric | Housing |
| `2021 Period of Construction - 2006 to 2010 (Census) (%)` | `Housing_2006_2010_Pct` | Percentage of housing built 2006-2010 | numeric | Housing |
| `2021 Period of Construction - 2011 to 2016 (Census)` | `Housing_2011_2016` | Count of housing built 2011-2016 | numeric | Housing |
| `2021 Period of Construction - 2011 to 2016 (Census) (%)` | `Housing_2011_2016_Pct` | Percentage of housing built 2011-2016 | numeric | Housing |
| `2021 Period of Construction - 2016 to 2021 (Census)` | `Housing_2016_2021` | Count of housing built 2016-2021 | numeric | Housing |
| `2021 Period of Construction - 2016 to 2021 (Census) (%)` | `Housing_2016_2021_Pct` | Percentage of housing built 2016-2021 | numeric | Housing |

### Economic Change Indicators

| Original Field Name | Mapped Name | Description | Type | Category |
|---------------------|-------------|-------------|------|----------|
| `2023-2024 Total Population % Change` | `Recent_Population_Change` | Recent population growth rate | numeric | Economic |
| `2023-2024 Current$ Household Average Income % Change` | `Recent_Income_Change` | Recent income growth rate | numeric | Economic |
| `2022-2023 Total Population % Change` | `Prior_Year_Population_Change` | Previous year's population growth rate | numeric | Economic |
| `2022-2023 Current$ Household Average Income % Change` | `Prior_Year_Income_Change` | Previous year's income growth rate | numeric | Economic |
| `2021-2022 Total Population % Change` | `Two_Year_Prior_Population_Change` | Population growth rate from two years prior | numeric | Economic |
| `2024-2025 Total Population % Change` | `Next_Year_Population_Change_Projection` | Projected population growth for next year | numeric | Economic |
| `2024-2025 Current$ Household Average Income % Change` | `Next_Year_Income_Change_Projection` | Projected income growth for next year | numeric | Economic |
| `2025-2026 Total Population % Change` | `Two_Year_Population_Change_Projection` | Two-year forward population growth projection | numeric | Economic |
| `2025-2026 Current$ Household Average Income % Change` | `Two_Year_Income_Change_Projection` | Two-year forward income growth projection | numeric | Economic |
| `2026-2027 Total Population % Change` | `Three_Year_Population_Change_Projection` | Three-year forward population growth projection | numeric | Economic |
| `2026-2027 Current$ Household Average Income % Change` | `Three_Year_Income_Change_Projection` | Three-year forward income growth projection | numeric | Economic |

### Marital Status Demographics

| Original Field Name | Mapped Name | Description | Type | Category |
|---------------------|-------------|-------------|------|----------|
| `2024 Pop 15+: Married (And Not Separated)` | `Married_Population` | Count of married population | numeric | Demographic |
| `2024 Pop 15+: Married (And Not Separated) (%)` | `Married_Population_Pct` | Percentage of married population | numeric | Demographic |
| `2024 Pop 15+: Single (Never Legally Married)` | `Single_Population` | Count of single (never married) population | numeric | Demographic |
| `2024 Pop 15+: Single (Never Legally Married) (%)` | `Single_Population_Pct` | Percentage of single population | numeric | Demographic |
| `2024 Pop 15+: Divorced` | `Divorced_Population` | Count of divorced population | numeric | Demographic |
| `2024 Pop 15+: Divorced (%)` | `Divorced_Population_Pct` | Percentage of divorced population | numeric | Demographic |
| `2024 Pop 15+: Separated` | `Separated_Population` | Count of separated population | numeric | Demographic |
| `2024 Pop 15+: Separated (%)` | `Separated_Population_Pct` | Percentage of separated population | numeric | Demographic |
| `2024 Pop 15+: Widowed` | `Widowed_Population` | Count of widowed population | numeric | Demographic |
| `2024 Pop 15+: Widowed (%)` | `Widowed_Population_Pct` | Percentage of widowed population | numeric | Demographic |
| `2024 Pop 15+: Living Common Law` | `Common_Law_Population` | Count of population in common law relationships | numeric | Demographic |
| `2024 Pop 15+: Living Common Law (%)` | `Common_Law_Population_Pct` | Percentage of population in common law relationships | numeric | Demographic |
| `2024 Pop 15+: Married or Living Common-Law` | `Combined_Married_Population` | Count of population either married or in common law | numeric | Demographic |
| `2024 Pop 15+: Married or Living Common-Law (%)` | `Combined_Married_Population_Pct` | Percentage of population either married or in common law | numeric | Demographic |
| `2024 Pop 15+: Not Married or Common-Law` | `Not_Married_Population` | Count of population not in marriage or common law | numeric | Demographic |
| `2024 Pop 15+: Not Married or Common-Law (%)` | `Not_Married_Population_Pct` | Percentage of population not in marriage or common law | numeric | Demographic |

### Financial Metrics - Income and Housing Costs

| Original Field Name | Mapped Name | Description | Type | Category |
|---------------------|-------------|-------------|------|----------|
| `2024 Household Average Income (Current Year $)` | `Income` | Average household income in CAD | numeric | Economic |
| `2024 Household Median Income (Current Year $)` | `Median_Income` | Median household income in CAD | numeric | Economic |
| `2024 Property Taxes (Shelter)` | `Property_Tax_Total` | Total property taxes for the area | numeric | Economic |
| `2024 Property Taxes (Shelter) (Avg)` | `Avg_Property_Tax` | Average property taxes | numeric | Economic |
| `2024 Regular Mortgage Payments (Shelter)` | `Mortgage_Payments_Total` | Total mortgage payments for the area | numeric | Economic |
| `2024 Regular Mortgage Payments (Shelter) (Avg)` | `Avg_Mortgage_Payment` | Average mortgage payments | numeric | Economic |
| `2024 Condominium Charges (Shelter)` | `Condo_Fees_Total` | Total condominium fees for the area | numeric | Economic |
| `2024 Condominium Charges (Shelter) (Avg)` | `Avg_Condo_Fees` | Average condominium fees | numeric | Economic |
| `2024 Household Aggregate Income` | `Total_Income` | Total household income | numeric | Economic |
| `2024 Household Aggregate Income (Current Year $)` | `Current_Year_Total_Income` | Total household income in current year dollars | numeric | Economic |
| `2024 Household Discretionary Aggregate Income` | `Discretionary_Income` | Total discretionary income | numeric | Economic |
| `2024 Household Disposable Aggregate Income` | `Disposable_Income` | Total disposable income | numeric | Economic |

### Banking and Financial Services

| Original Field Name | Mapped Name | Description | Type | Category |
|---------------------|-------------|-------------|------|----------|
| `2024 Financial Services` | `Financial_Services_Total` | Total financial services expenditures | numeric | Economic |
| `2024 Financial Services (Avg)` | `Avg_Financial_Services` | Average financial services expenditures | numeric | Economic |
| `2024 Service Charges for Banks, Other Financial Institutions` | `Bank_Service_Charges_Total` | Total bank service charges | numeric | Economic |
| `2024 Service Charges for Banks, Other Financial Institutions (Avg)` | `Avg_Bank_Service_Charges` | Average bank service charges | numeric | Economic |

### Mortgage Data

| Original Field Name | Mapped Name | Description | Type | Category |
|---------------------|-------------|-------------|------|----------|
| `Mortgage Applicationns` | `Mortgage_Applications` | Number of mortgage applications submitted | numeric | Mortgage |
| `Funded Applications` | `Mortgage_Approvals` | Number of approved mortgage applications (target variable) | numeric | Mortgage |

### Additional Legacy Fields

| Original Field Name | Mapped Name | Description | Type | Category |
|---------------------|-------------|-------------|------|----------|
| `SUM_ECYMARNMCL` | `Single_Status` | Number of single/never married adults | numeric | Demographic |
| `SUM_ECYSTYSING` | `Single_Family_Homes` | Number of single-family homes | numeric | Housing |
| `SUM_ECYMARM` | `Married_Population` | Number of married adults | numeric | Demographic |
| `SUM_HSHNIAGG` | `Aggregate_Income` | Aggregate income for the area | numeric | Economic |
| `Sum_Weight` | `Market_Weight` | Market significance weight | numeric | Business |

## Feature Importance for Mortgage Approvals

Based on analysis, the most important features affecting mortgage approvals are:

1. **Income** - Strong positive correlation with mortgage approval rates
2. **Homeownership_Pct** - Areas with high homeownership show different approval patterns
3. **Employment_Rate** - Economic stability indicator strongly correlated with approvals
4. **Age** - Demographic factor with moderate influence on mortgage decisions
5. **Total_Minority_Population_Pct** - Demographic variable with significant correlations

## Usage in Model

These mapped field names are used in the model training and prediction processes to provide more interpretable results. When accessing SHAP values or feature importance, the mapped names will be used.

## Data Preprocessing Notes

During preprocessing, we:
1. Convert string types to appropriate numeric values
2. Handle missing values with mean imputation
3. Scale features to normalize ranges
4. Exclude non-predictive geographic identifiers from model features

## Importance of Percentage Fields

Percentage fields are particularly valuable for analysis because:

1. **Normalization**: They account for different area sizes and populations
2. **Comparability**: Makes it easier to compare different regions regardless of absolute numbers
3. **Correlation Analysis**: Often shows stronger and more meaningful correlations than absolute values
4. **Regression Analysis**: Typically more statistically significant in predictive models
5. **Interpretability**: Easier to understand the relative importance of variables

For example, while knowing the total number of rental units in an area is useful, the percentage of rental units provides insight into the market characteristics regardless of the area's total size. This can help identify trends that absolute numbers might obscure.

## Adding New Fields

To add new fields to the mapping:

1. Identify the original field name in nesto_merge_0.csv
2. Choose an intuitive, descriptive mapped name:
   - For percentage fields, add a `_Pct` suffix
   - For absolute counts, use clear descriptive names
3. Add to this document and to the field_mapping dictionary in map_nesto_data.py
4. Update the train_model.py script to account for the new field

## Feature Importance for Mortgage Approvals

Based on analysis, the most important features affecting mortgage approvals are:

1. **Income** - Strong positive correlation with mortgage approval rates
2. **Homeownership_Pct** - Areas with high homeownership show different approval patterns
3. **Employment_Rate** - Economic stability indicator strongly correlated with approvals
4. **Age** - Demographic factor with moderate influence on mortgage decisions
5. **Total_Minority_Population_Pct** - Demographic variable with significant correlations

## Usage in Model

These mapped field names are used in the model training and prediction processes to provide more interpretable results. When accessing SHAP values or feature importance, the mapped names will be used.

## Data Preprocessing Notes

During preprocessing, we:
1. Convert string types to appropriate numeric values
2. Handle missing values with mean imputation
3. Scale features to normalize ranges
4. Exclude non-predictive geographic identifiers from model features

## Importance of Percentage Fields

Percentage fields are particularly valuable for analysis because:

1. **Normalization**: They account for different area sizes and populations
2. **Comparability**: Makes it easier to compare different regions regardless of absolute numbers
3. **Correlation Analysis**: Often shows stronger and more meaningful correlations than absolute values
4. **Regression Analysis**: Typically more statistically significant in predictive models
5. **Interpretability**: Easier to understand the relative importance of variables

For example, while knowing the total number of rental units in an area is useful, the percentage of rental units provides insight into the market characteristics regardless of the area's total size. This can help identify trends that absolute numbers might obscure.

## Adding New Fields

To add new fields to the mapping:

1. Identify the original field name in nesto_merge_0.csv
2. Choose an intuitive, descriptive mapped name:
   - For percentage fields, add a `_Pct` suffix
   - For absolute counts, use clear descriptive names
3. Add to this document and to the field_mapping dictionary in map_nesto_data.py
4. Update the train_model.py script to account for the new field