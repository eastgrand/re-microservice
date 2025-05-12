# Proposed Additional Field Mappings

Based on analysis of the `nesto_merge_0.csv` file, the following additional fields are recommended for inclusion in the field mapping table:

## Construction Period Fields
| Original Field Name | Proposed Mapped Name | Description | Type | Category |
|---------------------|-------------|-------------|------|----------|
| `2021 Period of Construction - 1960 or Before (Census) (%)` | `Old_Housing_Pct` | Percentage of housing built before 1960 | numeric | Housing |
| `2021 Period of Construction - 1961 to 1980 (Census) (%)` | `Housing_1961_1980_Pct` | Percentage of housing built 1961-1980 | numeric | Housing |
| `2021 Period of Construction - 1981 to 1990 (Census) (%)` | `Housing_1981_1990_Pct` | Percentage of housing built 1981-1990 | numeric | Housing |
| `2021 Period of Construction - 1991 to 2000 (Census) (%)` | `Housing_1991_2000_Pct` | Percentage of housing built 1991-2000 | numeric | Housing |
| `2021 Period of Construction - 2001 to 2010 (Census) (%)` | `Housing_2001_2010_Pct` | Percentage of housing built 2001-2010 | numeric | Housing |
| `2021 Period of Construction - 2011 to 2021 (Census) (%)` | `New_Housing_Pct` | Percentage of housing built 2011-2021 | numeric | Housing |

## Additional Housing Type Fields
| Original Field Name | Proposed Mapped Name | Description | Type | Category |
|---------------------|-------------|-------------|------|----------|
| `2021 Housing: Apartment or Flat in Duplex (Census) (%)` | `Duplex_Apt_Pct` | Percentage of apartments in duplexes | numeric | Housing |
| `2024 Structure Type Movable Dwelling (%)` | `Movable_Dwelling_Pct` | Percentage of movable dwellings | numeric | Housing |
| `2024 Structure Type Other Single-Attached House (%)` | `Other_Attached_House_Pct` | Percentage of other single-attached houses | numeric | Housing |
| `2024 Condominium Status - Not In Condo (%)` | `Non_Condo_Pct` | Percentage of non-condo housing units | numeric | Housing |

## Additional Population Demographics
| Original Field Name | Proposed Mapped Name | Description | Type | Category |
|---------------------|-------------|-------------|------|----------|
| `2024 Visible Minority Arab (%)` | `Arab_Population_Pct` | Percentage of Arab population | numeric | Demographic |
| `2024 Visible Minority Filipino (%)` | `Filipino_Population_Pct` | Percentage of Filipino population | numeric | Demographic |
| `2024 Visible Minority South Asian (%)` | `South_Asian_Population_Pct` | Percentage of South Asian population | numeric | Demographic |
| `2024 Visible Minority Southeast Asian (%)` | `Southeast_Asian_Population_Pct` | Percentage of Southeast Asian population | numeric | Demographic |

## Marital Status
| Original Field Name | Proposed Mapped Name | Description | Type | Category |
|---------------------|-------------|-------------|------|----------|
| `2024 Pop 15+: Married (And Not Separated) (%)` | `Married_Population_Pct` | Percentage of married population | numeric | Demographic |
| `2024 Pop 15+: Single (Never Legally Married) (%)` | `Single_Population_Pct` | Percentage of single population | numeric | Demographic |
| `2024 Pop 15+: Divorced (%)` | `Divorced_Population_Pct` | Percentage of divorced population | numeric | Demographic |

## Economic Indicators
| Original Field Name | Proposed Mapped Name | Description | Type | Category |
|---------------------|-------------|-------------|------|----------|
| `2023-2024 Total Population % Change` | `Recent_Population_Change` | Recent population growth rate | numeric | Economic |
| `2023-2024 Current$ Household Average Income % Change` | `Recent_Income_Change` | Recent income growth rate | numeric | Economic |
| `2024 Household Median Income (Current Year $)` | `Median_Income` | Median household income | numeric | Economic |
| `2024 Property Taxes (Shelter) (Avg)` | `Avg_Property_Tax` | Average property taxes | numeric | Economic |
| `2024 Regular Mortgage Payments (Shelter) (Avg)` | `Avg_Mortgage_Payment` | Average mortgage payments | numeric | Economic |

## Implementation Recommendations:

1. Add these fields to the `FIELD_MAPPINGS` dictionary in `map_nesto_data.py`
2. Update the `NESTO_FIELD_MAPPING.md` document with these new fields
3. Verify that the percentage fields provide additional predictive value for the mortgage approval analysis
4. Consider grouping some of the construction period fields if they show similar correlations with mortgage approvals
