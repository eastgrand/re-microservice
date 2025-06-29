# Frontend Development Gotchas

This document lists common issues and pitfalls that developers might encounter when working on the MPIQ frontend, especially when integrating with the ArcGIS API.

---

## 1. ArcGIS `Graphic.attributes` vs. GeoJSON `properties`

A critical distinction to remember is how ArcGIS `Graphic` objects and standard GeoJSON features store their data. This has been a recurring source of bugs, particularly during data merging operations in `geospatial-chat-interface.tsx`.

### The Problem

-   **GeoJSON Features** store their metadata in a `properties` object.
-   **ArcGIS `Graphic` Objects** (which are often returned from ArcGIS services or created by the API) store their metadata in an `attributes` object.

When merging data from our microservice (which is in a standard object format) with geographic features from ArcGIS, it's easy to forget this distinction.

### Incorrect Merge Logic (The Bug)

If you have a `geoFeature` (an ArcGIS `Graphic`) and a `record` (from our analysis microservice), you might be tempted to merge them like this:

```typescript
// --- THIS IS WRONG ---
const merged = {
  ...geoFeature,
  properties: {
    ...geoFeature.properties, // geoFeature.properties is UNDEFINED!
    ...record,
  }
};
```

This code fails because `geoFeature.properties` is `undefined`. The `...` spread operator will silently fail, and you will lose all the original attributes of the geographic feature. The only data remaining in the new `properties` object will be from the `record`.

This was the root cause of a major bug where the visualization factory failed to find any numeric fields from the base geography, as they had been wiped out during the merge.

### Correct Merge Logic (The Fix)

The correct way to perform the merge is to explicitly spread both `geoFeature.attributes` and `geoFeature.properties`. Spreading `properties` as well is a safe practice in case the feature object has both.

```typescript
// --- THIS IS CORRECT ---
const merged = {
  ...geoFeature,
  properties: {
    ...geoFeature.attributes, // Correctly includes the ArcGIS data
    ...geoFeature.properties, // Handles any other properties that might exist
    ...record,
  }
};
```

By including `...geoFeature.attributes`, you ensure that all the data from the base geographic layer is preserved and correctly joined with the analysis results from the microservice. This allows downstream components, like the visualization factory, to find all the necessary fields. 