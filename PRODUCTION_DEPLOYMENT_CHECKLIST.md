# 🚀 PRODUCTION DEPLOYMENT CHECKLIST

## Pre-Deployment Validation (MANDATORY)

### ✅ Step 1: Generate Complete Field Mappings
```bash
node scripts/generate-field-mappings.js
```
- [ ] Auto-generates mappings for ALL fields in config/layers.ts
- [ ] Creates concept-mapping-generated.ts with 100% coverage
- [ ] Creates query-analyzer-generated.ts with complete brand/term mappings
- [ ] Generates field-mappings-report.md with statistics

### ✅ Step 2: Integrate Auto-Generated Mappings
```bash
node scripts/integrate-field-mappings.js
```
- [ ] Backs up existing concept-mapping.ts and query-analyzer.ts
- [ ] Replaces manual mappings with auto-generated ones
- [ ] Ensures 100% field coverage from layer configuration

### ✅ Step 3: Validate All Queries
```bash
node scripts/validate-queries.js
```
- [ ] Tests field mapping coverage (must be 80%+)
- [ ] Tests critical brand queries (Nike, Adidas, Jordan, Converse)
- [ ] Generates query-validation-report.md
- [ ] FAILS if any critical queries don't work

### ✅ Step 4: Build Verification
```bash
npm run build
```
- [ ] Ensures no TypeScript compilation errors
- [ ] Verifies all imports resolve correctly
- [ ] Confirms production build succeeds

### ✅ Step 5: UI Functionality Test
**Manual Testing Required:**
- [ ] IQbuilder has 4 buttons (quickstartIQ, infographIQ, Target, Persona)
- [ ] Target selector shows all brands with correct icons
- [ ] Persona selector shows all personas with descriptions
- [ ] Test these queries manually:
  - [ ] "How do Jordan sales compare to Converse sales?"
  - [ ] "Compare Nike vs Adidas athletic shoe purchases across regions"
  - [ ] "Show me the top 10 areas with highest Nike athletic shoe purchases"
  - [ ] "Rank areas by Adidas athletic shoe sales"

## Automated Production Check
```bash
npm run production-check
```
This runs all steps 1-4 automatically and fails if any issues are found.

## Critical Success Criteria

### 🎯 Field Mapping Coverage
- **Target**: 95%+ coverage of all fields in config/layers.ts
- **Critical**: All brand fields (Nike, Adidas, Jordan, Converse, etc.) must be mapped
- **Critical**: All sports fields (NBA, NFL, Running, Yoga, etc.) must be mapped
- **Critical**: All demographic fields (Population, Income, Race, etc.) must be mapped

### 🎯 Query Success Rate
- **Target**: 100% of critical brand queries must work
- **Critical**: Jordan vs Converse comparison must work
- **Critical**: Nike vs Adidas comparison must work
- **Critical**: All queries in ANALYSIS_CATEGORIES must find field matches

### 🎯 UI Functionality
- **Critical**: All 4 IQbuilder buttons must be functional
- **Critical**: Target variable selection must work for all brands
- **Critical**: Persona selection must work for all personas
- **Critical**: No JavaScript console errors

## Deployment Blockers

### 🚨 BLOCK DEPLOYMENT IF:
1. Field mapping coverage < 80%
2. Any critical brand query fails
3. Build fails with TypeScript errors
4. IQbuilder missing buttons or functionality
5. Console errors during query processing

## Post-Deployment Verification

### 🔍 Smoke Tests (Run in Production)
1. Load the application successfully
2. Test Jordan vs Converse query
3. Test Nike vs Adidas query
4. Verify all 4 IQbuilder buttons work
5. Check browser console for errors

### 📊 Monitoring
- Monitor query success rates
- Track field mapping coverage
- Watch for JavaScript errors
- Monitor microservice response times

## Rollback Plan

### 🔄 If Issues Found in Production:
1. Revert to previous stable version
2. Restore backup field mappings
3. Run validation scripts to confirm rollback
4. Investigate and fix issues in development
5. Re-run full checklist before next deployment

## Team Responsibilities

### 🧑‍💻 Developer Checklist:
- [ ] Run all validation scripts
- [ ] Fix any failing tests
- [ ] Test UI functionality manually
- [ ] Verify no console errors
- [ ] Document any changes made

### 🧪 QA Checklist:
- [ ] Verify all critical queries work
- [ ] Test edge cases and error scenarios
- [ ] Confirm UI matches expected behavior
- [ ] Validate performance is acceptable

### 🚀 DevOps Checklist:
- [ ] Run automated production check
- [ ] Verify build artifacts are correct
- [ ] Confirm deployment environment is ready
- [ ] Have rollback plan ready

## Success Metrics

### 📈 Deployment is Successful When:
- ✅ Field mapping coverage ≥ 95%
- ✅ All critical queries return results
- ✅ UI has full functionality (4 buttons)
- ✅ No JavaScript console errors
- ✅ Build completes without errors
- ✅ Smoke tests pass in production

---

**⚠️ IMPORTANT: This checklist must be completed EVERY TIME before production deployment. No exceptions.** 