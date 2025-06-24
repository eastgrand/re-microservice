#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

// Simple validation script to test query field mappings
async function validateQueries() {
  console.log('🔍 Starting Comprehensive Query Validation...\n');
  
  try {
    // Read the layers configuration
    const layersPath = path.join(__dirname, '../config/layers.ts');
    const layersContent = fs.readFileSync(layersPath, 'utf8');
    
    // Read the chat constants
    const chatConstantsPath = path.join(__dirname, '../components/chat/chat-constants.ts');
    const chatConstantsContent = fs.readFileSync(chatConstantsPath, 'utf8');
    
    // Read the concept mapping
    const conceptMappingPath = path.join(__dirname, '../lib/concept-mapping.ts');
    const conceptMappingContent = fs.readFileSync(conceptMappingPath, 'utf8');
    
    // Read the query analyzer
    const queryAnalyzerPath = path.join(__dirname, '../lib/query-analyzer.ts');
    const queryAnalyzerContent = fs.readFileSync(queryAnalyzerPath, 'utf8');
    
    // Extract all field names from layers
    const fieldMatches = layersContent.match(/"alias":\s*"([^"]+)"/g) || [];
    const allFields = fieldMatches.map(match => match.match(/"([^"]+)"/)[1]);
    
    console.log(`📊 Found ${allFields.length} fields in layer configuration`);
    
    // Extract queries from chat constants
    const queryMatches = chatConstantsContent.match(/'([^']+)'/g) || [];
    const allQueries = queryMatches
      .map(match => match.slice(1, -1))
      .filter(query => query.length > 20); // Filter out short non-query strings
    
    console.log(`📝 Found ${allQueries.length} queries in chat constants`);
    
    // Check field mapping coverage
    let mappedFields = 0;
    let unmappedFields = [];
    
    allFields.forEach(field => {
      const lowerField = field.toLowerCase();
      
      // Check if field is mentioned in concept mapping or query analyzer
      const isMappedInConcept = conceptMappingContent.toLowerCase().includes(lowerField);
      const isMappedInAnalyzer = queryAnalyzerContent.toLowerCase().includes(lowerField);
      
      if (isMappedInConcept || isMappedInAnalyzer) {
        mappedFields++;
      } else {
        unmappedFields.push(field);
      }
    });
    
    const coverage = (mappedFields / allFields.length) * 100;
    
    console.log(`\n📈 FIELD MAPPING COVERAGE: ${coverage.toFixed(1)}%`);
    console.log(`   ✅ Mapped: ${mappedFields} fields`);
    console.log(`   ❌ Unmapped: ${unmappedFields.length} fields`);
    
    // Test critical brand queries
    console.log('\n🧪 TESTING CRITICAL BRAND QUERIES...');
    
    const criticalQueries = [
      'How do Jordan sales compare to Converse sales?',
      'Compare Nike vs Adidas athletic shoe purchases across regions',
      'Show me the top 10 areas with highest Nike athletic shoe purchases',
      'Rank areas by Adidas athletic shoe sales'
    ];
    
    let passedQueries = 0;
    let failedQueries = [];
    
    criticalQueries.forEach(query => {
      const lowerQuery = query.toLowerCase();
      let hasFieldMatches = false;
      
      // Check if query terms are mapped
      const brandTerms = ['nike', 'adidas', 'jordan', 'converse'];
      const queryBrands = brandTerms.filter(brand => lowerQuery.includes(brand));
      
      if (queryBrands.length > 0) {
        // Check if these brands are mapped in our systems
        const mappedBrands = queryBrands.filter(brand => {
          return conceptMappingContent.toLowerCase().includes(brand) ||
                 queryAnalyzerContent.toLowerCase().includes(brand);
        });
        
        if (mappedBrands.length === queryBrands.length) {
          hasFieldMatches = true;
        }
      }
      
      if (hasFieldMatches) {
        passedQueries++;
        console.log(`   ✅ "${query}"`);
      } else {
        failedQueries.push(query);
        console.log(`   ❌ "${query}"`);
      }
    });
    
    // Generate report
    console.log('\n📋 VALIDATION REPORT');
    console.log('='.repeat(50));
    
    if (coverage < 80) {
      console.log('🚨 CRITICAL: Field mapping coverage below 80%');
    }
    
    if (failedQueries.length > 0) {
      console.log('🚨 CRITICAL: Brand queries failing');
      failedQueries.forEach(query => {
        console.log(`   - "${query}"`);
      });
    }
    
    if (unmappedFields.length > 0) {
      console.log('\n📝 TOP UNMAPPED FIELDS (need immediate attention):');
      unmappedFields.slice(0, 10).forEach(field => {
        console.log(`   - ${field}`);
      });
    }
    
    console.log('\n💡 RECOMMENDATIONS:');
    console.log('1. Implement systematic field mapping from layer config');
    console.log('2. Add missing brand mappings to concept-mapping.ts');
    console.log('3. Add missing brand mappings to query-analyzer.ts');
    console.log('4. Test all queries before production deployment');
    
    // Create detailed report file
    const reportContent = `# Query Validation Report
Generated: ${new Date().toISOString()}

## Summary
- Total Fields: ${allFields.length}
- Mapped Fields: ${mappedFields}
- Coverage: ${coverage.toFixed(1)}%
- Critical Queries Tested: ${criticalQueries.length}
- Passed: ${passedQueries}
- Failed: ${failedQueries.length}

## Failed Queries
${failedQueries.map(q => `- "${q}"`).join('\n')}

## Unmapped Fields
${unmappedFields.slice(0, 20).map(f => `- ${f}`).join('\n')}

## Recommendations
1. Implement systematic field mapping from layer config
2. Add missing brand mappings to concept-mapping.ts
3. Add missing brand mappings to query-analyzer.ts
4. Test all queries before production deployment
`;
    
    fs.writeFileSync('query-validation-report.md', reportContent);
    console.log('\n📄 Detailed report saved to: query-validation-report.md');
    
    // Exit with appropriate code
    if (coverage < 80 || failedQueries.length > 0) {
      console.log('\n❌ VALIDATION FAILED - Critical issues found');
      process.exit(1);
    } else {
      console.log('\n✅ VALIDATION PASSED - All critical queries working');
      process.exit(0);
    }
    
  } catch (error) {
    console.error('❌ Validation script failed:', error.message);
    process.exit(1);
  }
}

// Run validation
validateQueries(); 