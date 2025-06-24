#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

// Auto-generate field mappings from layer configuration
async function generateFieldMappings() {
  console.log('🔧 Generating Comprehensive Field Mappings...\n');
  
  try {
    // Read the layers configuration
    const layersPath = path.join(__dirname, '../config/layers.ts');
    const layersContent = fs.readFileSync(layersPath, 'utf8');
    
    // Extract all fields with their aliases
    const fieldRegex = /{\s*"name":\s*"([^"]+)",\s*"type":\s*"[^"]+",\s*"alias":\s*"([^"]+)"/g;
    const fields = [];
    let match;
    
    while ((match = fieldRegex.exec(layersContent)) !== null) {
      const [, name, alias] = match;
      
      // Skip system fields
      if (!['OBJECTID', 'Shape__Area', 'Shape__Length', 'CreationDate', 'Creator', 'EditDate', 'Editor', 'thematic_value'].includes(name)) {
        fields.push({ name, alias });
      }
    }
    
    console.log(`📊 Found ${fields.length} fields to map`);
    
    // Generate concept mapping keywords
    const conceptKeywords = new Map();
    
    fields.forEach(field => {
      const alias = field.alias.toLowerCase();
      const keywords = [];
      
      // Extract meaningful terms from alias
      const terms = alias.split(/[\s,\-\(\)]+/).filter(term => term.length > 2);
      
      // Add original terms
      keywords.push(...terms);
      
      // Add brand-specific terms
      if (alias.includes('nike')) keywords.push('nike');
      if (alias.includes('adidas')) keywords.push('adidas');
      if (alias.includes('jordan')) keywords.push('jordan');
      if (alias.includes('converse')) keywords.push('converse');
      if (alias.includes('puma')) keywords.push('puma');
      if (alias.includes('reebok')) keywords.push('reebok');
      if (alias.includes('new balance')) keywords.push('new balance');
      if (alias.includes('asics')) keywords.push('asics');
      if (alias.includes('skechers')) keywords.push('skechers');
      
      // Add sports terms
      if (alias.includes('nba')) keywords.push('nba', 'basketball');
      if (alias.includes('nfl')) keywords.push('nfl', 'football');
      if (alias.includes('mlb')) keywords.push('mlb', 'baseball');
      if (alias.includes('nhl')) keywords.push('nhl', 'hockey');
      if (alias.includes('running')) keywords.push('running', 'jogging');
      if (alias.includes('yoga')) keywords.push('yoga');
      if (alias.includes('weight lifting')) keywords.push('weight lifting', 'weightlifting', 'gym');
      
      // Add demographic terms
      if (alias.includes('population')) keywords.push('population', 'people', 'residents');
      if (alias.includes('income')) keywords.push('income', 'earnings', 'salary');
      if (alias.includes('age')) keywords.push('age', 'demographic');
      if (alias.includes('race') || alias.includes('ethnicity')) keywords.push('race', 'ethnicity', 'diversity');
      
      // Add retail terms
      if (alias.includes('dick')) keywords.push('dicks', 'dick\'s sporting goods');
      if (alias.includes('foot locker')) keywords.push('foot locker', 'footlocker');
      
      // Store unique keywords
      conceptKeywords.set(field.name, [...new Set(keywords)]);
    });
    
    // Generate concept mapping code
    const conceptMappingCode = `// AUTO-GENERATED FIELD MAPPINGS
// Generated: ${new Date().toISOString()}
// DO NOT EDIT MANUALLY - Use scripts/generate-field-mappings.js

import { baseLayerConfigs } from '../config/layers';

export function buildFieldKeywordsFromLayers(): Map<string, string[]> {
  const fieldKeywords = new Map<string, string[]>();
  
  // Auto-generated mappings from layer configuration
${Array.from(conceptKeywords.entries()).map(([fieldName, keywords]) => 
  `  fieldKeywords.set('${fieldName}', ${JSON.stringify(keywords)});`
).join('\n')}
  
  return fieldKeywords;
}

// Legacy manual mappings (for backward compatibility)
export const LAYER_KEYWORDS = {
  athleticShoePurchases: [
    'athletic shoe', 'athletic shoes', 'shoes', 'sneakers', 'footwear',
    'nike', 'adidas', 'jordan', 'converse', 'puma', 'reebok', 'new balance', 'asics', 'skechers'
  ],
  sportsParticipation: [
    'sports', 'athletic', 'exercise', 'participation', 'activity',
    'running', 'jogging', 'yoga', 'weight lifting', 'gym', 'fitness'
  ],
  sportsFandom: [
    'sports fan', 'fan', 'nba', 'nfl', 'mlb', 'nhl', 'soccer', 'basketball', 'football', 'baseball', 'hockey'
  ],
  demographics: [
    'population', 'people', 'residents', 'demographics', 'age', 'race', 'ethnicity', 'income', 'household'
  ],
  retail: [
    'retail', 'shopping', 'store', 'dick\'s sporting goods', 'foot locker', 'outlet', 'mall'
  ]
};

export function conceptMapping(query: string, layerConfigs = baseLayerConfigs) {
  const fieldKeywords = buildFieldKeywordsFromLayers();
  const lowerQuery = query.toLowerCase();
  const matchedFields: string[] = [];
  
  // Check auto-generated mappings
  for (const [fieldName, keywords] of fieldKeywords.entries()) {
    if (keywords.some(keyword => lowerQuery.includes(keyword.toLowerCase()))) {
      matchedFields.push(fieldName);
    }
  }
  
  // Check legacy mappings for backward compatibility
  for (const [category, keywords] of Object.entries(LAYER_KEYWORDS)) {
    if (keywords.some(keyword => lowerQuery.includes(keyword.toLowerCase()))) {
      // Find fields that match this category
      for (const [fieldName, fieldKeywords] of fieldKeywords.entries()) {
        if (fieldKeywords.some(fk => keywords.includes(fk.toLowerCase()))) {
          matchedFields.push(fieldName);
        }
      }
    }
  }
  
  return [...new Set(matchedFields)]; // Remove duplicates
}`;
    
    // Generate query analyzer mappings
    const queryAnalyzerMappings = new Map();
    
    fields.forEach(field => {
      const alias = field.alias.toLowerCase();
      
      // Extract brand names and terms
      if (alias.includes('nike')) queryAnalyzerMappings.set('nike', field.name);
      if (alias.includes('adidas')) queryAnalyzerMappings.set('adidas', field.name);
      if (alias.includes('jordan')) queryAnalyzerMappings.set('jordan', field.name);
      if (alias.includes('converse')) queryAnalyzerMappings.set('converse', field.name);
      if (alias.includes('puma')) queryAnalyzerMappings.set('puma', field.name);
      if (alias.includes('reebok')) queryAnalyzerMappings.set('reebok', field.name);
      if (alias.includes('new balance')) queryAnalyzerMappings.set('new balance', field.name);
      if (alias.includes('asics')) queryAnalyzerMappings.set('asics', field.name);
      if (alias.includes('skechers')) queryAnalyzerMappings.set('skechers', field.name);
      
      // Sports terms
      if (alias.includes('nba')) queryAnalyzerMappings.set('nba', field.name);
      if (alias.includes('nfl')) queryAnalyzerMappings.set('nfl', field.name);
      if (alias.includes('mlb')) queryAnalyzerMappings.set('mlb', field.name);
      if (alias.includes('nhl')) queryAnalyzerMappings.set('nhl', field.name);
      if (alias.includes('running')) queryAnalyzerMappings.set('running', field.name);
      if (alias.includes('yoga')) queryAnalyzerMappings.set('yoga', field.name);
      
      // Retail terms
      if (alias.includes('dick')) queryAnalyzerMappings.set('dicks', field.name);
      if (alias.includes('foot locker')) queryAnalyzerMappings.set('foot locker', field.name);
    });
    
    const queryAnalyzerCode = `// AUTO-GENERATED QUERY ANALYZER MAPPINGS
// Generated: ${new Date().toISOString()}
// DO NOT EDIT MANUALLY - Use scripts/generate-field-mappings.js

export function buildFieldNameMap(): Map<string, string> {
  const fieldNameMap = new Map<string, string>();
  
  // Auto-generated brand and term mappings
${Array.from(queryAnalyzerMappings.entries()).map(([term, fieldName]) => 
  `  fieldNameMap.set('${term}', '${fieldName}');`
).join('\n')}
  
  return fieldNameMap;
}

export function extractMeaningfulTerms(text: string): string[] {
  const terms = text.toLowerCase()
    .split(/[\\s,\\-\\(\\)]+/)
    .filter(term => term.length > 2)
    .filter(term => !['the', 'and', 'for', 'with', 'from', 'last', 'months', 'years'].includes(term));
  
  return [...new Set(terms)];
}

export function analyzeQuery(query: string): { primaryTerms: string[], mappedFields: string[] } {
  const fieldNameMap = buildFieldNameMap();
  const lowerQuery = query.toLowerCase();
  const primaryTerms = extractMeaningfulTerms(query);
  const mappedFields: string[] = [];
  
  // Check direct mappings
  for (const [term, fieldName] of fieldNameMap.entries()) {
    if (lowerQuery.includes(term)) {
      mappedFields.push(fieldName);
    }
  }
  
  return {
    primaryTerms,
    mappedFields: [...new Set(mappedFields)]
  };
}`;
    
    // Write the generated files
    const conceptMappingPath = path.join(__dirname, '../lib/concept-mapping-generated.ts');
    fs.writeFileSync(conceptMappingPath, conceptMappingCode);
    
    const queryAnalyzerPath = path.join(__dirname, '../lib/query-analyzer-generated.ts');
    fs.writeFileSync(queryAnalyzerPath, queryAnalyzerCode);
    
    console.log(`✅ Generated concept mapping: ${conceptMappingPath}`);
    console.log(`✅ Generated query analyzer: ${queryAnalyzerPath}`);
    
    // Create integration script
    const integrationScript = `#!/usr/bin/env node

// Integration script to replace manual mappings with auto-generated ones
const fs = require('fs');
const path = require('path');

console.log('🔄 Integrating auto-generated field mappings...');

// Backup existing files
const conceptMappingPath = path.join(__dirname, '../lib/concept-mapping.ts');
const queryAnalyzerPath = path.join(__dirname, '../lib/query-analyzer.ts');

if (fs.existsSync(conceptMappingPath)) {
  fs.copyFileSync(conceptMappingPath, conceptMappingPath + '.backup');
  console.log('📋 Backed up concept-mapping.ts');
}

if (fs.existsSync(queryAnalyzerPath)) {
  fs.copyFileSync(queryAnalyzerPath, queryAnalyzerPath + '.backup');
  console.log('📋 Backed up query-analyzer.ts');
}

// Replace with generated versions
fs.copyFileSync(
  path.join(__dirname, '../lib/concept-mapping-generated.ts'),
  conceptMappingPath
);

fs.copyFileSync(
  path.join(__dirname, '../lib/query-analyzer-generated.ts'),
  queryAnalyzerPath
);

console.log('✅ Integrated auto-generated mappings');
console.log('🧪 Run npm test to verify everything works');
`;
    
    const integrationPath = path.join(__dirname, 'integrate-field-mappings.js');
    fs.writeFileSync(integrationPath, integrationScript);
    fs.chmodSync(integrationPath, '755');
    
    console.log(`✅ Generated integration script: ${integrationPath}`);
    
    // Generate summary report
    const report = `# Auto-Generated Field Mappings Summary

Generated: ${new Date().toISOString()}

## Statistics
- Total Fields Mapped: ${fields.length}
- Concept Keywords Generated: ${conceptKeywords.size}
- Query Analyzer Mappings: ${queryAnalyzerMappings.size}

## Coverage Analysis
- Brand Fields: ${fields.filter(f => f.alias.toLowerCase().includes('nike') || f.alias.toLowerCase().includes('adidas')).length}
- Sports Fields: ${fields.filter(f => f.alias.toLowerCase().includes('nba') || f.alias.toLowerCase().includes('running')).length}
- Demographic Fields: ${fields.filter(f => f.alias.toLowerCase().includes('population') || f.alias.toLowerCase().includes('income')).length}

## Next Steps
1. Run: \`node scripts/integrate-field-mappings.js\` to replace manual mappings
2. Run: \`node scripts/validate-queries.js\` to test coverage
3. Test critical queries in the UI
4. Deploy to production

## Files Generated
- lib/concept-mapping-generated.ts
- lib/query-analyzer-generated.ts
- scripts/integrate-field-mappings.js
`;
    
    fs.writeFileSync('field-mappings-report.md', report);
    console.log('\n📄 Summary report saved to: field-mappings-report.md');
    
    console.log('\n🎯 NEXT STEPS:');
    console.log('1. Run: node scripts/integrate-field-mappings.js');
    console.log('2. Run: node scripts/validate-queries.js');
    console.log('3. Test queries in the UI');
    console.log('4. Deploy to production');
    
  } catch (error) {
    console.error('❌ Field mapping generation failed:', error.message);
    process.exit(1);
  }
}

// Run generation
generateFieldMappings(); 