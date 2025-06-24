#!/usr/bin/env node

/**
 * Nike vs Adidas Query Flow Test
 * 
 * This test traces the exact flow of how "Compare Nike vs Adidas" gets processed
 * to identify where the data is being lost.
 */

const query = "Compare Nike vs Adidas athletic shoe purchases across regions";

console.log('🔍 NIKE VS ADIDAS QUERY FLOW TRACE');
console.log('===================================\n');

console.log('Query:', query);

// Step 1: Test concept mapping
console.log('\n📋 STEP 1: CONCEPT MAPPING');
console.log('Testing if Nike/Adidas fields are found...');

// Mock the concept mapping logic
const lowerQuery = query.toLowerCase();
console.log('Lower query:', lowerQuery);

// Check for brand keywords
const hasNike = lowerQuery.includes('nike');
const hasAdidas = lowerQuery.includes('adidas');
const hasCompare = lowerQuery.includes('compare') || lowerQuery.includes('vs');

console.log(`Contains "nike": ${hasNike}`);
console.log(`Contains "adidas": ${hasAdidas}`);
console.log(`Contains comparison: ${hasCompare}`);

if (hasNike && hasAdidas && hasCompare) {
  console.log('✅ Query should match both Nike and Adidas fields');
  console.log('Expected matched fields: MP30034A_B (Nike), MP30029A_B (Adidas)');
} else {
  console.log('❌ Query not matching expected pattern');
}

// Step 2: Test field name mapping
console.log('\n📋 STEP 2: FIELD NAME MAPPING');
console.log('Testing if brand names map to field codes...');

// Mock field name mapping logic
const mockFieldNameMap = {
  'nike': 'MP30034A_B',
  'adidas': 'MP30029A_B',
  'jordan': 'MP30032A_B',
  'converse': 'MP30031A_B'
};

const expectedFields = ['nike', 'adidas'];
const mappedFields = expectedFields.map(brand => mockFieldNameMap[brand]);

console.log('Expected brands:', expectedFields);
console.log('Mapped fields:', mappedFields);

// Step 3: Test microservice request
console.log('\n📋 STEP 3: MICROSERVICE REQUEST');
console.log('Testing if fields are included in request...');

const mockAnalysisResult = {
  queryType: 'correlation',
  visualizationStrategy: 'correlation',
  relevantLayers: ['athleticShoePurchases'],
  targetVariable: 'MP30034A_B',
  relevantFields: mappedFields,
  confidence: 0.85
};

console.log('Mock analysis result:');
console.log('- Query type:', mockAnalysisResult.queryType);
console.log('- Target variable:', mockAnalysisResult.targetVariable);
console.log('- Relevant fields:', mockAnalysisResult.relevantFields);

// Step 4: Check if both fields would be sent to microservice
console.log('\n📋 STEP 4: MICROSERVICE PAYLOAD');
const hasNikeField = mockAnalysisResult.relevantFields.includes('MP30034A_B');
const hasAdidasField = mockAnalysisResult.relevantFields.includes('MP30029A_B');

console.log(`Nike field in payload: ${hasNikeField}`);
console.log(`Adidas field in payload: ${hasAdidasField}`);

if (hasNikeField && hasAdidasField) {
  console.log('✅ Both Nike and Adidas fields would be sent to microservice');
} else {
  console.log('❌ Missing fields in microservice payload');
  console.log('This is likely where the issue occurs!');
}

// Step 5: Expected microservice response
console.log('\n📋 STEP 5: EXPECTED MICROSERVICE RESPONSE');
console.log('Microservice should return data with both fields:');
console.log('- Features should contain MP30034A_B values (Nike data)');
console.log('- Features should contain MP30029A_B values (Adidas data)');
console.log('- Claude should receive both datasets for comparison');

console.log('\n🎯 DEBUGGING CHECKLIST:');
console.log('1. Check browser console for concept mapping debug logs');
console.log('2. Verify field name mapping logs show nike->MP30034A_B, adidas->MP30029A_B');
console.log('3. Check microservice request payload includes both fields');
console.log('4. Verify microservice response contains both Nike and Adidas data');
console.log('5. Confirm Claude receives features with both MP30034A_B and MP30029A_B');

console.log('\n💡 LIKELY ROOT CAUSE:');
console.log('If concept mapping is working but Claude says "no data", the issue is likely:');
console.log('- Concept mapping not finding Nike/Adidas fields (check FIELD_KEYWORDS)');
console.log('- Query analyzer not mapping brand names to field codes');
console.log('- Microservice not returning data for the requested fields');
console.log('- Data harmonization step losing the Nike/Adidas data'); 