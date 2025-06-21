// Export all AI personas for easy importing
export { default as strategistPersona } from './strategist';
export { default as tacticianPersona } from './tactician';
export { default as creativePersona } from './creative';
export { default as productSpecialistPersona } from './product-specialist';
export { default as customerAdvocatePersona } from './customer-advocate';

// Persona registry for dynamic loading
export const personaRegistry = {
  strategist: () => import('./strategist').then(m => m.default),
  tactician: () => import('./tactician').then(m => m.default),
  creative: () => import('./creative').then(m => m.default),
  'product-specialist': () => import('./product-specialist').then(m => m.default),
  'customer-advocate': () => import('./customer-advocate').then(m => m.default)
};

// Persona metadata for UI display
export const personaMetadata = [
  {
    id: 'strategist',
    name: 'Strategist',
    description: 'High-level market insights, competitive positioning, and long-term growth opportunities',
    icon: '🎯',
    color: 'blue'
  },
  {
    id: 'tactician',
    name: 'Tactician', 
    description: 'Operational efficiency, resource allocation, and tactical implementation',
    icon: '⚡',
    color: 'orange'
  },
  {
    id: 'creative',
    name: 'Creative',
    description: 'Innovation opportunities, emerging trends, and creative solutions',
    icon: '💡',
    color: 'purple'
  },
  {
    id: 'product-specialist',
    name: 'Product Specialist',
    description: 'Product development, feature optimization, and user experience insights',
    icon: '🛠️',
    color: 'green'
  },
  {
    id: 'customer-advocate',
    name: 'Customer Advocate',
    description: 'Customer satisfaction, experience optimization, and service improvements',
    icon: '❤️',
    color: 'red'
  }
];

// Default persona (fallback)
export const defaultPersona = 'strategist';

// Utility function to get persona by ID
export async function getPersona(personaId: string) {
  const loader = personaRegistry[personaId as keyof typeof personaRegistry];
  if (!loader) {
    console.warn(`Persona '${personaId}' not found, falling back to default`);
    return personaRegistry[defaultPersona]();
  }
  return loader();
} 