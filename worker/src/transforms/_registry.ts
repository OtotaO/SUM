// Transform registry — single dispatch surface for POST /api/transform.
//
// Mirrors sum_engine_internal/transforms/__init__.py.

import type { Transform } from "./_base";
import { SLIDER_TRANSFORM } from "./slider";

const REGISTRY: Map<string, Transform> = new Map();

function register(t: Transform): void {
  const existing = REGISTRY.get(t.name);
  if (existing && existing !== t) {
    throw new Error(
      `transform name ${JSON.stringify(t.name)} already registered to a ` +
        `different instance; registry is fixed at module load time.`,
    );
  }
  REGISTRY.set(t.name, t);
}

// Auto-register on module import. Order matters only for
// tie-breaking in listTransforms() (which sorts alphabetically).
register(SLIDER_TRANSFORM);

export function getTransform(name: string): Transform {
  const t = REGISTRY.get(name);
  if (!t) {
    throw new Error(
      `unknown transform ${JSON.stringify(name)}; known: ${
        listTransforms().join(", ")
      }`,
    );
  }
  return t;
}

export function listTransforms(): string[] {
  return [...REGISTRY.keys()].sort();
}

export function hasTransform(name: string): boolean {
  return REGISTRY.has(name);
}
