// TypeScript port of sum_engine_internal/ensemble/tome_sliders.py's
// per-axis prompt fragments + build_system_prompt. Keep byte-for-byte
// equivalent to the Python so a Python-rendered tome and a Worker-
// rendered tome from the same (triples, sliders) input are
// indistinguishable to a downstream verifier.

const SLIDER_BINS_PER_AXIS = 5;

const LENGTH_FRAGMENTS: Record<string, string> = {
  "0.1": "Use the most concise prose possible: each fact in one short sentence, no elaboration.",
  "0.3": "Be brief. Prefer short sentences. Minimal connective tissue between facts.",
  "0.5": "",
  "0.7": "Expand each fact with relevant elaboration and context.",
  "0.9": "Write expansively: develop each fact into a detailed paragraph with rich context and examples.",
};

const FORMALITY_FRAGMENTS: Record<string, string> = {
  "0.1": "Use casual, conversational tone. Contractions are encouraged. Address the reader directly.",
  "0.3": "Use a friendly, approachable tone. Light contractions are fine.",
  "0.5": "",
  "0.7": "Use formal academic register. Prefer precise vocabulary. Avoid contractions.",
  "0.9": "Write in strict academic register: passive voice where appropriate, no contractions, measured hedging language.",
};

const AUDIENCE_FRAGMENTS: Record<string, string> = {
  "0.1": "Write for a curious general reader. Avoid all domain-specific jargon. Use everyday words.",
  "0.3": "Write for an interested non-specialist. Define any technical terms inline on first use.",
  "0.5": "",
  "0.7": "Write for a domain practitioner. Use field-specific terminology freely.",
  "0.9": "Write for a domain expert. Use precise technical jargon without explanation.",
};

const PERSPECTIVE_FRAGMENTS: Record<string, string> = {
  "0.1": "Write in first person throughout ('I observed', 'we found', 'our data shows').",
  "0.3": "Write primarily in first person, with occasional third-person framing.",
  "0.5": "",
  "0.7": "Write in third-person omniscient. Avoid first-person pronouns.",
  "0.9": "Write in pure third-person omniscient narration. Use no first-person pronouns at all.",
};

const NEUTRAL_BASE =
  "You are a precise technical writer. Render the following " +
  "facts as a cohesive narrative. Do not invent facts. Do not " +
  "omit facts. Preserve every (subject, predicate, object) " +
  "relationship in the input.";

// v0.7 hardening — appended to the system prompt when any non-density
// axis is at an extreme low position. Empirically (v0.6 bench) those
// are where the LLM over-complies with register/audience directives
// at the cost of dropping source facts. See SLIDER_CONTRACT.md
// §"Catastrophic outliers".
const FACT_PRESERVATION_REINFORCEMENT =
  "CRITICAL FACT-PRESERVATION INSTRUCTION: regardless of the register, " +
  "audience, length, or perspective directives above, every input fact " +
  "(subject, predicate, object) MUST appear in your output. Rephrase " +
  "facts to fit the directives — but never omit them. If a directive " +
  "seems to require dropping a fact, find a way to include it anyway. " +
  "An output that follows the directives but loses input facts is a " +
  "FAILED render.";

function snapToBin(value: number, bins: number = SLIDER_BINS_PER_AXIS): number {
  if (value < 0 || value > 1) throw new Error(`slider value out of [0, 1]: ${value}`);
  const idx = Math.min(Math.floor(value * bins), bins - 1);
  return (idx + 0.5) / bins;
}

function lookupFragment(table: Record<string, string>, value: number, axis: string): string {
  const snapped = snapToBin(value);
  // Use one decimal place to match the Python {0.1, 0.3, 0.5, 0.7, 0.9} keys.
  const key = snapped.toFixed(1);
  if (!(key in table)) {
    throw new Error(`${axis}: snapped value ${snapped} not in 5-bin grid`);
  }
  return table[key];
}

export interface SlidersForPrompt {
  density: number;
  length: number;
  formality: number;
  audience: number;
  perspective: number;
}

export function buildSystemPrompt(sliders: SlidersForPrompt): string {
  const fragments: string[] = [];
  if (Math.abs(sliders.length - 0.5) >= 1e-6) {
    fragments.push(lookupFragment(LENGTH_FRAGMENTS, sliders.length, "length"));
  }
  if (Math.abs(sliders.formality - 0.5) >= 1e-6) {
    fragments.push(lookupFragment(FORMALITY_FRAGMENTS, sliders.formality, "formality"));
  }
  if (Math.abs(sliders.audience - 0.5) >= 1e-6) {
    fragments.push(lookupFragment(AUDIENCE_FRAGMENTS, sliders.audience, "audience"));
  }
  if (Math.abs(sliders.perspective - 0.5) >= 1e-6) {
    fragments.push(lookupFragment(PERSPECTIVE_FRAGMENTS, sliders.perspective, "perspective"));
  }
  // v0.7 hardening: extreme low positions are the empirical failure mode.
  const lowAxes: Array<keyof SlidersForPrompt> = ["length", "formality", "audience", "perspective"];
  if (lowAxes.some((axis) => sliders[axis] <= 0.3 + 1e-6)) {
    fragments.push(FACT_PRESERVATION_REINFORCEMENT);
  }
  return fragments.length === 0 ? NEUTRAL_BASE : NEUTRAL_BASE + " " + fragments.join(" ");
}

export function requiresExtrapolator(sliders: SlidersForPrompt): boolean {
  // Mirror of Python TomeSliders.requires_extrapolator.
  return !(
    sliders.length === 0.5 &&
    sliders.formality === 0.5 &&
    sliders.audience === 0.5 &&
    sliders.perspective === 0.5
  );
}

export function applyDensity(triples: Array<[string, string, string]>, density: number): Array<[string, string, string]> {
  // Mirror of Python apply_density: lex-sort by canonical key, keep first floor(N * density).
  if (triples.length === 0) return [];
  if (density >= 1.0) return triples.slice().sort((a, b) => keyOf(a).localeCompare(keyOf(b)));
  if (density <= 0.0) return [];
  const sorted = triples.slice().sort((a, b) => keyOf(a).localeCompare(keyOf(b)));
  const k = Math.floor(sorted.length * density);
  return sorted.slice(0, k);
}

function keyOf(t: [string, string, string]): string {
  return `${t[0]}||${t[1]}||${t[2]}`;
}

export function deterministicTome(triples: Array<[string, string, string]>): string {
  if (triples.length === 0) return "";
  return triples.map(([s, p, o]) => `The ${s} ${p} ${o}.`).join(" ");
}
