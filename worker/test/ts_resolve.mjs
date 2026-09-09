// Match the Worker's bundler resolution for extensionless local TS imports.
// Node strips types natively; this test-only hook adds no transpiler dependency.
import { extname } from "node:path";

export async function resolve(specifier, context, nextResolve) {
  if (specifier.startsWith(".") && !extname(specifier)) {
    return nextResolve(`${specifier}.ts`, context);
  }
  return nextResolve(specifier, context);
}
