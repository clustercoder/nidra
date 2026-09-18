import path from "path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Pins Turbopack's project root to this directory rather than letting it
  // infer one by walking up the tree for a lockfile (this repo also has
  // non-JS trees like nidra/, docs/, services/ alongside web/).
  turbopack: {
    root: path.join(__dirname),
  },
  experimental: {
    // Next 16.3 turned on Turbopack's persistent build cache under
    // .next/cache/turbopack by default. Vercel restores that directory
    // between deployments, and a stale/incompatible cache entry there was
    // causing `next build` to fail immediately with "Couldn't find any
    // `pages` or `app` directory" right after config load — reproduced
    // across multiple commits, always right after "Restored build cache
    // from previous deployment". Opting out makes Turbopack ignore
    // whatever Vercel restores instead of trusting it.
    turbopackFileSystemCacheForBuild: false,
  },
};

export default nextConfig;
