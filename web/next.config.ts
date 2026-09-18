import path from "path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Pins Turbopack's project root to this directory. Without it, Turbopack's
  // upward lockfile search can resolve to the repo root in Vercel's build
  // sandbox (this repo also has non-JS trees like nidra/, docs/, services/
  // alongside web/) and then fail with "Couldn't find any `pages` or `app`
  // directory" because src/app only exists under web/.
  turbopack: {
    root: path.join(__dirname),
  },
};

export default nextConfig;
