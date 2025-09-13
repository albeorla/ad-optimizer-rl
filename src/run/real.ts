import { argv } from "process";

function getArg(name: string, fallback?: string) {
  const p = argv.slice(2).find((a) => a.startsWith(`--${name}=`));
  return p ? p.split("=")[1] : fallback;
}

async function main() {
  const mode = (getArg("mode", "shadow") as "shadow" | "pilot");
  const dailyTarget = Number(getArg("daily-budget-target", process.env.DAILY_BUDGET_TARGET || "30"));
  const peak = getArg("peak-hours", "18-22");
  const deltaMax = Number(getArg("delta-max", "0.10"));
  const lambdaSpend = Number(getArg("lambda-spend", process.env.LAMBDA_SPEND || "0.25"));
  const lagrangeStep = Number(getArg("lagrange-step", process.env.LAGRANGE_STEP || "0.05"));
  const canaries = (getArg("canary-list", "") || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);

  console.log("Real Runner (skeleton)");
  console.log({ mode, dailyTarget, peak, deltaMax, lambdaSpend, lagrangeStep, canaries });
  console.log("TODO: wire adapters (Meta/TikTok), Shopify datasource, guardrails, and reward");
}

if (require.main === module) {
  main().catch(console.error);
}

export { main };

