import { argv } from "process";
import {
  applyGuardrails,
  GuardrailConfig,
  GuardrailContext,
  Mode,
} from "../execution/guardrails";
/**
 * Real runner skeleton with guardrails. In pilot mode, perform the minimal
 * computation to clamp hourly budgets and print would-apply changes. No writes yet.
 *
 * NOTE: When wiring real adapters, ensure the state encoding matches the simulator
 * and that guardrails run before any platform writes.
 */

function getArg(name: string, fallback?: string) {
  const p = argv.slice(2).find((a) => a.startsWith(`--${name}=`));
  return p ? p.split("=")[1] : fallback;
}

async function main() {
  const mode = getArg("mode", "shadow") as Mode;
  const dailyTarget = Number(
    getArg("daily-budget-target", process.env.DAILY_BUDGET_TARGET || "30"),
  );
  const peak = getArg("peak-hours", "18-22");
  const deltaMax = Number(getArg("delta-max", "0.10"));
  const lambdaSpend = Number(
    getArg("lambda-spend", process.env.LAMBDA_SPEND || "0.25"),
  );
  const lagrangeStep = Number(
    getArg("lagrange-step", process.env.LAGRANGE_STEP || "0.05"),
  );
  const minHourly = Number(getArg("min-hourly", "0.5"));
  const maxHourly = Number(getArg("max-hourly", "3.0"));
  const projected = Number(getArg("projected-daily-spend", "0"));
  const currentHourly = Number(getArg("current-hourly", "1.0"));
  const proposedHourly = Number(getArg("proposed-hourly", "1.0"));
  const currentHour = Number(
    getArg("current-hour", String(new Date().getHours())),
  );
  const canaries = (getArg("canary-list", "") || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);

  console.log("Real Runner (skeleton)");
  console.log({
    mode,
    dailyTarget,
    peak,
    deltaMax,
    lambdaSpend,
    lagrangeStep,
    minHourly,
    maxHourly,
    canaries,
  });

  // Minimal guardrails application: clamp a proposed hourly budget under daily cap
  const cfg: GuardrailConfig = {
    dailyBudgetTarget: dailyTarget,
    deltaMax,
    minHourly,
    maxHourly,
  };
  const ctx: GuardrailContext = {
    currentHour,
    projectedDailySpend: projected,
    trailingHoursWithoutConversions: Number(getArg("no-conv-hours", "0")),
    trailingROAS: Number(getArg("trailing-roas", "2.0")),
  };

  if (mode === "pilot") {
    const res = applyGuardrails({
      cfg,
      ctx,
      currentHourlyBudget: currentHourly,
      proposedHourlyBudget: proposedHourly,
    });
    console.log("Guardrails:", {
      currentHourly,
      proposedHourly,
      projectedDailySpend: projected,
      allowedBudget: res.allowedBudget,
      reasons: res.reasons,
    });
    console.log(
      "NOTE: In pilot mode, the allowedBudget should be applied per canary before writes.",
    );
  } else {
    console.log(
      "Shadow mode: computing decisions only; no writes. You can still test guardrails with flags:",
    );
    console.log(
      "  --current-hourly --proposed-hourly --projected-daily-spend --min-hourly --max-hourly",
    );
  }

  console.log(
    "TODO: wire adapters (Meta/TikTok), Shopify datasource, guardrails application to agent actions, and reward",
  );
}

if (require.main === module) {
  main().catch(console.error);
}

export { main };
