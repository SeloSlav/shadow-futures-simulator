import { useMemo, useState, useDeferredValue } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { RefreshCcw } from "lucide-react";
import { motion } from "framer-motion";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  BarChart,
  Bar,
  ReferenceLine,
} from "recharts";

// Types
interface SeriesDataPoint {
  t: number;
  MI_bits: number;
  Gini: number;
  Top1: number;
  Top10: number;
  taxRevenue?: number;
  postTaxGini?: number;
}

interface BarDataPoint {
  rank: number;
  share: number;
}

interface EffortCurvePoint {
  bin: string;
  pRewarded: number;
}

interface SimulationResult {
  series: SeriesDataPoint[];
  bars: BarDataPoint[];
  effortCurve: EffortCurvePoint[];
  totalTaxRevenue?: number;
}

interface PhasePoint {
  alpha: number;
  lambda: number;
  regime: string;
  color: string;
  mi: number;
  gini: number;
}

interface TaxRevenuePoint {
  rate: number;
  revenue: number;
  postTaxGini: number;
}

interface SimulationParams {
  T?: number;
  alpha?: number;
  lambda?: number;
  churn?: number;
  A0?: number;
  bins?: number;
  seed?: number;
  incomeTaxRate?: number;
  wealthTaxRate?: number;
}

interface Preset {
  key: string;
  name: string;
  desc: string;
  interpretation: string;
  params: {
    alpha: number;
    lambda: number;
    churn: number;
    T: number;
  };
}

interface LabeledSliderProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step: number;
  format: (value: number) => string;
}

// Edge of chaos style toy model
// Agents enter over time. Each agent has verifiable effort V (transcript).
// Rewards follow reinforcement via attachment A, modulated by effort weight lambda.
// Optional churn decays attachment mass, preventing full lock-in.
// We estimate mutual information I(V;R) via binning effort and treating reward as "ever rewarded".

function mulberry32(seed: number): () => number {
  let t = seed >>> 0;
  return function () {
    t += 0x6d2b79f5;
    let x = Math.imul(t ^ (t >>> 15), 1 | t);
    x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

function clamp(x: number, a: number, b: number): number {
  return Math.max(a, Math.min(b, x));
}

function gini(values: number[]): number {
  const n = values.length;
  if (n === 0) return 0;
  const v = [...values].sort((a, b) => a - b);
  const sum = v.reduce((s, x) => s + x, 0);
  if (sum <= 0) return 0;
  let cum = 0;
  for (let i = 0; i < n; i++) cum += (i + 1) * v[i];
  return (2 * cum) / (n * sum) - (n + 1) / n;
}

function mutualInformationBinned({
  effort,
  rewarded,
  bins = 10,
}: {
  effort: number[];
  rewarded: number[];
  bins?: number;
}): number {
  // effort: array in [0,1]
  // rewarded: array of 0/1 indicating whether agent got at least one reward
  const n = effort.length;
  if (n === 0) return 0;

  const counts = Array.from({ length: bins }, () => ({ n: 0, r1: 0 }));
  for (let i = 0; i < n; i++) {
    const b = clamp(Math.floor(effort[i] * bins), 0, bins - 1);
    counts[b].n += 1;
    counts[b].r1 += rewarded[i] ? 1 : 0;
  }

  const pR1 = rewarded.reduce((s, x) => s + (x ? 1 : 0), 0) / n;
  const pR0 = 1 - pR1;

  let mi = 0;
  for (let b = 0; b < bins; b++) {
    const nb = counts[b].n;
    if (nb === 0) continue;
    const pV = nb / n;
    const pR1gV = counts[b].r1 / nb;
    const pR0gV = 1 - pR1gV;

    // I(V;R) = sum_v p(v) sum_r p(r|v) log( p(r|v) / p(r) )
    if (pR1gV > 0 && pR1 > 0) mi += pV * pR1gV * Math.log(pR1gV / pR1);
    if (pR0gV > 0 && pR0 > 0) mi += pV * pR0gV * Math.log(pR0gV / pR0);
  }

  // Convert nats to bits
  return mi / Math.log(2);
}

function softmaxPick(weights: number[], rng: () => number): number {
  // weights are nonnegative
  let total = 0;
  for (let i = 0; i < weights.length; i++) total += weights[i];
  if (total <= 0) return 0;
  let r = rng() * total;
  for (let i = 0; i < weights.length; i++) {
    r -= weights[i];
    if (r <= 0) return i;
  }
  return weights.length - 1;
}

function simulate({
  T = 400,
  alpha = 1.0,
  lambda = 0.0,
  churn = 0.0,
  A0 = 1.0,
  bins = 10,
  seed = 7,
  incomeTaxRate = 0.0,
  wealthTaxRate = 0.0,
}: SimulationParams): SimulationResult {
  const rng = mulberry32(seed);

  // State arrays
  const effort: number[] = [];
  const A: number[] = [];
  const gotAnyReward: number[] = [];
  const rewardCount: number[] = [];
  const wealth: number[] = []; // Post-tax accumulated wealth

  // Track time series for plotting
  const series: SeriesDataPoint[] = [];
  let totalTaxRevenue = 0;

  for (let t = 0; t < T; t++) {
    // Entry
    const v = rng(); // verifiable effort transcript in [0,1]
    effort.push(v);
    A.push(A0);
    gotAnyReward.push(0);
    rewardCount.push(0);
    wealth.push(0);

    // Churn: decay attachments slightly each step (prevents full lock-in)
    if (churn > 0) {
      for (let i = 0; i < A.length; i++) A[i] *= 1 - churn;
    }

    // Wealth tax: applied to accumulated wealth/attachment AND accumulated wealth
    if (wealthTaxRate > 0) {
      for (let i = 0; i < A.length; i++) {
        // Tax attachment (affects allocation)
        const attachmentTax = A[i] * wealthTaxRate;
        A[i] -= attachmentTax;
        totalTaxRevenue += attachmentTax;
        
        // Tax accumulated wealth (affects inequality measurement)
        const wealthTax = wealth[i] * wealthTaxRate;
        wealth[i] -= wealthTax;
        // Note: wealth tax revenue already counted via attachment tax
      }
    }

    // Reward allocation
    // Baseline reinforcement: A_i^alpha
    // Effort modulation: multiply by exp(lambda * (v_i - meanV))
    // This keeps weights positive and gives a smooth effort channel.
    const meanV = effort.reduce((s, x) => s + x, 0) / effort.length;

    const weights = A.map((ai, i) => {
      const base = Math.pow(Math.max(ai, 1e-9), alpha);
      const mod = Math.exp(lambda * (effort[i] - meanV));
      return base * mod;
    });

    const winner = softmaxPick(weights, rng);
    const reward = 1;
    
    // Income tax on reward
    const afterTaxReward = reward * (1 - incomeTaxRate);
    const taxCollected = reward * incomeTaxRate;
    totalTaxRevenue += taxCollected;
    
    A[winner] += afterTaxReward;
    wealth[winner] += afterTaxReward;
    gotAnyReward[winner] = 1;
    rewardCount[winner] += 1;

    // Diagnostics
    const totalRewards = rewardCount.reduce((s, x) => s + x, 0);
    const shares = rewardCount.map((x) => (totalRewards > 0 ? x / totalRewards : 0));
    const top1 = [...shares].sort((a, b) => b - a)[0] ?? 0;
    const top10share = [...shares]
      .sort((a, b) => b - a)
      .slice(0, Math.max(1, Math.floor(shares.length * 0.1)))
      .reduce((s, x) => s + x, 0);

    const mi = mutualInformationBinned({ effort, rewarded: gotAnyReward, bins });
    const g = gini(rewardCount);
    const postTaxG = wealth.length > 0 && wealth.reduce((s, x) => s + x, 0) > 0 ? gini(wealth) : g;

    series.push({
      t: t + 1,
      MI_bits: mi,
      Gini: g,
      Top1: top1,
      Top10: top10share,
      taxRevenue: totalTaxRevenue,
      postTaxGini: postTaxG,
    });
  }

  // Final distributions for bars
  const totalRewards = rewardCount.reduce((s, x) => s + x, 0);
  const finalShares = rewardCount.map((x) => (totalRewards > 0 ? x / totalRewards : 0));
  const sorted = finalShares
    .map((s, i) => ({ i, s }))
    .sort((a, b) => b.s - a.s)
    .slice(0, 20);

  const bars: BarDataPoint[] = sorted.map((d, idx) => ({ rank: idx + 1, share: d.s }));

  // Effort bins vs probability of ever being rewarded
  const counts = Array.from({ length: bins }, () => ({ n: 0, r: 0 }));
  for (let i = 0; i < effort.length; i++) {
    const b = clamp(Math.floor(effort[i] * bins), 0, bins - 1);
    counts[b].n += 1;
    counts[b].r += gotAnyReward[i] ? 1 : 0;
  }
  const effortCurve: EffortCurvePoint[] = counts.map((c, b) => ({
    bin: `${b + 1}`,
    pRewarded: c.n ? c.r / c.n : 0,
  }));

  return { series, bars, effortCurve, totalTaxRevenue };
}

// Classify regime based on alpha, lambda, churn
function classifyRegime(alpha: number, lambda: number, churn: number): { name: string; color: string } {
  // Ordered: low alpha, high lambda, moderate churn
  if (alpha < 0.5 && lambda > 0.8 && churn > 0.005) {
    return { name: "Ordered", color: "#3b82f6" }; // Blue
  }
  // Periodic: moderate alpha, high lambda, low churn
  if (alpha >= 0.5 && alpha < 1.0 && lambda > 0.6 && churn < 0.01) {
    return { name: "Periodic", color: "#10b981" }; // Green
  }
  // Complex (Edge of Chaos): balanced parameters
  if (alpha >= 0.9 && alpha <= 1.2 && lambda >= 0.4 && lambda <= 0.8 && churn >= 0.005 && churn <= 0.015) {
    return { name: "Complex", color: "#f59e0b" }; // Amber
  }
  // Chaotic: high alpha, low lambda, low churn
  if (alpha > 1.2 && lambda < 0.5 && churn < 0.005) {
    return { name: "Chaotic", color: "#ef4444" }; // Red
  }
  // Transitional
  return { name: "Transitional", color: "#8b5cf6" }; // Purple
}

// Generate phase transition data
function generatePhaseTransition(
  alphaRange: [number, number],
  lambdaRange: [number, number],
  churn: number,
  resolution: number = 10
): PhasePoint[] {
  const points: PhasePoint[] = [];
  const alphaStep = (alphaRange[1] - alphaRange[0]) / resolution;
  const lambdaStep = (lambdaRange[1] - lambdaRange[0]) / resolution;

  // Generate grid row by row (top to bottom = high λ to low λ)
  // Each row: left to right = low α to high α
  for (let row = 0; row < resolution; row++) {
    const lambda = lambdaRange[1] - (row + 0.5) * lambdaStep; // Start from top (high λ)
    for (let col = 0; col < resolution; col++) {
      const alpha = alphaRange[0] + (col + 0.5) * alphaStep; // Left to right (low to high α)
      const regime = classifyRegime(alpha, lambda, churn);

      // Quick simulation to get MI and Gini - reduced T for performance
      const quickResult = simulate({ T: 100, alpha, lambda, churn, bins: 10, seed: 42 });
      const last = quickResult.series[quickResult.series.length - 1];

      points.push({
        alpha,
        lambda,
        regime: regime.name,
        color: regime.color,
        mi: last.MI_bits,
        gini: last.Gini,
      });
    }
  }
  return points;
}

// Generate tax revenue curve
function generateTaxRevenueCurve(
  alpha: number,
  lambda: number,
  churn: number,
  T: number,
  taxType: "income" | "wealth",
  maxRate: number = 0.5,
  steps: number = 10
): TaxRevenuePoint[] {
  const points: TaxRevenuePoint[] = [];
  const rateStep = maxRate / steps;
  // Use reduced T for performance when generating curves
  const curveT = Math.min(T, 300);

  for (let i = 0; i <= steps; i++) {
    const rate = i * rateStep;
    const params: SimulationParams = {
      T: curveT,
      alpha,
      lambda,
      churn,
      bins: 10,
      seed: 42,
    };
    if (taxType === "income") {
      params.incomeTaxRate = rate;
    } else {
      params.wealthTaxRate = rate;
    }

    const result = simulate(params);
    const last = result.series[result.series.length - 1];
    
    // Use Gini of reward distribution (who won how many times), not instantaneous wealth
    // This measures fairness of allocation over time, which is what taxes actually affect
    points.push({
      rate: rate * 100, // Convert to percentage
      revenue: result.totalTaxRevenue || 0,
      postTaxGini: last.Gini, // Reward distribution concentration
    });
  }
  return points;
}

function LabeledSlider({ label, value, onChange, min, max, step, format }: LabeledSliderProps) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="text-sm font-medium">{label}</div>
        <div className="text-sm text-muted">{format(value)}</div>
      </div>
      <Slider
        value={[value]}
        min={min}
        max={max}
        step={step}
        onValueChange={(v) => onChange(v[0])}
      />
    </div>
  );
}

export default function EdgeOfChaosExplorer() {
  const [alpha, setAlpha] = useState(1.0);
  const [lambda, setLambda] = useState(0.0);
  const [churn, setChurn] = useState(0.0);
  const [T, setT] = useState(200);
  const [seed, setSeed] = useState(7);
  const [activePreset, setActivePreset] = useState<Preset | null>(null);
  const [incomeTaxRate, setIncomeTaxRate] = useState(0.0);
  const [wealthTaxRate, setWealthTaxRate] = useState(0.0);

  const bins = 10;

  const presets: Preset[] = useMemo(
    () => [
      {
        key: "agrarian",
        name: "18th c. agrarian",
        desc: "Weak reinforcement, effort visible, moderate mixing",
        interpretation:
          "Production is local and scale is limited. Output mostly tracks labor input, and advantages decay over time (moderate churn). Reinforcement is weak (low α), so early luck does not permanently dominate. Because λ is high, effort transcripts remain informative: working harder reliably increases the chance of reward.",
        params: { alpha: 0.35, lambda: 1.2, churn: 0.012, T: 200 },
      },
      {
        key: "industrial",
        name: "Industrial",
        desc: "Some scale effects, effort matters, low mixing",
        interpretation:
          "Mechanization and capital introduce scale effects. Early success helps, but effort and skill still matter. Reinforcement is present but not overwhelming (α < 1). Churn is low, so firms and workers can accumulate advantage, yet effort remains partially identifiable.",
        params: { alpha: 0.8, lambda: 0.8, churn: 0.004, T: 200 },
      },
      {
        key: "platform",
        name: "High-tech platform",
        desc: "Preferential attachment, weak mixing, transcripts fade",
        interpretation:
          "Digital platforms amplify visibility and network position. Preferential attachment dominates (α > 1), so early winners gain disproportionate reach. Individual effort matters locally, but its signal decays quickly. Many agents exert similar effort yet receive radically different outcomes.",
        params: { alpha: 1.25, lambda: 0.25, churn: 0.002, T: 200 },
      },
      {
        key: "winner",
        name: "Winner-take-most",
        desc: "Strong increasing returns, near lock-in",
        interpretation:
          "Markets with extreme scale economies and low turnover. Once dominance emerges, it is rarely overturned. Reinforcement overwhelms effort, and churn is nearly absent. Work verification mainly certifies participation rather than value creation.",
        params: { alpha: 1.6, lambda: 0.15, churn: 0.0, T: 200 },
      },
      {
        key: "agentic",
        name: "Far-future agentic",
        desc: "Extreme reinforcement, pooling, low human identifiability",
        interpretation:
          "Highly automated systems where agents, models, or capital pools act at massive scale. Reinforcement is extreme and mixing negligible. Individual human effort becomes statistically irrelevant to outcomes. Attribution collapses almost entirely.",
        params: { alpha: 1.85, lambda: 0.05, churn: 0.0, T: 200 },
      },
      {
        key: "edge",
        name: "Edge regime",
        desc: "Mixed: reinforcement + mixing keeps signals alive",
        interpretation:
          "A transitional regime near the edge of chaos. Reinforcement exists but is counterbalanced by churn and meaningful effort channels. Outcomes are neither fully random nor fully locked in. This is where effort is most informative and adaptation remains possible.",
        params: { alpha: 1.05, lambda: 0.6, churn: 0.01, T: 200 },
      },
    ],
    []
  );

  const applyPreset = (p: Preset) => {
    setAlpha(p.params.alpha);
    setLambda(p.params.lambda);
    setChurn(p.params.churn);
    setT(p.params.T);
    setActivePreset(p);
    setSeed((s) => s + 1);
  };

  const { series, bars, effortCurve, totalTaxRevenue } = useMemo(() => {
    return simulate({ T, alpha, lambda, churn, bins, seed, incomeTaxRate, wealthTaxRate });
  }, [T, alpha, lambda, churn, seed, incomeTaxRate, wealthTaxRate]);

  // Use deferred values for expensive computations to prevent blocking
  const deferredChurn = useDeferredValue(churn);
  const deferredAlpha = useDeferredValue(alpha);
  const deferredLambda = useDeferredValue(lambda);
  const deferredT = useDeferredValue(T);
  
  // Track if deferred values are stale
  const isPending = churn !== deferredChurn || alpha !== deferredAlpha || lambda !== deferredLambda || T !== deferredT;

  const phaseTransitionData = useMemo(() => {
    return generatePhaseTransition([0, 2], [0, 2], deferredChurn, 10);
  }, [deferredChurn]);

  const incomeTaxCurve = useMemo(() => {
    return generateTaxRevenueCurve(deferredAlpha, deferredLambda, deferredChurn, deferredT, "income", 1.0, 15);
  }, [deferredAlpha, deferredLambda, deferredChurn, deferredT]);

  const wealthTaxCurve = useMemo(() => {
    return generateTaxRevenueCurve(deferredAlpha, deferredLambda, deferredChurn, deferredT, "wealth", 1.0, 15);
  }, [deferredAlpha, deferredLambda, deferredChurn, deferredT]);

  // Find optimal tax rates (max revenue and min concentration)
  const optimalIncomeTax = useMemo(() => {
    if (incomeTaxCurve.length === 0) return null;
    const maxRevenue = incomeTaxCurve.reduce((max, point) => point.revenue > max.revenue ? point : max, incomeTaxCurve[0]);
    const minGini = incomeTaxCurve.reduce((min, point) => point.postTaxGini < min.postTaxGini ? point : min, incomeTaxCurve[0]);
    const baseGini = incomeTaxCurve[0]?.postTaxGini || 0;
    return { 
      peakRevenueRate: maxRevenue.rate, 
      peakRevenue: maxRevenue.revenue,
      minGiniRate: minGini.rate,
      minGini: minGini.postTaxGini,
      baseGini
    };
  }, [incomeTaxCurve]);

  const optimalWealthTax = useMemo(() => {
    if (wealthTaxCurve.length === 0) return null;
    const maxRevenue = wealthTaxCurve.reduce((max, point) => point.revenue > max.revenue ? point : max, wealthTaxCurve[0]);
    const minGini = wealthTaxCurve.reduce((min, point) => point.postTaxGini < min.postTaxGini ? point : min, wealthTaxCurve[0]);
    const baseGini = wealthTaxCurve[0]?.postTaxGini || 0;
    return { 
      peakRevenueRate: maxRevenue.rate, 
      peakRevenue: maxRevenue.revenue,
      minGiniRate: minGini.rate,
      minGini: minGini.postTaxGini,
      baseGini
    };
  }, [wealthTaxCurve]);

  const last = series[series.length - 1] || { MI_bits: 0, Gini: 0, Top1: 0, Top10: 0 };

  const regimeHint = useMemo(() => {
    const regime = classifyRegime(alpha, lambda, churn);
    return regime.name;
  }, [alpha, lambda, churn]);

  const currentRegime = useMemo(() => {
    return classifyRegime(alpha, lambda, churn);
  }, [alpha, lambda, churn]);

  return (
    <div className="explorer-container">
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35 }}
        className="header"
      >
        <h1 className="title">Shadow Futures Simulator</h1>
        <p className="subtitle">
          A toy model for when work can be identified across different economic regimes. Reinforcement (α) pushes toward lock-in. Effort weight (λ)
          makes transcripts matter. Churn prevents permanent state capture.
        </p>
      </motion.div>

      <div className="main-grid">
        <Card className="controls-card">
          <CardContent className="p-5 space-y-5">
            <div className="controls-header">
              <div className="section-title">Controls</div>
              <Button
                variant="outline"
                size="sm"
                className="gap-2"
                onClick={() => setSeed((s) => s + 1)}
              >
                <RefreshCcw className="h-4 w-4" />
                Reroll
              </Button>
            </div>

            <div className="space-y-2">
              <div className="text-sm font-medium">Presets</div>
              <div className="presets-grid">
                {presets.map((p) => (
                  <Button
                    key={p.key}
                    variant="outline"
                    size="sm"
                    className="justify-start"
                    onClick={() => applyPreset(p)}
                    title={p.desc}
                  >
                    {p.name}
                  </Button>
                ))}
              </div>
              <div className="text-xs text-muted">
                These are narrative labels mapped to (α, λ, churn). They are not historical claims.
              </div>
            </div>

            <LabeledSlider
              label="Reinforcement α"
              value={alpha}
              onChange={setAlpha}
              min={0}
              max={2}
              step={0.05}
              format={(v) => v.toFixed(2)}
            />
            <LabeledSlider
              label="Effort weight λ"
              value={lambda}
              onChange={setLambda}
              min={0}
              max={2}
              step={0.05}
              format={(v) => v.toFixed(2)}
            />
            <LabeledSlider
              label="Churn"
              value={churn}
              onChange={setChurn}
              min={0}
              max={0.05}
              step={0.001}
              format={(v) => v.toFixed(3)}
            />
            <LabeledSlider
              label="Steps T"
              value={T}
              onChange={setT}
              min={100}
              max={1200}
              step={50}
              format={(v) => `${v}`}
            />

            <div className="space-y-2">
              <div className="text-sm font-medium">Tax Policy</div>
              <LabeledSlider
                label="Income Tax Rate"
                value={incomeTaxRate}
                onChange={setIncomeTaxRate}
                min={0}
                max={1.0}
                step={0.01}
                format={(v) => `${(v * 100).toFixed(0)}%`}
              />
              <LabeledSlider
                label="Wealth Tax Rate"
                value={wealthTaxRate}
                onChange={setWealthTaxRate}
                min={0}
                max={1.0}
                step={0.01}
                format={(v) => `${(v * 100).toFixed(0)}%`}
              />
              {totalTaxRevenue !== undefined && totalTaxRevenue > 0 && (
                <div className="text-xs text-muted">
                  Total tax revenue: {totalTaxRevenue.toFixed(2)}
                </div>
              )}
            </div>


            <div className="regime-section">
              <div className="text-sm font-medium">Regime</div>
              <div className="text-sm" style={{ color: currentRegime.color }}>
                {regimeHint}
              </div>
            </div>

            {activePreset && (
              <div className="interpretation-section">
                <div className="text-sm font-medium">Preset interpretation</div>
                <div className="text-sm text-muted leading-relaxed">
                  <span className="font-semibold">{activePreset.name}:</span> {activePreset.interpretation}
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <Card className="charts-card">
          <CardContent className="p-5 space-y-4">
            <div className="charts-header">
              <div>
                <div className="section-title">Signals and concentration over time</div>
                <div className="text-sm text-muted">
                  <strong>MI (Mutual Information)</strong> measures how much information about reward you gain by knowing verified effort. 
                  Near 0 bits means effort tells you nothing about reward; higher values indicate effort is informative.
                </div>
              </div>
              <div className="stats-panel">
                <div className="text-sm text-muted mb-2">Final Metrics</div>
                <div className="text-sm flex items-center gap-2">
                  <span style={{ width: 10, height: 10, backgroundColor: "#06b6d4", borderRadius: 2, flexShrink: 0 }}></span>
                  <span>MI: <span className="font-semibold">{last.MI_bits.toFixed(3)}</span> bits</span>
                </div>
                <div className="text-sm flex items-center gap-2" title="Concentration of cumulative rewards. Compare between scenarios, not to real-world Gini.">
                  <span style={{ width: 10, height: 10, backgroundColor: "#f472b6", borderRadius: 2, flexShrink: 0 }}></span>
                  <span>Conc: <span className="font-semibold">{last.Gini.toFixed(3)}</span>
                    <span className="text-xs text-muted ml-1">
                      ({last.Gini < 0.5 ? "low" : last.Gini < 0.8 ? "mod" : "high"})
                    </span>
                  </span>
                </div>
                <div className="text-sm flex items-center gap-2">
                  <span style={{ width: 10, height: 10, backgroundColor: "#a78bfa", borderRadius: 2, flexShrink: 0 }}></span>
                  <span>Top 1%: <span className="font-semibold">{(last.Top1 * 100).toFixed(1)}%</span></span>
                </div>
                {(incomeTaxRate > 0 || wealthTaxRate > 0) && (
                  <div className="text-xs text-green-400 mt-1 pt-1" style={{ borderTop: "1px solid rgba(255,255,255,0.1)" }}>
                    Tax policy active — metrics reflect taxed allocation
                  </div>
                )}
              </div>
            </div>

            <div className="main-chart" style={{ minHeight: "300px", minWidth: 0 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={series}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis 
                    dataKey="t" 
                    stroke="#888"
                    label={{ value: "Time Step", position: "insideBottom", offset: -5 }}
                  />
                  <YAxis 
                    stroke="#888"
                    label={{ value: "Value", angle: -90, position: "insideLeft" }}
                  />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #333' }}
                    labelStyle={{ color: '#fff' }}
                  />
                  <Line type="monotone" dataKey="MI_bits" stroke="#06b6d4" dot={false} name="MI (bits)" />
                  <Line type="monotone" dataKey="Gini" stroke="#f472b6" dot={false} name="Concentration" />
                  <Line type="monotone" dataKey="Top1" stroke="#a78bfa" dot={false} name="Top 1%" />
                </LineChart>
              </ResponsiveContainer>
            </div>
            
            {/* Chart legend */}
            <div style={{ display: "flex", justifyContent: "center", flexWrap: "wrap", gap: "16px", fontSize: "12px", marginTop: "8px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                <div style={{ width: "20px", height: "3px", backgroundColor: "#06b6d4" }}></div>
                <span style={{ color: "#9ca3af" }}>MI (effort informativeness)</span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                <div style={{ width: "20px", height: "3px", backgroundColor: "#f472b6" }}></div>
                <span style={{ color: "#9ca3af" }}>Concentration{(incomeTaxRate > 0 || wealthTaxRate > 0) ? " (with tax policy)" : ""}</span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                <div style={{ width: "20px", height: "3px", backgroundColor: "#a78bfa" }}></div>
                <span style={{ color: "#9ca3af" }}>Top 1% share</span>
              </div>
            </div>

            <div className="space-y-4 mt-6">
              <div>
                <div className="section-title">
                  Tax Revenue Analysis
                  {isPending && <span className="text-xs text-muted ml-2">(Updating...)</span>}
                </div>
                <div className="text-sm text-muted leading-relaxed mb-4">
                  <strong>Solid lines:</strong> Tax revenue at each rate. <strong>Dashed lines:</strong> How concentrated 
                  rewards become (lower = more equal allocation). Vertical lines mark your current settings.
                </div>
              </div>
              <div style={{ height: "300px", width: "100%", minHeight: "300px", minWidth: 0 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis 
                      dataKey="rate" 
                      name="Tax Rate (%)" 
                      stroke="#888"
                      label={{ value: "Tax Rate (%)", position: "insideBottom", offset: -5 }}
                    />
                    <YAxis 
                      yAxisId="left"
                      stroke="#888"
                      label={{ value: "Revenue", angle: -90, position: "insideLeft" }}
                    />
                    <YAxis 
                      yAxisId="right" 
                      orientation="right"
                      stroke="#888"
                      domain={[0, 1]}
                      label={{ value: "Concentration", angle: 90, position: "insideRight" }}
                    />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #333' }}
                      labelStyle={{ color: '#fff' }}
                    />
                    <Line 
                      yAxisId="left"
                      type="monotone" 
                      dataKey="revenue" 
                      data={incomeTaxCurve} 
                      stroke="#06b6d4" 
                      name="Income Tax Revenue"
                      dot={false}
                    />
                    <Line 
                      yAxisId="left"
                      type="monotone" 
                      dataKey="revenue" 
                      data={wealthTaxCurve} 
                      stroke="#f472b6" 
                      name="Wealth Tax Revenue"
                      dot={false}
                    />
                    <Line 
                      yAxisId="right"
                      type="monotone" 
                      dataKey="postTaxGini" 
                      data={incomeTaxCurve} 
                      stroke="#06b6d4" 
                      strokeDasharray="5 5"
                      name="Concentration (Income Tax)"
                      dot={false}
                    />
                    <Line 
                      yAxisId="right"
                      type="monotone" 
                      dataKey="postTaxGini" 
                      data={wealthTaxCurve} 
                      stroke="#f472b6" 
                      strokeDasharray="5 5"
                      name="Concentration (Wealth Tax)"
                      dot={false}
                    />
                    {incomeTaxRate > 0 && (
                      <ReferenceLine 
                        x={incomeTaxRate * 100} 
                        stroke="#06b6d4" 
                        strokeWidth={2}
                        label={{ value: `Income: ${(incomeTaxRate * 100).toFixed(0)}%`, position: "top", fill: "#06b6d4", fontSize: 11 }}
                      />
                    )}
                    {wealthTaxRate > 0 && (
                      <ReferenceLine 
                        x={wealthTaxRate * 100} 
                        stroke="#f472b6" 
                        strokeWidth={2}
                        label={{ value: `Wealth: ${(wealthTaxRate * 100).toFixed(0)}%`, position: "top", fill: "#f472b6", fontSize: 11 }}
                      />
                    )}
                  </LineChart>
                </ResponsiveContainer>
              </div>
              {/* Optimal rates summary */}
              <div style={{ 
                display: "grid", 
                gridTemplateColumns: "1fr 1fr", 
                gap: "12px", 
                marginTop: "12px",
                padding: "12px",
                backgroundColor: "rgba(0,0,0,0.2)",
                borderRadius: "8px"
              }}>
                <div style={{ padding: "10px", backgroundColor: "rgba(6, 182, 212, 0.1)", borderRadius: "6px", borderLeft: "3px solid #06b6d4" }}>
                  <div style={{ fontSize: "12px", color: "#06b6d4", fontWeight: 600, marginBottom: "6px" }}>Income Tax (cyan)</div>
                  <div style={{ fontSize: "12px", color: "#e5e7eb", marginBottom: "4px" }}>
                    Peak revenue: <strong>{optimalIncomeTax?.peakRevenueRate.toFixed(0)}%</strong>
                  </div>
                  <div style={{ fontSize: "12px", color: "#e5e7eb", marginBottom: "4px" }}>
                    Min concentration: <strong>{optimalIncomeTax?.minGiniRate.toFixed(0)}%</strong> → {optimalIncomeTax?.minGini.toFixed(2)}
                  </div>
                  <div style={{ fontSize: "11px", color: "#9ca3af" }}>
                    (baseline: {optimalIncomeTax?.baseGini.toFixed(2)})
                  </div>
                </div>
                <div style={{ padding: "10px", backgroundColor: "rgba(244, 114, 182, 0.1)", borderRadius: "6px", borderLeft: "3px solid #f472b6" }}>
                  <div style={{ fontSize: "12px", color: "#f472b6", fontWeight: 600, marginBottom: "6px" }}>Wealth Tax (pink)</div>
                  <div style={{ fontSize: "12px", color: "#e5e7eb", marginBottom: "4px" }}>
                    Peak revenue: <strong>{optimalWealthTax?.peakRevenueRate.toFixed(0)}%</strong>
                  </div>
                  <div style={{ fontSize: "12px", color: "#e5e7eb", marginBottom: "4px" }}>
                    Min concentration: <strong>{optimalWealthTax?.minGiniRate.toFixed(0)}%</strong> → {optimalWealthTax?.minGini.toFixed(2)}
                  </div>
                  <div style={{ fontSize: "11px", color: "#9ca3af" }}>
                    (baseline: {optimalWealthTax?.baseGini.toFixed(2)})
                  </div>
                </div>
              </div>
              <div className="text-xs text-muted mt-2">
                Lower dashed line = fairer allocation. In high-α regimes, wealth taxes often reduce concentration more effectively.
              </div>
            </div>

            <div className="sub-charts-grid mt-6">
              <div className="space-y-2">
                <div className="text-sm font-medium">Top reward shares</div>
                <div className="sub-chart" style={{ minHeight: "200px", minWidth: 0 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={bars}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                      <XAxis 
                        dataKey="rank" 
                        stroke="#888"
                        label={{ value: "Rank", position: "insideBottom", offset: -5 }}
                      />
                      <YAxis 
                        stroke="#888"
                        label={{ value: "Share", angle: -90, position: "insideLeft" }}
                      />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #333' }}
                        labelStyle={{ color: '#fff' }}
                      />
                      <Bar dataKey="share" fill="#06b6d4" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <div className="space-y-2">
                <div className="text-sm font-medium">Effort bins vs chance of any reward</div>
                <div className="sub-chart" style={{ minHeight: "200px", minWidth: 0 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={effortCurve}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                      <XAxis 
                        dataKey="bin" 
                        stroke="#888"
                        label={{ value: "Effort Bin", position: "insideBottom", offset: -5 }}
                      />
                      <YAxis 
                        stroke="#888"
                        label={{ value: "Probability", angle: -90, position: "insideLeft" }}
                      />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #333' }}
                        labelStyle={{ color: '#fff' }}
                      />
                      <Line type="monotone" dataKey="pRewarded" stroke="#22c55e" dot={false} name="P(Rewarded)" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card className="info-card">
          <CardContent className="p-6 space-y-6">
            <div>
              <div className="section-title" style={{ fontSize: "18px", marginBottom: "8px" }}>
                Phase Transition Map
                {isPending && <span className="text-xs text-muted ml-2">(Updating...)</span>}
              </div>
              <p style={{ fontSize: "14px", color: "#b0b0b0", lineHeight: 1.6 }}>
                Each cell shows the system behavior at that (α, λ) combination. 
                The <span style={{ 
                  border: "2px solid white",
                  padding: "2px 8px", 
                  borderRadius: "4px",
                  fontWeight: 600,
                  color: "#e5e7eb"
                }}>highlighted cell</span> is your current position. Hover any cell for details.
              </p>
            </div>
            
            {/* Custom heatmap-style phase diagram */}
            <div style={{ width: "100%", maxWidth: "520px", margin: "0 auto" }}>
              {/* Main grid with axes */}
              <div style={{ display: "flex", alignItems: "stretch" }}>
                {/* Y-axis label (rotated) */}
                <div style={{ 
                  writingMode: "vertical-rl", 
                  transform: "rotate(180deg)", 
                  fontSize: "13px", 
                  color: "#d1d5db",
                  fontWeight: 500,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  paddingRight: "8px"
                }}>
                  λ (Effort Weight) →
                </div>
                
                {/* Y-axis ticks */}
                <div style={{ 
                  display: "flex", 
                  flexDirection: "column", 
                  justifyContent: "space-between", 
                  fontSize: "12px", 
                  color: "#9ca3af",
                  paddingRight: "8px",
                  height: "320px",
                  textAlign: "right",
                  minWidth: "32px"
                }}>
                  <span>2.0</span>
                  <span>1.5</span>
                  <span>1.0</span>
                  <span>0.5</span>
                  <span>0.0</span>
                </div>
                
                {/* The actual grid */}
                <div style={{ position: "relative", flex: 1, height: "320px" }}>
                  <div 
                    style={{ 
                      display: "grid",
                      gridTemplateColumns: "repeat(10, 1fr)",
                      gridTemplateRows: "repeat(10, 1fr)",
                      width: "100%",
                      height: "100%",
                      gap: "2px",
                      backgroundColor: "rgba(0,0,0,0.4)",
                      borderRadius: "4px",
                      padding: "2px"
                    }}
                  >
                    {(() => {
                      // Calculate which single cell should be highlighted (grid is 10x10, range 0-2)
                      const alphaIdx = Math.min(9, Math.max(0, Math.floor(alpha / 0.2)));
                      const lambdaIdx = Math.min(9, Math.max(0, Math.floor(lambda / 0.2)));
                      // Grid is generated top-to-bottom (high λ first), so row = 9 - lambdaIdx
                      const currentCellIdx = (9 - lambdaIdx) * 10 + alphaIdx;
                      
                      return phaseTransitionData.map((point, idx) => {
                      const isCurrentCell = idx === currentCellIdx;
                      
                      return (
                        <div
                          key={idx}
                          className={`phase-cell ${isCurrentCell ? "phase-cell-current" : ""}`}
                          style={{ backgroundColor: point.color }}
                        >
                          {isCurrentCell && (
                            <span style={{ 
                              fontSize: "14px", 
                              fontWeight: "bold",
                              color: "#fff",
                              textShadow: "0 0 4px rgba(0,0,0,0.8)"
                            }}>●</span>
                          )}
                          <div className="phase-tooltip">
                            <div className="phase-tooltip-regime" style={{ color: point.color }}>
                              {point.regime}
                            </div>
                            <div className="phase-tooltip-params">
                              α = {point.alpha.toFixed(1)}, λ = {point.lambda.toFixed(1)}
                            </div>
                            <div className="phase-tooltip-metrics">
                              MI: {point.mi.toFixed(3)} bits · Conc: {point.gini.toFixed(2)}
                            </div>
                          </div>
                        </div>
                      );
                    });
                    })()}
                  </div>
                </div>
              </div>
              
              {/* X-axis ticks */}
              <div style={{ 
                display: "flex", 
                justifyContent: "space-between", 
                fontSize: "12px", 
                color: "#9ca3af",
                paddingTop: "8px",
                marginLeft: "58px"
              }}>
                <span>0.0</span>
                <span>0.5</span>
                <span>1.0</span>
                <span>1.5</span>
                <span>2.0</span>
              </div>
              
              {/* X-axis label */}
              <div style={{ textAlign: "center", fontSize: "13px", color: "#d1d5db", fontWeight: 500, marginTop: "6px", marginLeft: "58px" }}>
                α (Reinforcement) →
              </div>
            </div>
            
            {/* Legend - horizontal color key */}
            <div style={{ 
              display: "flex", 
              flexWrap: "wrap", 
              justifyContent: "center", 
              gap: "16px", 
              padding: "16px",
              backgroundColor: "rgba(0, 0, 0, 0.2)",
              borderRadius: "8px"
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <div style={{ width: "20px", height: "20px", borderRadius: "4px", backgroundColor: "#3b82f6" }}></div>
                <span style={{ color: "#e5e7eb", fontSize: "13px" }}>Ordered</span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <div style={{ width: "20px", height: "20px", borderRadius: "4px", backgroundColor: "#10b981" }}></div>
                <span style={{ color: "#e5e7eb", fontSize: "13px" }}>Periodic</span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <div style={{ width: "20px", height: "20px", borderRadius: "4px", backgroundColor: "#f59e0b" }}></div>
                <span style={{ color: "#e5e7eb", fontSize: "13px" }}>Edge of Chaos</span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <div style={{ width: "20px", height: "20px", borderRadius: "4px", backgroundColor: "#ef4444" }}></div>
                <span style={{ color: "#e5e7eb", fontSize: "13px" }}>Chaotic</span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <div style={{ width: "20px", height: "20px", borderRadius: "4px", backgroundColor: "#8b5cf6" }}></div>
                <span style={{ color: "#e5e7eb", fontSize: "13px" }}>Transitional</span>
              </div>
            </div>

            {/* What the regimes mean */}
            <div style={{ 
              backgroundColor: "rgba(55, 65, 81, 0.4)", 
              borderRadius: "8px", 
              padding: "20px",
              border: "1px solid rgba(255,255,255,0.1)"
            }}>
              <h4 style={{ color: "#f3f4f6", fontSize: "15px", fontWeight: 600, marginBottom: "16px" }}>
                What Each Regime Means
              </h4>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
                <div style={{ padding: "12px", backgroundColor: "rgba(59, 130, 246, 0.15)", borderRadius: "6px", borderLeft: "3px solid #3b82f6" }}>
                  <div style={{ color: "#60a5fa", fontWeight: 600, fontSize: "14px", marginBottom: "4px" }}>Ordered</div>
                  <div style={{ color: "#d1d5db", fontSize: "13px", lineHeight: 1.5 }}>Effort directly leads to reward. Predictable, meritocratic outcomes.</div>
                </div>
                <div style={{ padding: "12px", backgroundColor: "rgba(16, 185, 129, 0.15)", borderRadius: "6px", borderLeft: "3px solid #10b981" }}>
                  <div style={{ color: "#34d399", fontWeight: 600, fontSize: "14px", marginBottom: "4px" }}>Periodic</div>
                  <div style={{ color: "#d1d5db", fontSize: "13px", lineHeight: 1.5 }}>Effort matters, but cycles and patterns emerge over time.</div>
                </div>
                <div style={{ padding: "12px", backgroundColor: "rgba(245, 158, 11, 0.15)", borderRadius: "6px", borderLeft: "3px solid #f59e0b" }}>
                  <div style={{ color: "#fbbf24", fontWeight: 600, fontSize: "14px", marginBottom: "4px" }}>Edge of Chaos</div>
                  <div style={{ color: "#d1d5db", fontSize: "13px", lineHeight: 1.5 }}>Balanced tension between order and chaos. Maximum adaptability.</div>
                </div>
                <div style={{ padding: "12px", backgroundColor: "rgba(239, 68, 68, 0.15)", borderRadius: "6px", borderLeft: "3px solid #ef4444" }}>
                  <div style={{ color: "#f87171", fontWeight: 600, fontSize: "14px", marginBottom: "4px" }}>Chaotic</div>
                  <div style={{ color: "#d1d5db", fontSize: "13px", lineHeight: 1.5 }}>Winner-take-all lock-in. Effort becomes uninformative.</div>
                </div>
              </div>
              <div style={{ marginTop: "12px", padding: "12px", backgroundColor: "rgba(139, 92, 246, 0.15)", borderRadius: "6px", borderLeft: "3px solid #8b5cf6" }}>
                <div style={{ color: "#a78bfa", fontWeight: 600, fontSize: "14px", marginBottom: "4px" }}>Transitional</div>
                <div style={{ color: "#d1d5db", fontSize: "13px", lineHeight: 1.5 }}>Boundary zones between regimes. Mixed dynamics with features of adjacent states.</div>
              </div>
            </div>
            
            {/* Navigation guide - 4 corners */}
            <div style={{ 
              backgroundColor: "rgba(0, 0, 0, 0.25)", 
              borderRadius: "8px", 
              padding: "20px",
              border: "1px solid rgba(255,255,255,0.08)"
            }}>
              <h4 style={{ color: "#f3f4f6", fontSize: "15px", fontWeight: 600, marginBottom: "8px" }}>
                What the α-λ Corners Mean
              </h4>
              <p style={{ color: "#9ca3af", fontSize: "13px", marginBottom: "16px", lineHeight: 1.5 }}>
                These are fixed interpretations of parameter space regions — they don't change between simulations.
              </p>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
                <div style={{ padding: "12px", backgroundColor: "rgba(96, 165, 250, 0.1)", borderRadius: "6px" }}>
                  <div style={{ color: "#60a5fa", fontWeight: 600, fontSize: "14px", marginBottom: "4px" }}>↖ Low α, High λ</div>
                  <div style={{ color: "#e5e7eb", fontSize: "13px" }}><strong>Meritocracy</strong> — Hard work reliably pays off</div>
                </div>
                <div style={{ padding: "12px", backgroundColor: "rgba(251, 191, 36, 0.1)", borderRadius: "6px" }}>
                  <div style={{ color: "#fbbf24", fontWeight: 600, fontSize: "14px", marginBottom: "4px" }}>↗ High α, High λ</div>
                  <div style={{ color: "#e5e7eb", fontSize: "13px" }}><strong>Tension Zone</strong> — Effort and advantage compete</div>
                </div>
                <div style={{ padding: "12px", backgroundColor: "rgba(167, 139, 250, 0.1)", borderRadius: "6px" }}>
                  <div style={{ color: "#a78bfa", fontWeight: 600, fontSize: "14px", marginBottom: "4px" }}>↙ Low α, Low λ</div>
                  <div style={{ color: "#e5e7eb", fontSize: "13px" }}><strong>Random</strong> — Neither effort nor history matters</div>
                </div>
                <div style={{ padding: "12px", backgroundColor: "rgba(248, 113, 113, 0.1)", borderRadius: "6px" }}>
                  <div style={{ color: "#f87171", fontWeight: 600, fontSize: "14px", marginBottom: "4px" }}>↘ High α, Low λ</div>
                  <div style={{ color: "#e5e7eb", fontSize: "13px" }}><strong>Lock-in</strong> — Early winners dominate forever</div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

      <Card className="info-card">
          <CardContent className="p-5 space-y-4">
            <div className="section-title">Glossary</div>
            <div className="space-y-4 text-sm">
              <div>
                <div className="font-semibold text-base mb-1">Reinforcement (α)</div>
                <div className="text-muted leading-relaxed">
                  Controls how strongly past rewards amplify future reward probability. When α &gt; 1, the system exhibits 
                  increasing returns: early winners gain disproportionate advantage. When α &lt; 1, advantages decay over time.
                  This creates path dependence where historical accidents compound.
                </div>
              </div>
              <div>
                <div className="font-semibold text-base mb-1">Effort Weight (λ)</div>
                <div className="text-muted leading-relaxed">
                  Determines how much verifiable effort (work transcripts) influences reward allocation. High λ means effort 
                  is highly informative about outcomes. Low λ means effort provides little signal, even when perfectly verifiable.
                  In high α regimes, even high λ can fail to create informative signals due to lock-in.
                </div>
              </div>
              <div>
                <div className="font-semibold text-base mb-1">Churn</div>
                <div className="text-muted leading-relaxed">
                  The rate at which accumulated advantage (attachment A) decays over time. High churn prevents permanent 
                  lock-in by continuously mixing the system state. Low churn allows early advantages to persist indefinitely.
                  Churn represents turnover, competition, or institutional mechanisms that reset advantages.
                </div>
              </div>
              <div>
                <div className="font-semibold text-base mb-1">Work Transcripts</div>
                <div className="text-muted leading-relaxed">
                  Verifiable records of effort (V). In the model, each agent has a transcript value v ∈ [0,1] representing 
                  their verified work. Despite perfect verification, transcripts may fail to signal value creation when 
                  reinforcement dominates allocation.
                </div>
              </div>
              <div>
                <div className="font-semibold text-base mb-1">Lock-in</div>
                <div className="text-muted leading-relaxed">
                  A state where early advantages become permanent due to high reinforcement (α) and low churn. Once lock-in 
                  occurs, effort becomes uninformative even though it remains verifiable. The system converges to a state 
                  where outcomes are determined by historical accidents rather than current effort.
                </div>
              </div>
              <div>
                <div className="font-semibold text-base mb-1">Mutual Information (MI)</div>
                <div className="text-muted leading-relaxed">
                  Measures how much information about reward outcomes you gain by knowing effort transcripts. Measured in bits.
                  MI = 0 means effort tells you nothing about rewards. Higher MI means effort is informative. In lock-in regimes,
                  MI collapses toward zero even with perfect verification.
                </div>
              </div>
              <div>
                <div className="font-semibold text-base mb-1">Concentration (Gini-based)</div>
                <div className="text-muted leading-relaxed">
                  Measures how concentrated cumulative rewards are. 0 = perfectly equal, 1 = one agent has everything.
                  <strong className="text-yellow-500"> Important:</strong> These values are for comparing scenarios within this model, 
                  not for comparison to real-world Gini coefficients. The model's cumulative structure and growing population 
                  produce higher absolute values than real economies. Focus on <em>relative</em> differences: 
                  lower concentration in Ordered vs. Chaotic regimes, and how taxes reduce concentration.
                </div>
              </div>
              <div>
                <div className="font-semibold text-base mb-1">Attachment (A)</div>
                <div className="text-muted leading-relaxed">
                  Accumulated advantage that determines future reward probability. Each reward increases A, creating reinforcement.
                  A represents network position, capital, reputation, or any advantage that compounds. Wealth taxes target A directly.
                </div>
              </div>
              <div>
                <div className="font-semibold text-base mb-1">Regimes</div>
                <div className="text-muted leading-relaxed space-y-1">
                  <div><strong>Ordered:</strong> Low α, high λ, moderate churn. Effort is highly informative, outcomes are predictable.</div>
                  <div><strong>Periodic:</strong> Moderate α, high λ, low churn. Patterns emerge but remain effort-dependent.</div>
                  <div><strong>Complex (Edge of Chaos):</strong> Balanced parameters. Maximum adaptability, effort remains informative.</div>
                  <div><strong>Chaotic:</strong> High α, low λ, low churn. Lock-in dominates, effort becomes uninformative.</div>
                </div>
              </div>
              <div>
                <div className="font-semibold text-base mb-1">Wealth Tax vs Income Tax</div>
                <div className="text-muted leading-relaxed">
                  <strong>Income Tax:</strong> Applied to reward flows. Captures new rewards but doesn't address accumulated advantage.
                  In high α regimes, income taxes may reduce revenue as rates increase (Laffer curve effect).
                  <br />
                  <strong>Wealth Tax:</strong> Applied to accumulated attachment (A). Directly targets the source of increasing returns.
                  More effective in high α regimes because it reduces the compounding advantage that drives inequality.
                  Can generate more revenue at lower rates while reducing inequality more effectively.
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

      <Card className="info-card">
        <CardContent className="p-5 space-y-3">
          <div className="section-title">How to read this</div>
          <div className="text-sm text-muted leading-relaxed">
            If α is high and churn is low, early wins compound and the system locks in. In that regime, the effort curve can flatten
            and MI can drift toward zero, even though all effort is perfectly verifiable. Increasing λ makes effort matter more in the
            allocation rule. Increasing churn prevents permanent advantage by continually mixing the state. Wealth taxes are particularly
            effective in high α regimes because they target accumulated advantage directly, unlike income taxes which only capture flows.
          </div>
        </CardContent>
      </Card>

      <div className="text-xs text-muted">
        Notes: MI is estimated by binning effort and treating reward as the event of receiving any reward. 
        Concentration values are Gini-based but should be compared <em>between scenarios</em>, not to real-world data—the model's 
        cumulative structure produces higher absolute values. Tax simulations use simplified models and don't account for 
        behavioral responses or general equilibrium effects. This is a visualization aid, not an econometric estimator.
      </div>
    </div>
  );
}
