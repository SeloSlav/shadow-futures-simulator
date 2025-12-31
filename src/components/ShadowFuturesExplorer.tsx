import { useMemo, useState } from "react";
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
} from "recharts";

// Types
interface SeriesDataPoint {
  t: number;
  MI_bits: number;
  Gini: number;
  Top1: number;
  Top10: number;
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
}

interface SimulationParams {
  T?: number;
  alpha?: number;
  lambda?: number;
  churn?: number;
  A0?: number;
  bins?: number;
  seed?: number;
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
}: SimulationParams): SimulationResult {
  const rng = mulberry32(seed);

  // State arrays
  const effort: number[] = [];
  const A: number[] = [];
  const gotAnyReward: number[] = [];
  const rewardCount: number[] = [];

  // Track time series for plotting
  const series: SeriesDataPoint[] = [];

  for (let t = 0; t < T; t++) {
    // Entry
    const v = rng(); // verifiable effort transcript in [0,1]
    effort.push(v);
    A.push(A0);
    gotAnyReward.push(0);
    rewardCount.push(0);

    // Churn: decay attachments slightly each step (prevents full lock-in)
    if (churn > 0) {
      for (let i = 0; i < A.length; i++) A[i] *= 1 - churn;
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
    A[winner] += 1;
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

    series.push({
      t: t + 1,
      MI_bits: mi,
      Gini: g,
      Top1: top1,
      Top10: top10share,
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

  return { series, bars, effortCurve };
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
  const [T, setT] = useState(400);
  const [seed, setSeed] = useState(7);
  const [activePreset, setActivePreset] = useState<Preset | null>(null);

  const bins = 10;

  const presets: Preset[] = useMemo(
    () => [
      {
        key: "agrarian",
        name: "18th c. agrarian",
        desc: "Weak reinforcement, effort visible, moderate mixing",
        interpretation:
          "Production is local and scale is limited. Output mostly tracks labor input, and advantages decay over time (moderate churn). Reinforcement is weak (low α), so early luck does not permanently dominate. Because λ is high, effort transcripts remain informative: working harder reliably increases the chance of reward.",
        params: { alpha: 0.35, lambda: 1.2, churn: 0.012, T: 350 },
      },
      {
        key: "industrial",
        name: "Industrial",
        desc: "Some scale effects, effort matters, low mixing",
        interpretation:
          "Mechanization and capital introduce scale effects. Early success helps, but effort and skill still matter. Reinforcement is present but not overwhelming (α < 1). Churn is low, so firms and workers can accumulate advantage, yet effort remains partially identifiable.",
        params: { alpha: 0.8, lambda: 0.8, churn: 0.004, T: 500 },
      },
      {
        key: "platform",
        name: "High-tech platform",
        desc: "Preferential attachment, weak mixing, transcripts fade",
        interpretation:
          "Digital platforms amplify visibility and network position. Preferential attachment dominates (α > 1), so early winners gain disproportionate reach. Individual effort matters locally, but its signal decays quickly. Many agents exert similar effort yet receive radically different outcomes.",
        params: { alpha: 1.25, lambda: 0.25, churn: 0.002, T: 700 },
      },
      {
        key: "winner",
        name: "Winner-take-most",
        desc: "Strong increasing returns, near lock-in",
        interpretation:
          "Markets with extreme scale economies and low turnover. Once dominance emerges, it is rarely overturned. Reinforcement overwhelms effort, and churn is nearly absent. Work verification mainly certifies participation rather than value creation.",
        params: { alpha: 1.6, lambda: 0.15, churn: 0.0, T: 900 },
      },
      {
        key: "agentic",
        name: "Far-future agentic",
        desc: "Extreme reinforcement, pooling, low human identifiability",
        interpretation:
          "Highly automated systems where agents, models, or capital pools act at massive scale. Reinforcement is extreme and mixing negligible. Individual human effort becomes statistically irrelevant to outcomes. Attribution collapses almost entirely.",
        params: { alpha: 1.85, lambda: 0.05, churn: 0.0, T: 1100 },
      },
      {
        key: "edge",
        name: "Edge regime",
        desc: "Mixed: reinforcement + mixing keeps signals alive",
        interpretation:
          "A transitional regime near the edge of chaos. Reinforcement exists but is counterbalanced by churn and meaningful effort channels. Outcomes are neither fully random nor fully locked in. This is where effort is most informative and adaptation remains possible.",
        params: { alpha: 1.05, lambda: 0.6, churn: 0.01, T: 600 },
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

  const { series, bars, effortCurve } = useMemo(() => {
    return simulate({ T, alpha, lambda, churn, bins, seed });
  }, [T, alpha, lambda, churn, seed]);

  const last = series[series.length - 1] || { MI_bits: 0, Gini: 0, Top1: 0, Top10: 0 };

  const regimeHint = useMemo(() => {
    if (churn >= 0.02) return "High mixing";
    if (alpha < 0.7) return "Weak reinforcement";
    if (alpha >= 1.2 && churn <= 0.005 && lambda < 0.5) return "Lock-in";
    if (alpha >= 1.0 && lambda >= 0.8 && churn <= 0.01) return "Effort-visible";
    return "Intermediate";
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
          Toy model for when work can be identified across regimes. Reinforcement (α) pushes toward lock-in. Effort weight (λ)
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

            <div className="regime-section">
              <div className="text-sm font-medium">Regime</div>
              <div className="text-sm text-muted">{regimeHint}</div>
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
                  Mutual information estimates how much reward outcomes tell you about verified effort.
                </div>
              </div>
              <div className="stats-panel">
                <div className="text-sm text-muted">Final</div>
                <div className="text-sm">
                  MI: <span className="font-semibold">{last.MI_bits.toFixed(3)}</span> bits
                </div>
                <div className="text-sm">
                  Gini: <span className="font-semibold">{last.Gini.toFixed(3)}</span>
                </div>
                <div className="text-sm">
                  Top1: <span className="font-semibold">{(last.Top1 * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>

            <div className="main-chart">
              <ResponsiveContainer>
                <LineChart data={series}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis dataKey="t" stroke="#888" />
                  <YAxis stroke="#888" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #333' }}
                    labelStyle={{ color: '#fff' }}
                  />
                  <Line type="monotone" dataKey="MI_bits" stroke="#06b6d4" dot={false} name="MI (bits)" />
                  <Line type="monotone" dataKey="Gini" stroke="#f472b6" dot={false} name="Gini" />
                  <Line type="monotone" dataKey="Top1" stroke="#a78bfa" dot={false} name="Top 1%" />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="sub-charts-grid">
              <div className="space-y-2">
                <div className="text-sm font-medium">Top reward shares</div>
                <div className="sub-chart">
                  <ResponsiveContainer>
                    <BarChart data={bars}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                      <XAxis dataKey="rank" stroke="#888" />
                      <YAxis stroke="#888" />
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
                <div className="sub-chart">
                  <ResponsiveContainer>
                    <LineChart data={effortCurve}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                      <XAxis dataKey="bin" stroke="#888" />
                      <YAxis stroke="#888" />
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
        <CardContent className="p-5 space-y-3">
          <div className="section-title">How to read this</div>
          <div className="text-sm text-muted leading-relaxed">
            If α is high and churn is low, early wins compound and the system locks in. In that regime, the effort curve can flatten
            and MI can drift toward zero, even though all effort is perfectly verifiable. Increasing λ makes effort matter more in the
            allocation rule. Increasing churn prevents permanent advantage by continually mixing the state.
          </div>
        </CardContent>
      </Card>

      <div className="text-xs text-muted">
        Notes: MI is estimated by binning effort and treating reward as the event of receiving any reward. This is a visualization aid,
        not an econometric estimator.
      </div>
    </div>
  );
}
