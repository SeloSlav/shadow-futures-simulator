# Shadow Futures Simulator

An interactive visualization demonstrating why **verifiable work cannot signal value** in path-dependent economies.

**[Live Demo](https://seloslav.github.io/shadow-futures-simulator)**

## Overview

This simulator illustrates the core concepts from the paper *"Shadow Futures: Why Verifiable Work Cannot Signal Value in Path-Dependent Economies"* by Martin Erlic.

In economies governed by **path dependence** and **increasing returns to scale**, even perfectly verified effort provides no recoverable information about realized outcomes. Early stochastic advantages compound over time, causing identical work histories to produce radically different results.

## Key Concepts

### Shadow Futures
Unrealized but observationally indistinguishable effort trajectories that fail due solely to unfavorable timing or network position. For every success, there exists a large set of equivalent failures—identical in effort, different only in luck.

### Mutual Information Collapse
As reinforcement compounds, the mutual information between verified work and realized reward converges to zero. Verification confirms that work *occurred*, but cannot establish that work *caused* the outcome.

### Preferential Attachment
Rewards follow a "rich get richer" dynamic where the probability of receiving future rewards is proportional to accumulated past rewards:

```
P(reward) = Aᵅ / Σ Aᵅ
```

Where `α` controls the strength of path dependence:
- `α = 0`: Uniform random allocation
- `α = 1`: Linear preferential attachment  
- `α > 1`: Winner-take-all dynamics

## The Simulation

Watch agents with **identical effort** enter a system over time. Despite performing the same verified work, their outcomes diverge dramatically based on:

- **Entry timing** — Early entrants accumulate advantages
- **Stochastic realization** — Random early rewards compound
- **Network position** — Attachment begets attachment

The simulation reveals that observed success cannot be causally attributed to work, even ex post. It is a historical artifact of path dependence rather than a signal of value creation.

## Implications

This framework applies to:
- **Labor markets** — Career trajectories locked in by early opportunities
- **Scientific publishing** — Citation networks and the Matthew Effect
- **Digital platforms** — Viral content and algorithmic amplification
- **Proof-of-work systems** — Mining pools and hash rate concentration

In all cases, verification mechanisms persist despite the loss of causal and informational content. Proof of work converges to proof of position.

## Running Locally

```bash
npm install
npm run dev
```

## Built With

- React + TypeScript
- Vite
- Recharts
- Framer Motion
- Tailwind CSS

## License

MIT License — see [LICENSE](LICENSE) for details.

## Citation

If referencing the underlying theory:

> Erlic, M. (2025). Shadow Futures: Why Verifiable Work Cannot Signal Value in Path-Dependent Economies.
