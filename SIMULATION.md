# Simulation Rules and Concepts

## Core Concepts
Based on Particle Life: Particles interact via type-based rules, creating emergent structures like clusters or chasers. Asymmetric forces add energy, friction damps it.

## Parameters
- Particle Count: 5000 (default).
- Types: 4 (default, up to 16).
- Interaction Radius: 50.
- Collision Radius: 5.
- Collision Strength: 2.0.
- Friction: 0.1.
- Force Scale: 1.0.
- DT: 0.1.
- World Size: 800x600 (toroidal).

## Force Matrix
NxN float64 array, randomized [-1,1]. Symmetric option available but default asymmetric.

## Extensions
- Evolution: Rules adapt, simulating natural selection.
- Lifecycle: Adds dynamism, preventing stagnation.

This respects the original while extending for more complexity.