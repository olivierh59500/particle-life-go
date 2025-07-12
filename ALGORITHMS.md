# Algorithms in Particle Life Simulation

## Spatial Partitioning (Grid Binning)
To avoid O(N²) force computations:
- Divide world into grid cells (size = max interaction radius).
- Assign particles to bins via hash (x,y -> bin index).
- For each particle, compute forces only from particles in its bin and 8 neighbors.
- Parallelized: Use goroutines to process bins concurrently, with sync.WaitGroup.

## Force Calculation
For each pair (p1 of type i, p2 of type j):
- Distance r = sqrt(dx² + dy²), handling toroidal wrap.
- If r < collision_radius: Repulsive force = collision_strength * (collision_radius - r) / collision_radius * direction away.
- If r < interaction_radius: Interaction force = matrix[i][j] * (interaction_radius - r) / interaction_radius * direction (towards if >0, away if <0).
- Total force summed, velocity += force * dt (unit mass).

## Integration
Semi-implicit Euler:
1. Compute all forces -> update velocities.
2. Apply friction: v *= (1 - friction * dt).
3. Update positions: pos += v * dt.
4. Wrap boundaries.

## Evolution Mode
Every 1000 ticks: Mutate matrix entries by ±0.1 (Gaussian noise), clamped [-1,1].

## Birth/Death
- Death: If <3 neighbors within interaction_radius, remove particle.
- Birth: In dense bins (>10 particles), add new particle nearby with random type.

## Visualization
- Particles: Draw circles colored by type.
- Heatmap: Sample force magnitude at grid points, color by intensity.
- Trails: Keep history of last 10 positions, draw lines.