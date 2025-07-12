# Algorithms in Particle Life Simulation

## Spatial Partitioning (Grid Binning)
To avoid O(N²) force computations:
- Divide world into grid cells (size = max interaction radius).
- Assign particles to bins via hash (x,y -> bin index).
- For each particle, compute forces only from particles in its bin and 8 neighbors.
- Parallelized: Use goroutines to process bins concurrently, with sync.WaitGroup.
- Used for force calculations and particle connections.

## Force Calculation
For each pair (p1 of type i, p2 of type j):
- Distance r = sqrt(dx² + dy²), handling toroidal wrap.
- If r < collision_radius: Repulsive force = collision_strength * (collision_radius - r) / collision_radius * direction away.
- If r < interaction_radius: Interaction force = matrix[i][j] * (interaction_radius - r) / interaction_radius * direction (towards if >0, away if <0).
- Apply environmental force scale from EnvGrid.
- Total force summed, velocity += force * dt (unit mass).

## Integration
Semi-implicit Euler:
1. Compute all forces -> update velocities.
2. Apply friction: v *= (1 - friction * env_friction_scale * dt).
3. Update positions: pos += v * dt.
4. Wrap boundaries toroidally.

## Environmental Influences
- 2D grid (100x100 units) stores friction and force scale multipliers (0.5 to 2.0).
- Initialized with Perlin noise for smooth variation.
- Accessed in force and friction calculations via modular indexing.

## Dynamic Particle Appearance
- Size = ParticleSize * (1 + |velocity|/10).
- Opacity = 255 * max(0.3, 1 - neighbors/10).

## Particle Connections
- For each particle, check neighbors within interaction_radius (using bins).
- Draw line if |matrix[i][j]| > 0.1, with opacity = 255 * (1 - r/interaction_radius) * |a|.

## Rule Editor
- Render NxN grid (100x100 pixels) with colors: green for a > 0, red for a < 0, intensity = |a|.
- Mouse click selects cell (i,j), wheel adjusts matrix[i][j] by ±0.1, clamped [-1,1].

## Evolution Mode
Every 1000 ticks: Mutate matrix entries by ±0.1 (Gaussian noise), clamped [-1,1].

## Birth/Death
- Death: If <3 neighbors within interaction_radius, remove particle.
- Birth: In dense bins (>10 particles), add new particle nearby with random type.