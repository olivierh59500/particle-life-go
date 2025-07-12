# Particle Life Simulation in Go + Ebitengine

This is an implementation of the Particle Life simulation, inspired by the blog post at https://lisyarus.github.io/blog/posts/particle-life-simulation-in-browser-using-webgpu.html. It simulates particles of different types interacting via asymmetric attraction/repulsion rules, leading to emergent life-like behaviors.

## Features
- Core simulation with asymmetric force matrix, linear forces, friction, and toroidal boundaries.
- Up to 16 particle types with randomizable matrix.
- Extensions beyond original:
  - Evolutionary mode: Matrix mutates slightly over time.
  - Particle lifecycle: Birth in dense areas, death when isolated.
  - In-game UI for tweaking parameters (friction, scale, etc.).
  - Visualization modes: Standard particles, force heatmap, particle trails.
  - Save/load rules as JSON.
- Optimized with spatial grid binning and Go concurrency for force computation.
- Rendering via Ebitengine for smooth 2D graphics.

## Requirements
- Go 1.20+
- Ebitengine: `go get github.com/hajimehoshi/ebiten/v2`

## How to Run
1. Clone the repo.
2. `go mod tidy`
3. `go run .`

Controls:
- Space: Pause/resume.
- R: Randomize matrix.
- E: Toggle evolutionary mode.
- B: Toggle birth/death.
- H: Cycle visualization modes (particles, heatmap, trails).
- S: Save matrix to config.json.
- L: Load from config.json.
- Mouse wheel: Zoom.
- Drag: Pan view.

## Performance
Handles 10k+ particles at 60 FPS on modern CPUs (uses goroutines for parallelism). Grid binning reduces O(N²) to near O(N).

See ALGORITHMS.md for technical details and SIMULATION.md for rule explanations.