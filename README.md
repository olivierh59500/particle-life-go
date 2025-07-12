# Particle Life Simulation in Go + Ebitengine

This is an enhanced implementation of the Particle Life simulation, inspired by https://lisyarus.github.io/blog/posts/particle-life-simulation-in-browser-using-webgpu.html. It simulates particles of different types interacting via asymmetric attraction/repulsion rules, leading to emergent life-like behaviors.

## Features
- Core simulation with asymmetric force matrix, linear forces, friction, and toroidal boundaries.
- Up to 16 particle types with randomizable matrix.
- Enhanced features:
  - **Dynamic Particle Appearance**: Particle size scales with velocity, opacity decreases with crowding.
  - **Particle Connections**: Faint lines between interacting particles visualize relationships.
  - **User-Driven Particle Injection**: Right-click to add particles, select type with 1-9 keys.
  - **Environmental Influences**: Spatial variations in friction and force scale via Perlin noise.
  - **Rule Editor**: Edit force matrix in real-time with a graphical interface (toggle with M).
  - Evolutionary mode: Matrix mutates slightly over time.
  - Particle lifecycle: Birth in dense areas, death when isolated.
  - Visualization modes: Particles, heatmap, trails.
  - Save/load rules as JSON.
- Optimized with spatial grid binning and Go concurrency for force computation.
- Rendering via Ebitengine for smooth 2D graphics.

## Requirements
- Go 1.20+
- Ebitengine: `go get github.com/hajimehoshi/ebiten/v2@v2.8.8`
- Perlin noise: `go get github.com/aquilax/go-perlin`
- A TTF font file (e.g., Go-Mono from https://fonts.google.com/specimen/Go+Mono) placed in `assets/font.ttf`

## Setup
1. Clone the repo.
2. Download a TTF font (e.g., Go-Mono from https://fonts.google.com/specimen/Go+Mono) and save it as `assets/font.ttf`.
3. Run `go mod tidy` to fetch dependencies.
4. Run `go run .` to start the simulation.

## Controls
- **Space**: Pause/resume.
- **R**: Randomize matrix.
- **E**: Toggle evolutionary mode.
- **B**: Toggle birth/death.
- **H**: Cycle visualization modes (particles, heatmap, trails).
- **S**: Save matrix to config.json.
- **L**: Load from config.json.
- **M**: Toggle rule editor.
- **P**: Toggle debug info (FPS, particle count).
- **1-9**: Select particle type for injection.
- **Right-click**: Inject particle at mouse position.
- **Left-drag**: Pan (disabled in rule editor).
- **Left-click in rule editor**: Select cell, use wheel to adjust value.
- **Mouse wheel**: Zoom (min 0.1).

## Performance
Handles 10k+ particles at 60 FPS on modern CPUs (uses goroutines for parallelism). Grid binning reduces O(N²) to near O(N).

## Notes
- If `assets/font.ttf` is missing, text rendering (UI, debug info) is disabled, but the simulation runs normally.
- For production, replace the font with your preferred TTF file.
- Ensure Ebitengine version is v2.8.8 for compatibility.

See ALGORITHMS.md for technical details and SIMULATION.md for rule explanations.