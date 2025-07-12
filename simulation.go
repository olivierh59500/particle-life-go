package main

import (
	"encoding/json"
	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/inpututil"
	"github.com/hajimehoshi/ebiten/v2/vector"
	"image/color"
//	"log"
	"math"
	"math/rand"
	"os"
	"sync"
	"time"
)

// Simulation constants
const (
	InteractionRadius = 50.0
	CollisionRadius   = 5.0
	CollisionStrength = 2.0
	DT                = 0.1
	ParticleSize      = 2.0
	MaxTypes          = 16
	MinZoom           = 0.1 // Limit zoom out to prevent excessive tiling
)

// Particle struct: Represents a single particle
type Particle struct {
	X, Y   float64 // Position
	VX, VY float64 // Velocity
	Type   int     // Type (0 to NumTypes-1)
	Trail  []struct{ X, Y float64 } // For trail visualization (last 10 positions)
}

// Bin for spatial partitioning: List of particle indices
type Bin []int

// Simulation struct: Holds the game state
type Simulation struct {
	Width, Height float64
	Particles     []*Particle
	NumTypes      int
	ForceMatrix   [][]float64 // Attraction/repulsion matrix [from][to]
	Friction      float64
	ForceScale    float64
	Paused        bool
	EvolutionMode bool
	LifecycleMode bool
	VisMode       int // 0: particles, 1: heatmap, 2: trails
	Zoom          float64
	CamX, CamY    float64 // Camera pan
	PrevMX, PrevMY float64 // Previous mouse position for drag
	GridSize      float64 // Grid cell size for binning
	Bins          map[int]Bin // Bin hash -> particle indices
	BinMutex      sync.Mutex // For concurrent bin updates
	TickCount     int // For evolution timing
	rng           *rand.Rand
}

// NewSimulation creates a new simulation instance
func NewSimulation(width, height float64, numParticles, numTypes int) *Simulation {
	s := &Simulation{
		Width:       width,
		Height:      height,
		NumTypes:    numTypes,
		Friction:    0.1,
		ForceScale:  1.0,
		Zoom:        1.0,
		GridSize:    InteractionRadius,
		Bins:        make(map[int]Bin),
		TickCount:   0,
		rng:         rand.New(rand.NewSource(time.Now().UnixNano())),
	}

	// Initialize force matrix randomly [-1,1]
	s.ForceMatrix = make([][]float64, numTypes)
	for i := range s.ForceMatrix {
		s.ForceMatrix[i] = make([]float64, numTypes)
		for j := range s.ForceMatrix[i] {
			s.ForceMatrix[i][j] = s.rng.Float64()*2 - 1
		}
	}

	// Create particles
	s.Particles = make([]*Particle, numParticles)
	for i := range s.Particles {
		s.Particles[i] = &Particle{
			X:     s.rng.Float64() * width,
			Y:     s.rng.Float64() * height,
			VX:    0,
			VY:    0,
			Type:  s.rng.Intn(numTypes),
			Trail: make([]struct{ X, Y float64 }, 0, 10),
		}
	}

	return s
}

// Update is called each tick by Ebitengine
func (s *Simulation) Update() error {
	// Handle input
	s.handleInput()

	if s.Paused {
		return nil
	}

	// Build spatial bins
	s.buildBins()

	// Compute forces and update velocities (parallel)
	s.computeForces()

	// Update positions and apply friction
	for _, p := range s.Particles {
		p.VX *= (1 - s.Friction*DT)
		p.VY *= (1 - s.Friction*DT)
		p.X += p.VX * DT
		p.Y += p.VY * DT

		// Toroidal wrap
		p.X = math.Mod(p.X+s.Width, s.Width)
		p.Y = math.Mod(p.Y+s.Height, s.Height)

		// Update trail
		if s.VisMode == 2 {
			p.Trail = append(p.Trail, struct{ X, Y float64 }{p.X, p.Y})
			if len(p.Trail) > 10 {
				p.Trail = p.Trail[1:]
			}
		}
	}

	// Lifecycle if enabled
	if s.LifecycleMode {
		s.applyLifecycle()
	}

	// Evolution if enabled
	s.TickCount++
	if s.EvolutionMode && s.TickCount%1000 == 0 {
		s.mutateMatrix()
	}

	return nil
}

// Draw is called each frame by Ebitengine
func (s *Simulation) Draw(screen *ebiten.Image) {
	screenWidth := float64(screen.Bounds().Dx())
	screenHeight := float64(screen.Bounds().Dy())

	// Calculate visible world range
	visibleMinX := s.CamX
	visibleMaxX := s.CamX + screenWidth / s.Zoom
	visibleMinY := s.CamY
	visibleMaxY := s.CamY + screenHeight / s.Zoom

	// Calculate tile ranges
	dxFrom := math.Floor(visibleMinX / s.Width)
	dxTo := math.Ceil(visibleMaxX / s.Width)
	dyFrom := math.Floor(visibleMinY / s.Height)
	dyTo := math.Ceil(visibleMaxY / s.Height)

	switch s.VisMode {
	case 0: // Particles
		for dx := dxFrom; dx < dxTo; dx++ {
			for dy := dyFrom; dy < dyTo; dy++ {
				offsetX := dx * s.Width
				offsetY := dy * s.Height
				for _, p := range s.Particles {
					wx := p.X + offsetX
					wy := p.Y + offsetY
					sx := s.worldToScreenX(wx)
					sy := s.worldToScreenY(wy)
					if sx >= -ParticleSize && sx <= screenWidth+ParticleSize && sy >= -ParticleSize && sy <= screenHeight+ParticleSize {
						col := s.typeColor(p.Type)
						vector.DrawFilledCircle(screen, float32(sx), float32(sy), float32(ParticleSize*s.Zoom), col, true)
					}
				}
			}
		}
	case 1: // Heatmap
		step := 10.0
		for dx := dxFrom; dx < dxTo; dx++ {
			for dy := dyFrom; dy < dyTo; dy++ {
				offsetX := dx * s.Width
				offsetY := dy * s.Height
				for x := 0.0; x < s.Width; x += step {
					for y := 0.0; y < s.Height; y += step {
						wx := x + offsetX
						wy := y + offsetY
						sx := s.worldToScreenX(wx)
						sy := s.worldToScreenY(wy)
						if sx >= -step*s.Zoom && sx <= screenWidth+step*s.Zoom && sy >= -step*s.Zoom && sy <= screenHeight+step*s.Zoom {
							// Sample at modular position since periodic
							forceMag := s.sampleForceAt(math.Mod(x+s.Width, s.Width), math.Mod(y+s.Height, s.Height))
							intensity := uint8(math.Min(forceMag*50, 255))
							col := color.RGBA{intensity, 0, 255 - intensity, 255}
							vector.DrawFilledRect(screen, float32(sx), float32(sy), float32(step*s.Zoom), float32(step*s.Zoom), col, true)
						}
					}
				}
			}
		}
	case 2: // Trails
		for dx := dxFrom; dx < dxTo; dx++ {
			for dy := dyFrom; dy < dyTo; dy++ {
				offsetX := dx * s.Width
				offsetY := dy * s.Height
				for _, p := range s.Particles {
					col := s.typeColor(p.Type)
					for i := 1; i < len(p.Trail); i++ {
						prevWX := p.Trail[i-1].X + offsetX
						prevWY := p.Trail[i-1].Y + offsetY
						currWX := p.Trail[i].X + offsetX
						currWY := p.Trail[i].Y + offsetY
						prevSX := s.worldToScreenX(prevWX)
						prevSY := s.worldToScreenY(prevWY)
						currSX := s.worldToScreenX(currWX)
						currSY := s.worldToScreenY(currWY)
						if (prevSX >= -1 && prevSX <= screenWidth+1 && prevSY >= -1 && prevSY <= screenHeight+1) ||
							(currSX >= -1 && currSX <= screenWidth+1 && currSY >= -1 && currSY <= screenHeight+1) {
							vector.StrokeLine(screen, float32(prevSX), float32(prevSY), float32(currSX), float32(currSY), 1, col, true)
						}
					}
				}
			}
		}
	}
}

// Layout returns the screen size
func (s *Simulation) Layout(outsideWidth, outsideHeight int) (int, int) {
	return int(s.Width), int(s.Height)
}

// handleInput processes keyboard and mouse input
func (s *Simulation) handleInput() {
	if inpututil.IsKeyJustPressed(ebiten.KeySpace) {
		s.Paused = !s.Paused
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyR) {
		s.randomizeMatrix()
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyE) {
		s.EvolutionMode = !s.EvolutionMode
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyB) {
		s.LifecycleMode = !s.LifecycleMode
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyH) {
		s.VisMode = (s.VisMode + 1) % 3
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyS) {
		s.saveMatrix("config.json")
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyL) {
		s.loadMatrix("config.json")
	}

	// Zoom
	_, wheelY := ebiten.Wheel()
	s.Zoom += wheelY * 0.1
	if s.Zoom < MinZoom {
		s.Zoom = MinZoom
	}

	// Pan (drag)
	mx, my := ebiten.CursorPosition()
	if ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) {
		s.CamX -= (float64(mx) - s.PrevMX) / s.Zoom
		s.CamY -= (float64(my) - s.PrevMY) / s.Zoom
	}
	s.PrevMX = float64(mx)
	s.PrevMY = float64(my)
}

// buildBins assigns particles to grid bins
func (s *Simulation) buildBins() {
	s.Bins = make(map[int]Bin)
	s.BinMutex.Lock()
	defer s.BinMutex.Unlock()

	for i, p := range s.Particles {
		binX := int(p.X / s.GridSize)
		binY := int(p.Y / s.GridSize)
		key := binX*10000 + binY // Simple hash
		s.Bins[key] = append(s.Bins[key], i)
	}
}

// computeForces calculates forces in parallel using goroutines
func (s *Simulation) computeForces() {
	var wg sync.WaitGroup

	// Temp velocity updates to avoid race conditions
	tempVX := make([]float64, len(s.Particles))
	tempVY := make([]float64, len(s.Particles))

	for key := range s.Bins {
		wg.Add(1)
		go func(binKey int) {
			defer wg.Done()
			bin := s.Bins[binKey]
			binX := binKey / 10000
			binY := binKey % 10000

			// Check this bin and 8 neighbors
			for dx := -1; dx <= 1; dx++ {
				for dy := -1; dy <= 1; dy++ {
					nKey := (binX+dx)*10000 + (binY + dy)
					nBin, ok := s.Bins[nKey]
					if !ok {
						continue
					}
					for _, i := range bin {
						p1 := s.Particles[i]
						for _, j := range nBin {
							if i == j {
								continue // Skip self
							}
							p2 := s.Particles[j]
							dx, dy := s.shortestDelta(p1.X-p2.X, p1.Y-p2.Y)
							r := math.Sqrt(dx*dx + dy*dy)
							if r == 0 {
								continue
							}

							// Collision force (repulsive)
							if r < CollisionRadius {
								f := CollisionStrength * (CollisionRadius - r) / CollisionRadius * s.ForceScale
								tempVX[i] += f * (dx / r) * DT
								tempVY[i] += f * (dy / r) * DT
							}

							// Interaction force
							if r < InteractionRadius {
								a := s.ForceMatrix[p1.Type][p2.Type]
								f := a * (InteractionRadius - r) / InteractionRadius * s.ForceScale
								dir := 1.0
								if a < 0 {
									dir = -1.0 // Repel away
									f = -f // Make positive for calc
								}
								tempVX[i] += dir * f * (dx / r) * DT
								tempVY[i] += dir * f * (dy / r) * DT
							}
						}
					}
				}
			}
		}(key)
	}

	wg.Wait()

	// Apply temp updates
	for i := range s.Particles {
		s.Particles[i].VX += tempVX[i]
		s.Particles[i].VY += tempVY[i]
	}
}

// shortestDelta computes delta with toroidal wrap
func (s *Simulation) shortestDelta(dx, dy float64) (float64, float64) {
	if dx > s.Width/2 {
		dx -= s.Width
	} else if dx < -s.Width/2 {
		dx += s.Width
	}
	if dy > s.Height/2 {
		dy -= s.Height
	} else if dy < -s.Height/2 {
		dy += s.Height
	}
	return dx, dy
}

// applyLifecycle handles birth and death
func (s *Simulation) applyLifecycle() {
	var toRemove []int
	for i, p := range s.Particles {
		neighbors := s.countNeighbors(p)
		if neighbors < 3 {
			toRemove = append(toRemove, i)
		}
	}

	// Remove dead
	for i := len(toRemove) - 1; i >= 0; i-- {
		idx := toRemove[i]
		s.Particles = append(s.Particles[:idx], s.Particles[idx+1:]...)
	}

	// Birth in dense bins
	for _, bin := range s.Bins {
		if len(bin) > 10 {
			// Add new particle near average position
			var avgX, avgY float64
			for _, idx := range bin {
				p := s.Particles[idx]
				avgX += p.X
				avgY += p.Y
			}
			avgX /= float64(len(bin))
			avgY /= float64(len(bin))
			newP := &Particle{
				X:    avgX + s.rng.Float64()*10 - 5,
				Y:    avgY + s.rng.Float64()*10 - 5,
				VX:   0,
				VY:   0,
				Type: s.rng.Intn(s.NumTypes),
			}
			s.Particles = append(s.Particles, newP)
		}
	}
}

// countNeighbors returns number of nearby particles
func (s *Simulation) countNeighbors(p *Particle) int {
	count := 0
	binX := int(p.X / s.GridSize)
	binY := int(p.Y / s.GridSize)
	for dx := -1; dx <= 1; dx++ {
		for dy := -1; dy <= 1; dy++ {
			key := (binX+dx)*10000 + (binY + dy)
			bin, ok := s.Bins[key]
			if !ok {
				continue
			}
			for _, j := range bin {
				p2 := s.Particles[j]
				if p == p2 {
					continue
				}
				dx, dy := s.shortestDelta(p.X-p2.X, p.Y-p2.Y)
				r := math.Sqrt(dx*dx + dy*dy)
				if r < InteractionRadius {
					count++
				}
			}
		}
	}
	return count
}

// mutateMatrix slightly changes the force matrix
func (s *Simulation) mutateMatrix() {
	for i := range s.ForceMatrix {
		for j := range s.ForceMatrix[i] {
			s.ForceMatrix[i][j] += s.rng.NormFloat64() * 0.1
			if s.ForceMatrix[i][j] > 1 {
				s.ForceMatrix[i][j] = 1
			} else if s.ForceMatrix[i][j] < -1 {
				s.ForceMatrix[i][j] = -1
			}
		}
	}
}

// randomizeMatrix resets matrix to random values
func (s *Simulation) randomizeMatrix() {
	for i := range s.ForceMatrix {
		for j := range s.ForceMatrix[i] {
			s.ForceMatrix[i][j] = s.rng.Float64()*2 - 1
		}
	}
}

// saveMatrix saves to JSON
func (s *Simulation) saveMatrix(filename string) {
	data, err := json.Marshal(s.ForceMatrix)
	if err != nil {
		return
	}
	err = os.WriteFile(filename, data, 0644)
	if err != nil {
		return
	}
}

// loadMatrix loads from JSON
func (s *Simulation) loadMatrix(filename string) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return
	}
	err = json.Unmarshal(data, &s.ForceMatrix)
	if err != nil {
		return
	}
}

// typeColor returns color for a type
func (s *Simulation) typeColor(t int) color.RGBA {
	// Simple hue-based colors
	h := float64(t) / float64(s.NumTypes) * 360
	r, g, b := hsvToRGB(h, 1, 1)
	return color.RGBA{uint8(r * 255), uint8(g * 255), uint8(b * 255), 255}
}

// hsvToRGB helper
func hsvToRGB(h, s, v float64) (float64, float64, float64) {
	h = math.Mod(h, 360)
	c := v * s
	x := c * (1 - math.Abs(math.Mod(h/60, 2)-1))
	m := v - c
	var r, g, b float64
	switch {
	case h < 60:
		r, g, b = c, x, 0
	case h < 120:
		r, g, b = x, c, 0
	case h < 180:
		r, g, b = 0, c, x
	case h < 240:
		r, g, b = 0, x, c
	case h < 300:
		r, g, b = x, 0, c
	default:
		r, g, b = c, 0, x
	}
	return r + m, g + m, b + m
}

// sampleForceAt computes total force magnitude at a point (for heatmap)
func (s *Simulation) sampleForceAt(px, py float64) float64 {
	mag := 0.0
	for _, p := range s.Particles {
		dx, dy := s.shortestDelta(px-p.X, py-p.Y)
		r := math.Sqrt(dx*dx + dy*dy)
		if r < InteractionRadius {
			// Assume average type for sampling
			a := 0.5 // Placeholder
			f := math.Abs(a) * (InteractionRadius - r) / InteractionRadius
			mag += f
		}
	}
	return mag
}

// worldToScreenX/Y for camera
func (s *Simulation) worldToScreenX(wx float64) float64 {
	return (wx - s.CamX) * s.Zoom
}
func (s *Simulation) worldToScreenY(wy float64) float64 {
	return (wy - s.CamY) * s.Zoom
}
