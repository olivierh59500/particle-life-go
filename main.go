package main

import (
	"github.com/hajimehoshi/ebiten/v2"
	"log"
)

func main() {
	// Initialize simulation with default parameters
	sim := NewSimulation(800, 600, 5000, 4)

	// Set up Ebitengine game
	ebiten.SetWindowSize(800, 600)
	ebiten.SetWindowTitle("Particle Life Simulation")
	ebiten.SetTPS(60) // Target 60 ticks per second

	// Run the game loop
	if err := ebiten.RunGame(sim); err != nil {
		log.Fatal(err)
	}
}