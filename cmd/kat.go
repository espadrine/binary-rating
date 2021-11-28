package main

import (
	"fmt"
	"math/rand"
	"sort"
	"github.com/espadrine/binaryrating"
)

const (
  trueRatingMean = 400
)

func main() {
	// We have a set of people that have a hidden ranking.
	pop := newPopulation(10)
	popIndex := buildPopulationIndex(pop)

	// We gather information from binary comparisons.
	// Sadly, these comparisons are lossy: during them,
	// each person compared pairwise has a momentary performance ranking which is
	// taken from a logistic distribution whose mean is the hidden ranking,
	// and whose variance is constant.
	tsize := 400 // number of binary comparisons in the tournament.
	tournament := newTournament(pop, tsize)

	// Run the Minorization-Maximization algorithm on the Bradley-Terry model
	// applied to the tournament.
	rated := binaryrating.EstimateRatings(tournament)
	sort.Slice(rated, func(i, j int) bool {
		return rated[i].LogisticRating() < rated[j].LogisticRating()
	})
	for _, c := range rated {
		fmt.Println(c.ID, "Estimated:", c.LogisticRating(), "\tTrue:", popIndex[c.ID].trueRating)
	}
}

type Competitor struct {
	id         binaryrating.CompetitorID
	trueRating float64
}

func newPopulation(size int) []Competitor {
	pop := make([]Competitor, size)
	for i := 0; i < size; i++ {
		// Assume the true ratings are normally distributed.
		pop[binaryrating.CompetitorID(i)] = Competitor{
		  id: binaryrating.CompetitorID(i),
		  trueRating: trueRatingMean * rand.ExpFloat64(),
		}
	}
	return pop
}

func buildPopulationIndex(population []Competitor) map[binaryrating.CompetitorID]Competitor {
	htPop := make(map[binaryrating.CompetitorID]Competitor)
	for i := 0; i < len(population); i++ {
		htPop[population[i].id] = population[i]
	}
	return htPop
}

func newTournament(population []Competitor, size int) []binaryrating.Comparison {
	comp := make([]binaryrating.Comparison, size)
	for i := 0; i < size; i++ {
		// Select two competitors uniformly.
		c0 := population[rand.Intn(len(population))]
		c1 := population[rand.Intn(len(population))]
		if c0.id == c1.id {
			i -= 1
			continue
		}
		winProb := c0.trueRating / (c0.trueRating + c1.trueRating)
		if winProb > rand.Float64() {
			comp[i] = binaryrating.Comparison{c0.id, c1.id}
		} else {
			comp[i] = binaryrating.Comparison{c1.id, c0.id}
		}
	}
	return comp
}
