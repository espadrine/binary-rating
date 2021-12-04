package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"github.com/espadrine/binaryrating"
)

const (
  populationSize = 10
  numberOfComparisons = 100
  // The true rating mean is actually how uniform
  // the exponential sampling is, ie. how much overlap there is
  // in sampled ratings between competitors (high = low overlap).
  trueRatingMean = 1000
)

func main() {
	// We have a set of people that have a hidden ranking.
	pop := newPopulation(populationSize)
	popIndex := buildPopulationIndex(pop)

	// We gather information from binary comparisons.
	// Sadly, these comparisons are lossy: during them,
	// each person compared pairwise has a momentary performance ranking which is
	// taken from a logistic distribution whose mean is the hidden ranking,
	// and whose variance is constant.
	tournament := newTournament(pop, numberOfComparisons)

	// Run the Minorization-Maximization algorithm on the Bradley-Terry model
	// applied to the tournament.
	rated := binaryrating.EstimateRatings(tournament)
	sort.Slice(rated, func(i, j int) bool {
		return rated[i].LogisticRating() < rated[j].LogisticRating()
	})

	// Since the rating’s zero value is arbitrary,
	// we offset the ratings to match, so we can compare with true rating.
	offset := float64(0)
	for _, c := range rated {
		offset += c.LogisticRating() - ratingFromScore(popIndex[c.ID].trueScore)
	}
	offset /= float64(len(rated))

	for _, c := range rated {
		fmt.Println(c.ID, "Estimated:", c.LogisticRating(), "±", c.LogisticRatingConfidenceInterval(0.99), "\tTrue:", ratingFromScore(popIndex[c.ID].trueScore) + offset)
	}
}

type Competitor struct {
	id        binaryrating.CompetitorID
	trueScore float64
}

func newPopulation(size int) []Competitor {
	pop := make([]Competitor, size)
	for i := 0; i < size; i++ {
		// Assume the true ratings are exponentially distributed.
		// In other words: there are more people with lower ratings.
		rating := trueRatingMean * rand.ExpFloat64()
		pop[binaryrating.CompetitorID(i)] = Competitor{
		  id: binaryrating.CompetitorID(i),
		  trueScore: scoreFromRating(rating),
		}
	}
	return pop
}

func ratingFromScore(score float64) float64 {
  return 250*math.Log(score)
}

func scoreFromRating(rating float64) float64 {
  return math.Exp(rating/250)
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
		winProb := c0.trueScore / (c0.trueScore + c1.trueScore)
		if winProb > rand.Float64() {
			comp[i] = binaryrating.Comparison{c0.id, c1.id}
		} else {
			comp[i] = binaryrating.Comparison{c1.id, c0.id}
		}
	}
	return comp
}
