package main

import (
	"fmt"
	"math/rand"
	"sort"
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
	tsize := 100 // number of binary comparisons in the tournament.
	tournament := newTournament(pop, tsize)

	// Run the Minorization-Maximization algorithm on the Bradley-Terry model
	// applied to the tournament.
	rated := EstimateRatings(tournament)
	sort.Slice(rated, func(i, j int) bool {
		return rated[i].Rating < rated[j].Rating
	})
	for _, c := range rated {
		fmt.Println(c.ID, "Estimated:", c.Rating, "\tTrue:", popIndex[c.ID].trueRating)
	}
}

type CompetitorID uint64

type Competitor struct {
	id         CompetitorID
	trueRating float64
}

func newPopulation(size int) []Competitor {
	pop := make([]Competitor, size)
	for i := 0; i < size; i++ {
		// Assume the true ratings are normally distributed.
		pop[CompetitorID(i)] = Competitor{CompetitorID(i), rand.NormFloat64()}
	}
	return pop
}

func buildPopulationIndex(population []Competitor) map[CompetitorID]Competitor {
	htPop := make(map[CompetitorID]Competitor)
	for i := 0; i < len(population); i++ {
		htPop[population[i].id] = population[i]
	}
	return htPop
}

func newTournament(population []Competitor, size int) []Comparison {
	comp := make([]Comparison, size)
	for i := 0; i < size; i++ {
		// Select two competitors uniformly.
		c0 := population[rand.Intn(len(population))]
		c1 := population[rand.Intn(len(population))]
		if c0.id == c1.id {
			i -= 1
			continue
		}
		// FIXME: sample a rating from a logistic distribution.
		if c0.trueRating < c1.trueRating {
			comp[i] = Comparison{c1.id, c0.id}
		} else {
			comp[i] = Comparison{c0.id, c1.id}
		}
	}
	return comp
}

// ----

type Comparison struct {
	HigherCompetitorID CompetitorID
	LowerCompetitorID  CompetitorID
}

type RatedCompetitor struct {
	ID           CompetitorID
	Rating       float64
	wins         uint64
	pairwiseWins map[CompetitorID]uint64
}

// Take a list of sampled comparisons between competitors that each have a
// hidden rating such that the likelihood of a competitor being selected as
// higher than another is:
//
//     Pr(i > j) = rating[i] / (rating[i] + rating[j])
//
// This is also known as the Bradley-Terry model:
// https://sci-hub.se/10.2307/2334029
//
// Run the Minorization-Maximization algorithm on the samples
// to estimate their rating with arbitrary accuracy.
//
// cf. https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.110.7878&rep=rep1&type=pdf
func EstimateRatings(tournament []Comparison) []RatedCompetitor {
	// Initialize the competitors.
	var rc []RatedCompetitor
	rcSet := make(map[CompetitorID]bool)
	for _, t := range tournament {
		pair := [2]CompetitorID{t.HigherCompetitorID, t.LowerCompetitorID}
		for _, cid := range pair {
			if _, present := rcSet[cid]; !present {
				rc = append(rc, RatedCompetitor{
					ID:           cid,
					Rating:       1.0,
					wins:         0,
					pairwiseWins: make(map[CompetitorID]uint64),
				})
				rcSet[cid] = true
			}
		}
	}

	// As part of the initialization,
	// we pretend that each competitor won once and lost once
	// against every other competitor.
	// That has the effect of taking into account how sensitive the result is
	// to a single added comparison in the tournament.
	for i, c := range rc {
		rc[i].wins = uint64(len(rc) - 1)
		for _, c2 := range rc {
			if c.ID == c2.ID {
				continue
			}
			c.pairwiseWins[c2.ID] = 1
		}
	}

	// Build ID-based index.
	rcIdx := make(map[CompetitorID]*RatedCompetitor, len(rc))
	for i, c := range rc {
		rcIdx[c.ID] = &rc[i]
	}

	// Compute the number of wins for each competitor.
	for _, t := range tournament {
		c := rcIdx[t.HigherCompetitorID]
		(*c).wins += 1
		pw := (*c).pairwiseWins
		if _, present := pw[t.LowerCompetitorID]; !present {
			pw[t.LowerCompetitorID] = 1
		} else {
			pw[t.LowerCompetitorID] += 1
		}
	}

	// FIXME: switch from a fixed-iteration limit (which may not be enough)
	// to a likelihood-based one.
	for i := 0; i < 100; i++ {
		for i, c := range rc {
			denom := float64(0)
			for _, c2 := range rc {
				if c.ID == c2.ID {
					continue
				}
				denom += float64(c.pairwiseWins[c2.ID]+c2.pairwiseWins[c.ID]) / (c.Rating + c2.Rating)
			}
			rc[i].Rating = float64(c.wins) / denom
		}
	}
	return rc
}
