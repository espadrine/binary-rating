package binaryrating

import (
	"math"
)

type CompetitorID uint64

type Comparison struct {
	HigherCompetitorID CompetitorID
	LowerCompetitorID  CompetitorID
}

type RatedCompetitor struct {
	ID           CompetitorID
	score        float64
	wins         uint64
	matches      uint64
	pairwiseWins map[CompetitorID]uint64
}

const (
	logisticScale = 250
)

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
					score:        1.0,
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
		hc := rcIdx[t.HigherCompetitorID]
		lc := rcIdx[t.LowerCompetitorID]
		(*hc).wins += 1
		(*hc).matches += 1
		(*lc).matches += 1
		pw := (*hc).pairwiseWins
		pw[t.LowerCompetitorID] += 1
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
				denom += float64(c.pairwiseWins[c2.ID]+c2.pairwiseWins[c.ID]) / (c.score + c2.score)
			}
			rc[i].score = float64(c.wins) / denom
		}
	}
	return rc
}

// The Bradley-Terry model gives precise comparative probabilities
// for any pair of competitors, even if they never matched together.
// However, the score is arbitrary.
//
// Assuming their instantaneous performance is taken
// from a probability distribution Ri whose mean is a numeric rating,
// they will only surpass a competitor j if their sampled rating
// is higher than that sampled from their competitor˚s, Rj.
// The probability of winning that comparison thus computes
// to the probability that the difference between the two random variables,
// ΔR[ij] = Ri - Rj = δ → ∫ Ri(x) Rj(x-δ) dx, is higher than zero,
// which is the area under the curve of a new probability distribution
// whose mean is the difference of the ratings of each player,
// and whose variance is irrelevant because the scale is arbitrary,
// but is always twice the variance of an individual
// assuming all competitors have the same.
//
// The function that gives a success probability
// as a function of the rating difference, is its cumulative distribution:
// Φ(x = Δrating) = ∫δ∈[0,∞] ΔR(δ; μ=x) dδ
//
// Since the scale is arbitrary,
// we pick one that is most practical for computing by head
// the relative probabilities associated with a rating difference.
// We choose to set Φ(10) = 0.510, which makes it so that a rating difference
// of Δ approximatively indicates a winning chance of 0.5 + Δ÷1000.
// In other words a 10-point difference indicates a 1% increase in win rate,
// and from linear approximation,
// being 250 points lower indicates roughly a 25% chance.
//
// We assume that instantaneous performance follows a Gumbel distribution.
// The difference follows the logistic distribution.
// The Gumbel is a good fit for human performance,
// because it can describe maximum effort.
// Then Φ(Δrating) = 1÷(1+exp(-Δrating÷s)).
// From Φ(10) = 0.51, we get:
// 1÷(1+exp(-10÷s)) = 0.51 ⇒ s = -10÷ln(1÷0.51 - 1) = 250.
// From Bradley-Terry, we have Φ(r1-r2) = s1÷(s1+s2) = 1÷(1+s2÷s1).
// Thus exp(-(r1-r2)÷s) = s2÷s1 ⇒ r1 = r2 + s×(ln(s1)-ln(s2)).
//
// These ideas are inspired by Arpad Elo’s paper; see page 137:
// https://www.gwern.net/docs/statistics/comparison/1978-elo-theratingofchessplayerspastandpresent.pdf

// Return a normalized rating, assuming competitors’ sampled performance
// is taken from a Gumbel distribution, or more generally,
// any rating difference follows a logistic distribution.
//
// A rating difference of Δ approximatively indicates
// a winning chance of 0.5 + Δ÷1000.
// In other words a 10-point difference indicates a 1% increase in win rate,
// and from linear approximation,
// being 250 points lower indicates roughly a 25% chance.
func (rc *RatedCompetitor) LogisticRating() float64 {
	return logisticScale * math.Log(rc.score)
}

// We want to compute the confidence interval of the rating
// assuming it is sampled from a Gumbel distribution.
// Essentially, we can consider each comparison like a single sampling.
// Since there are two ratings being compared,
// the comparison accounts for half a sampling on each rating.
// For N samples, the corresponding estimated mean would act like
// the average of the samples, and the variance would thus become
// var(ΣRi÷N) = Σvar(Ri)÷N² = σ²÷N where σ² is the constant rating variance.
// Per the central limit theorem, ΣRi is akin to a normal distribution.
// The interval is: rating ± sqrt(2σ²÷N) erf^-1(2ρ-1),
// where ρ is the confidence level (eg. 0.95).
//
// The variance of the Gumbel distribution is σ² = β²π²÷6.
// The variance of the difference of two Gumbel distributions
// (our rating difference) is 2σ² = s²π²÷3,
// where s is the scale of the corresponding logistic distribution.
// Since we computed the scale to be set to s = 250,
// we get σ² = 250²π²÷6.
// Thus the interval is: rating ± 250π×sqrt(⅔÷N) erf^-1(2ρ-1).

// Return the delta to the mean of the rating produced by LogisticRating(),
// for a confidence interval at level ρ
// (eg. 0.95 for a 95% confidence level).
//
// It is estimated from the number of comparisons the competitor had,
// assuming the difference in ratings follows a logistic distribution.
func (rc *RatedCompetitor) LogisticRatingConfidenceInterval(ρ float64) float64 {
	return (logisticScale * math.Pi * math.Sqrt(2.0/3)) / math.Sqrt(float64(rc.matches)) * math.Erfinv(2.0*ρ-1)
}
