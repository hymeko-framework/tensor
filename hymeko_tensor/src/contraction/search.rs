//! Deterministic iterative branch-and-bound. Floating heuristics only order work.
use super::*;

#[derive(Clone)]
struct Value {
    id: usize,
    axes: Vec<usize>,
}
#[derive(Clone)]
struct State {
    live: Vec<Value>,
    steps: Vec<ContractionStep>,
    work: u128,
    peak: u128,
}
impl State {
    fn initial(problem: &ContractionProblem) -> Self {
        Self {
            live: problem
                .factors
                .iter()
                .enumerate()
                .map(|(id, axes)| Value {
                    id,
                    axes: axes.clone(),
                })
                .collect(),
            steps: Vec::new(),
            work: 0,
            peak: 0,
        }
    }
    fn merge(
        &self,
        problem: &ContractionProblem,
        i: usize,
        j: usize,
        budget: PlanningBudget,
    ) -> Result<Option<Self>, PlanningError> {
        let mut union = self.live[i].axes.clone();
        union.extend_from_slice(&self.live[j].axes);
        union.sort_unstable();
        union.dedup();
        let work = problem.volume(&union);
        let axes: Vec<_> = union
            .into_iter()
            .filter(|a| {
                problem.output.binary_search(a).is_ok()
                    || self
                        .live
                        .iter()
                        .enumerate()
                        .any(|(k, v)| k != i && k != j && v.axes.binary_search(a).is_ok())
            })
            .collect();
        let elements = problem.volume(&axes);
        if elements > budget.max_intermediate_elements || axes.len() > budget.max_intermediate_rank
        {
            return Ok(None);
        }
        let total = self.work.checked_add(work).ok_or(PlanningError::Overflow)?;
        let step = ContractionStep {
            left: self.live[i].id,
            right: self.live[j].id,
            axes: axes.clone(),
            work,
            elements,
        };
        let mut next = self.clone();
        next.live.remove(j);
        next.live.remove(i);
        next.live.push(Value {
            id: problem.factors.len() + self.steps.len(),
            axes,
        });
        next.steps.push(step);
        next.work = total;
        next.peak = self.peak.max(elements);
        Ok(Some(next))
    }
    fn into_plan(self, problem: &ContractionProblem) -> ContractionPlan<'_> {
        ContractionPlan {
            problem,
            steps: self.steps,
            work: self.work,
            peak: self.peak,
        }
    }
}

/// Shape-only planner. It borrows no numerical values and mutates no input graph.
pub struct ContractionPlanner;
impl ContractionPlanner {
    /// Find a low-work feasible contraction tree, bounded by an explicit budget.
    ///
    /// The incumbent is initialized greedily. Cumulative nonnegative exact work
    /// is an admissible lower bound on every completion; entropy never prunes.
    /// `complete` proves optimality only for the stated model and memory cap.
    pub fn plan(
        problem: &ContractionProblem,
        budget: PlanningBudget,
    ) -> Result<PlanningOutcome<'_>, PlanningError> {
        let initial = State::initial(problem);
        if initial.live.len() == 1 {
            let peak = problem.volume(&initial.live[0].axes);
            return Ok(PlanningOutcome {
                plan: (peak <= budget.max_intermediate_elements
                    && initial.live[0].axes.len() <= budget.max_intermediate_rank)
                    .then(|| ContractionPlan {
                        problem,
                        steps: Vec::new(),
                        work: 0,
                        peak,
                    }),
                complete: true,
                states: 0,
                pruned: usize::from(
                    peak > budget.max_intermediate_elements
                        || initial.live[0].axes.len() > budget.max_intermediate_rank,
                ),
            });
        }
        let best = Self::greedy(problem, &initial, budget)?;
        let outcome = BranchAndBound::solve(
            &ShapeSearch { problem, budget },
            initial,
            best,
            budget.max_states,
        )?;
        Ok(PlanningOutcome {
            plan: outcome.best.map(|state| state.into_plan(problem)),
            complete: outcome.complete,
            states: outcome.states,
            pruned: outcome.pruned,
        })
    }
    fn order(problem: &ContractionProblem, a: &State, b: &State) -> std::cmp::Ordering {
        a.work.cmp(&b.work).then(a.peak.cmp(&b.peak)).then_with(|| {
            // Distribution of surviving axis extents: ordering only, never a bound.
            let entropy = |s: &State| {
                crate::renyi::ShannonEntropy::from_counts(
                    s.live
                        .last()
                        .into_iter()
                        .flat_map(|v| v.axes.iter().map(|&axis| problem.extents[axis])),
                )
                .bits()
            };
            entropy(a).total_cmp(&entropy(b))
        })
    }
    fn greedy(
        problem: &ContractionProblem,
        initial: &State,
        budget: PlanningBudget,
    ) -> Result<Option<State>, PlanningError> {
        let mut state = initial.clone();
        while state.live.len() > 1 {
            let mut selected: Option<State> = None;
            for i in 0..state.live.len() {
                for j in i + 1..state.live.len() {
                    if let Some(candidate) = state.merge(problem, i, j, budget)? {
                        if selected
                            .as_ref()
                            .is_none_or(|s| Self::order(problem, &candidate, s).is_lt())
                        {
                            selected = Some(candidate);
                        }
                    }
                }
            }
            let Some(next) = selected else {
                return Ok(None);
            };
            state = next;
        }
        Ok(Some(state))
    }
}

struct ShapeSearch<'a> {
    problem: &'a ContractionProblem,
    budget: PlanningBudget,
}
impl BoundedProblem for ShapeSearch<'_> {
    type State = State;
    type Error = PlanningError;
    fn lower_bound(&self, state: &State) -> u128 {
        state.work
    }
    fn complete(&self, state: &State) -> bool {
        state.live.len() == 1
    }
    fn branches(&self, state: &State) -> usize {
        state.live.len() * (state.live.len() - 1) / 2
    }
    fn branch(&self, state: &State, mut index: usize) -> Result<Option<State>, PlanningError> {
        for i in 0..state.live.len() {
            let width = state.live.len() - i - 1;
            if index < width {
                return state.merge(self.problem, i, i + 1 + index, self.budget);
            }
            index -= width;
        }
        unreachable!("bounded search uses a valid branch ordinal")
    }
    fn order(&self, a: &State, b: &State) -> std::cmp::Ordering {
        ContractionPlanner::order(self.problem, a, b)
    }
}
