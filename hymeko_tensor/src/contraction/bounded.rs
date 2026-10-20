//! One budgeted branch-and-bound engine for shape and structural plans.

/// A discrete minimization problem. Bounds must never exceed the cost of any
/// feasible completion. Ordering is a heuristic and is never used for pruning.
pub trait BoundedProblem {
    type State;
    type Error;
    fn lower_bound(&self, state: &Self::State) -> u128;
    fn complete(&self, state: &Self::State) -> bool;
    fn branches(&self, state: &Self::State) -> usize;
    fn branch(&self, state: &Self::State, index: usize)
    -> Result<Option<Self::State>, Self::Error>;
    fn order(&self, left: &Self::State, right: &Self::State) -> std::cmp::Ordering;
}

/// The incumbent remains usable when the explicit successor budget is exhausted.
pub struct BoundedOutcome<S> {
    pub best: Option<S>,
    pub complete: bool,
    pub states: usize,
    pub pruned: usize,
}

pub struct BranchAndBound;
impl BranchAndBound {
    /// `incumbent`, when supplied, must be a feasible complete state. For a
    /// complete state `lower_bound` is its actual objective cost.
    pub fn solve<P: BoundedProblem>(
        problem: &P,
        initial: P::State,
        incumbent: Option<P::State>,
        max_states: usize,
    ) -> Result<BoundedOutcome<P::State>, P::Error> {
        assert!(incumbent.as_ref().is_none_or(|s| problem.complete(s)));
        let mut result = BoundedOutcome {
            best: incumbent,
            complete: true,
            states: 0,
            pruned: 0,
        };
        let mut stack = vec![initial];
        'search: while let Some(state) = stack.pop() {
            if result
                .best
                .as_ref()
                .is_some_and(|b| problem.lower_bound(&state) >= problem.lower_bound(b))
            {
                result.pruned += 1;
                continue;
            }
            if problem.complete(&state) {
                result.best = Some(state);
                continue;
            }
            let mut children = Vec::new();
            for index in 0..problem.branches(&state) {
                if result.states == max_states {
                    result.complete = false;
                    break 'search;
                }
                result.states += 1;
                match problem.branch(&state, index)? {
                    Some(next)
                        if result.best.as_ref().is_none_or(|b| {
                            problem.lower_bound(&next) < problem.lower_bound(b)
                        }) =>
                    {
                        if problem.complete(&next) {
                            result.best = Some(next);
                        } else {
                            children.push(next);
                        }
                    }
                    _ => result.pruned += 1,
                }
            }
            children.sort_by(|a, b| problem.order(a, b));
            stack.extend(children.into_iter().rev());
        }
        Ok(result)
    }
}
