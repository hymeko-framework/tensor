use hymeko_tensor::contraction::*;

// Independent subset-DP oracle for the documented index-product cost. It has no
// production dependency and does not obtain costs or axes from planner steps.
struct Oracle<'a>(&'a ContractionProblem);
impl Oracle<'_> {
    fn axes(&self, subset: usize) -> Vec<usize> {
        (0..self.0.extents().len())
            .filter(|axis| {
                let mut inside = false;
                let mut outside = false;
                for (i, factor) in self.0.factors().iter().enumerate() {
                    if factor.contains(axis) {
                        if subset & (1 << i) != 0 {
                            inside = true
                        } else {
                            outside = true
                        }
                    }
                }
                inside
                    && (subset.count_ones() == 1 || outside || self.0.output_axes().contains(axis))
            })
            .collect()
    }
    fn volume(&self, axes: &[usize]) -> u128 {
        axes.iter().map(|&a| self.0.extents()[a] as u128).product()
    }
    fn optimum(&self, cap: u128) -> Option<u128> {
        let count = 1usize << self.0.factors().len();
        let mut dp = vec![None; count];
        for (set, slot) in dp.iter_mut().enumerate().skip(1) {
            if set.count_ones() == 1 {
                *slot = Some(0u128);
            }
        }
        for set in 1..count {
            if set.count_ones() < 2 {
                continue;
            }
            if self.volume(&self.axes(set)) > cap {
                continue;
            }
            let mut a = (set - 1) & set;
            while a != 0 {
                let b = set ^ a;
                if a < b {
                    if let (Some(left), Some(right)) = (dp[a], dp[b]) {
                        let mut union = self.axes(a);
                        union.extend(self.axes(b));
                        union.sort_unstable();
                        union.dedup();
                        let cost = left + right + self.volume(&union);
                        if dp[set].is_none_or(|best| cost < best) {
                            dp[set] = Some(cost)
                        }
                    }
                }
                a = (a - 1) & set;
            }
        }
        dp[count - 1]
    }
}
fn budget(cap: u128) -> PlanningBudget {
    PlanningBudget {
        max_intermediate_rank: usize::MAX,
        max_states: 200_000,
        max_intermediate_elements: cap,
    }
}

#[test]
fn chain_and_hyperindex_optima_match_independent_subset_oracle() {
    let cases = [
        ContractionProblem::new(
            vec![10, 100, 5, 50],
            vec![vec![0, 1], vec![1, 2], vec![2, 3]],
            vec![0, 3],
        )
        .unwrap(),
        ContractionProblem::new(
            vec![3, 4, 5],
            vec![vec![0, 1], vec![1, 2], vec![1]],
            vec![0, 2],
        )
        .unwrap(),
        ContractionProblem::new(
            vec![2, 3, 4],
            vec![vec![0], vec![1], vec![2]],
            vec![0, 1, 2],
        )
        .unwrap(),
        ContractionProblem::new(vec![0, 9, 2], vec![vec![0, 1], vec![1, 2]], vec![0, 2]).unwrap(),
    ];
    for problem in &cases {
        for cap in [0, 1, 10, 100, 1000, u128::MAX] {
            let outcome = ContractionPlanner::plan(problem, budget(cap)).unwrap();
            assert!(outcome.complete);
            assert_eq!(
                outcome.plan.as_ref().map(|p| p.work()),
                Oracle(problem).optimum(cap)
            );
            if let Some(plan) = outcome.plan {
                assert!(plan.largest_intermediate() <= cap);
            }
        }
    }
    let plan = ContractionPlanner::plan(&cases[0], budget(u128::MAX))
        .unwrap()
        .plan
        .unwrap();
    assert_eq!(plan.work(), 7500);
    assert_eq!(plan.steps()[0].axes, vec![0, 2]);
    let result = ContractionPlanner::plan(&cases[1], budget(u128::MAX))
        .unwrap()
        .plan
        .unwrap();
    assert!(
        result.steps()[0].axes.contains(&1),
        "index shared by a third factor must survive"
    );
}

#[test]
fn deterministic_generated_factor_sets_match_exhaustive_optimum() {
    let mut seed = 0x8c04_19aau64;
    for _ in 0..70 {
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            seed
        };
        let n = 3 + (next() % 3) as usize;
        let dims: Vec<_> = (0..4).map(|_| 1 + (next() % 5) as usize).collect();
        let factors: Vec<Vec<usize>> = (0..n)
            .map(|_| {
                let mask = next() % 16;
                (0..4).filter(|a| mask & (1 << a) != 0).collect()
            })
            .collect();
        let output = (0..4)
            .filter(|a| factors.iter().filter(|f| f.contains(a)).count() == 1)
            .collect();
        let problem = ContractionProblem::new(dims, factors, output).unwrap();
        for cap in [8, 5000] {
            let answer = ContractionPlanner::plan(&problem, budget(cap)).unwrap();
            assert!(answer.complete);
            assert_eq!(answer.plan.map(|p| p.work()), Oracle(&problem).optimum(cap));
        }
    }
}

#[test]
fn budgets_do_not_masquerade_as_optimality_and_validation_is_checked() {
    let p = ContractionProblem::new(
        vec![4, 9, 2, 6],
        vec![vec![0, 1], vec![1, 2], vec![2, 3]],
        vec![0, 3],
    )
    .unwrap();
    for states in [0, 1, 2, 3] {
        let a = ContractionPlanner::plan(
            &p,
            PlanningBudget {
                max_intermediate_rank: usize::MAX,
                max_states: states,
                max_intermediate_elements: 1000,
            },
        )
        .unwrap();
        assert!(a.states <= states);
        assert!(a.plan.is_some());
        if states == 0 {
            assert!(!a.complete)
        }
    }
    let invalid = [
        (vec![], vec![], vec![], PlanningError::NoFactors),
        (vec![2], vec![vec![1]], vec![], PlanningError::InvalidAxes),
        (
            vec![2],
            vec![vec![0, 0]],
            vec![0],
            PlanningError::InvalidAxes,
        ),
        (vec![2], vec![vec![0]], vec![], PlanningError::UnpairedAxis),
        (vec![2], vec![vec![]], vec![0], PlanningError::UnpairedAxis),
        (
            vec![usize::MAX; 3],
            vec![vec![0, 1, 2]],
            vec![0, 1, 2],
            PlanningError::Overflow,
        ),
    ];
    for (dims, factors, out, error) in invalid {
        assert_eq!(
            ContractionProblem::new(dims, factors, out).unwrap_err(),
            error
        );
    }
    let scalar = ContractionProblem::new(vec![], vec![vec![]], vec![]).unwrap();
    assert!(
        ContractionPlanner::plan(&scalar, budget(0))
            .unwrap()
            .plan
            .is_none()
    );
    let one = ContractionPlanner::plan(&scalar, budget(1)).unwrap();
    assert!(one.complete);
    assert_eq!(one.plan.unwrap().work(), 0);
}

#[test]
fn bounded_search_improves_a_greedy_contraction_tree() {
    let p = ContractionProblem::new(
        vec![30, 35, 15, 5, 10, 20, 25],
        (0..6).map(|i| vec![i, i + 1]).collect(),
        vec![0, 6],
    )
    .unwrap();
    let greedy = ContractionPlanner::plan(
        &p,
        PlanningBudget {
            max_intermediate_rank: usize::MAX,
            max_states: 0,
            max_intermediate_elements: u128::MAX,
        },
    )
    .unwrap();
    let optimized = ContractionPlanner::plan(&p, budget(u128::MAX)).unwrap();
    assert!(optimized.complete);
    let before = greedy.plan.unwrap().work();
    let after = optimized.plan.unwrap().work();
    assert_eq!(Some(after), Oracle(&p).optimum(u128::MAX));
    assert_eq!(after, 15125);
    assert!(
        after < before,
        "branch-and-bound must improve this greedy witness: {before} -> {after}"
    );
    println!(
        "verified nominal contraction work: greedy={before}, optimum={after}, successors={}, pruned={}",
        optimized.states, optimized.pruned
    );
}

#[derive(Default)]
struct Trace {
    calls: usize,
    fail: bool,
}
impl ContractionVisitor for Trace {
    type Value = u64;
    type Error = &'static str;
    fn input(&mut self, factor: usize, _: &[usize]) -> Result<u64, Self::Error> {
        Ok(1 << factor)
    }
    fn contract(&mut self, left: u64, right: u64, _: &ContractionStep) -> Result<u64, Self::Error> {
        if self.fail {
            return Err("backend rejected operation");
        }
        assert_eq!(left & right, 0);
        self.calls += 1;
        Ok(left | right)
    }
}
#[test]
fn plans_replay_through_virtual_handle_visitor_and_propagate_errors() {
    let p = ContractionProblem::new(vec![2], vec![vec![0], vec![0], vec![0]], vec![]).unwrap();
    let plan = ContractionPlanner::plan(&p, budget(10))
        .unwrap()
        .plan
        .unwrap();
    let mut trace = Trace::default();
    assert_eq!(plan.execute(&mut trace).unwrap(), 7);
    assert_eq!(trace.calls, 2);
    assert_eq!(
        plan.execute(&mut Trace {
            fail: true,
            ..Trace::default()
        }),
        Err("backend rejected operation")
    );
}

#[test]
fn executor_rank_limit_is_a_hard_feasibility_constraint() {
    let problem =
        ContractionProblem::new(vec![2; 4], vec![vec![0, 1], vec![2, 3]], vec![0, 1, 2, 3])
            .unwrap();
    let mut limits = budget(100);
    limits.max_intermediate_rank = 3;
    let rejected = ContractionPlanner::plan(&problem, limits).unwrap();
    assert!(rejected.complete && rejected.plan.is_none());
    limits.max_intermediate_rank = 4;
    assert!(
        ContractionPlanner::plan(&problem, limits)
            .unwrap()
            .plan
            .is_some()
    );
    let one = ContractionProblem::new(vec![2, 2], vec![vec![0, 1]], vec![0, 1]).unwrap();
    limits.max_intermediate_rank = 1;
    assert!(
        ContractionPlanner::plan(&one, limits)
            .unwrap()
            .plan
            .is_none()
    );
}

#[test]
fn shared_bounded_search_matches_exhaustive_binary_choices_and_budget_contract() {
    struct Problem(Vec<[u128; 2]>);
    impl BoundedProblem for Problem {
        type State = (usize, u128);
        type Error = std::convert::Infallible;
        fn lower_bound(&self, s: &Self::State) -> u128 {
            s.1
        }
        fn complete(&self, s: &Self::State) -> bool {
            s.0 == self.0.len()
        }
        fn branches(&self, _: &Self::State) -> usize {
            2
        }
        fn branch(&self, s: &Self::State, i: usize) -> Result<Option<Self::State>, Self::Error> {
            Ok(Some((s.0 + 1, s.1 + self.0[s.0][i])))
        }
        fn order(&self, a: &Self::State, b: &Self::State) -> std::cmp::Ordering {
            a.1.cmp(&b.1)
        }
    }
    for seed in 0..32 {
        let problem = Problem(
            (0..6)
                .map(|i| {
                    [
                        ((i * 13 + seed) % 19) as u128,
                        ((i * 7 + seed * 3) % 23) as u128,
                    ]
                })
                .collect(),
        );
        let expected = (0..64)
            .map(|mask| {
                problem
                    .0
                    .iter()
                    .enumerate()
                    .map(|(i, c)| c[(mask >> i) & 1])
                    .sum::<u128>()
            })
            .min()
            .unwrap();
        let incumbent = (6, problem.0.iter().map(|c| c[0]).sum());
        let result = BranchAndBound::solve(&problem, (0, 0), Some(incumbent), 1000).unwrap();
        assert!(result.complete);
        assert_eq!(result.best.unwrap().1, expected);
        for budget in [0, 1, 2, 7] {
            let result = BranchAndBound::solve(&problem, (0, 0), Some(incumbent), budget).unwrap();
            assert!(result.states <= budget);
            assert!(result.best.unwrap().1 <= incumbent.1);
        }
    }
}
