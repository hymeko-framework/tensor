use hymeko_tensor::operator::{
    IndexedRow, OrderedAccumulator, RowContraction, RowTransform, RowVisitor, SparseRow,
};
use hymeko_tensor::renyi::ShannonEntropy;

#[test]
fn indexed_rows_preserve_offsets_duplicates_and_lazy_bounds() {
    let input = [2.0, 5.0, -3.0];
    for targets in [&[][..], &[2][..], &[1, 1][..], &[2, 0, 2][..]] {
        let row = IndexedRow::new(7, targets);
        assert_eq!(row.offset(), 7);
        assert_eq!(row.len(), targets.len());
        assert_eq!(row.is_empty(), targets.is_empty());
        assert_eq!(row.iter_targets().collect::<Vec<_>>(), targets);
        let mut ranges = Vec::new();
        row.for_each_range(|range| ranges.push(range));
        assert_eq!(
            ranges,
            targets
                .iter()
                .map(|&target| target..target + 1)
                .collect::<Vec<_>>()
        );
        // Literal fixtures and direct folds remain independent of the row kernel.
        let coefficient = |slot| (slot - 6) as f64;
        let expected = targets
            .iter()
            .enumerate()
            .fold(0.0_f64, |sum, (i, &v)| sum + input[v] * coefficient(7 + i));
        assert_eq!(
            row.gather::<f64>(&input, coefficient).to_bits(),
            expected.to_bits()
        );
        let mut output = [1.0; 3];
        let mut expected = output;
        for (i, &target) in targets.iter().enumerate() {
            expected[target] += 2.0 * coefficient(7 + i);
        }
        row.scatter(2.0, &mut output, coefficient);
        assert_eq!(output, expected);
    }
    for targets in [&[0, 3][..], &[0, 0, 3][..]] {
        let row = IndexedRow::new(7, targets);
        let calls = std::cell::RefCell::new(Vec::new());
        let coefficient = |slot| {
            calls.borrow_mut().push(slot);
            1.0
        };
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            row.gather::<f64>(&input, coefficient)
        }))
        .is_err());
        assert_eq!(
            *calls.borrow(),
            (7..7 + targets.len() - 1).collect::<Vec<_>>()
        );
        calls.borrow_mut().clear();
        let mut output = [1.0];
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            row.scatter(2.0, &mut output, coefficient);
        }))
        .is_err());
        assert_eq!(output, [1.0 + 2.0 * (targets.len() - 1) as f64]);
        assert_eq!(
            *calls.borrow(),
            (7..7 + targets.len() - 1).collect::<Vec<_>>()
        );
    }
}

#[test]
fn take_total_returns_exact_bits_and_restores_the_policy_default() {
    for bits in [
        0,
        1u64 << 63,
        1,
        0x3ff0000000000000,
        0x7ff0000000000000,
        0x7ff8000000000031,
    ] {
        let mut scalar = f64::from_bits(bits);
        assert_eq!(scalar.take_total().to_bits(), bits);
        assert_eq!(scalar.to_bits(), 0.0f64.to_bits());
    }
    struct NonzeroDefault(f64);
    impl Default for NonzeroDefault {
        fn default() -> Self {
            Self(2.0)
        }
    }
    impl OrderedAccumulator for NonzeroDefault {
        fn add(&mut self, value: f64) {
            self.0 += value;
        }
        fn total(&self) -> f64 {
            self.0
        }
    }
    let mut value = NonzeroDefault(7.0);
    assert_eq!(value.take_total(), 7.0);
    assert_eq!(value.total(), 2.0);
}

#[test]
fn row_gather_writes_partitioned_outputs_and_uses_the_shared_reduction_policy() {
    use hymeko_tensor::operator::RowGather;
    let mut output = [99.0, 99.0, 99.0];
    let mut gather = RowGather::new(
        &[2.0, 3.0, 5.0],
        &mut output[1..],
        |slot| (slot + 1) as f64,
        |row, value| value + row as f64,
    )
    .with_gather::<f64>()
    .starting_at(7);
    gather.visit(7, IndexedRow::new(0, &[2, 0, 2])); // 5*1 + 2*2 + 5*3 + 7.
    gather.visit(8, IndexedRow::new(0, &[]));
    assert_eq!(output, [99.0, 31.0, 8.0]);
    let mut disabled = [99.0];
    RowGather::new(
        &[],
        &mut disabled,
        |_| panic!("disabled coefficient"),
        Disabled,
    )
    .visit(0, IndexedRow::new(0, &[usize::MAX]));
    assert_eq!(disabled, [0.0]);
}

#[test]
fn fused_distinct_rows_and_same_row_visit_preserve_signed_duplicates() {
    let mut output = [0.0; 2];
    let mut contraction = RowContraction::new(
        &[2.0, 3.0],
        &mut output,
        |i| [1.0, -2.0][i],
        |i| [1.0, 2.0, 3.0][i],
        |_, v| v * 0.5,
    );
    assert_eq!(contraction.dimensions(), (2, 2));
    contraction.pair(
        0,
        &IndexedRow::new(0, &[0, 1]),
        &IndexedRow::new(0, &[1, 0, 1]),
    );
    assert_eq!(output, [-4.0, -8.0]);
    let mut output = [0.0; 2];
    RowContraction::new(&[2.0, 3.0], &mut output, |_| 1.0, |_| 1.0, |_, v| v)
        .visit(0, IndexedRow::new(0, &[0, 1, 1]));
    assert_eq!(output, [8.0, 16.0]);
}

struct Disabled;
impl RowTransform for Disabled {
    fn enabled(&self, _: usize) -> bool {
        false
    }
    fn apply(&self, _: usize, _: f64) -> f64 {
        panic!("disabled transform evaluated")
    }
}
#[test]
fn disabled_rows_do_not_read_vectors_or_coefficients() {
    RowContraction::<f64, _, _, _>::new(
        &[],
        &mut [],
        |_| panic!("source"),
        |_| panic!("sink"),
        Disabled,
    )
    .visit(0, IndexedRow::new(0, &[usize::MAX]));
}

#[test]
fn accumulation_and_entropy_contracts_cover_empty_extremes_and_scaling() {
    let row = IndexedRow::new(0, &[]);
    assert!(row.is_empty());
    assert_eq!(row.offset(), 0);
    assert_eq!(row.iter_targets().count(), 0);
    for terms in [
        &[][..],
        &[-0.0][..],
        &[1e16, 1.0, -1e16][..],
        &[f64::INFINITY, 1.0][..],
    ] {
        let expected = terms.iter().fold(0.0_f64, |a, &b| a + b).to_bits();
        assert_eq!(f64::sum(terms.iter().copied()).to_bits(), expected);
        assert_eq!(
            f64::sum_chunks(terms.chunks(2).map(|c| c.iter().copied())).to_bits(),
            expected
        );
    }
    assert_eq!(f64::sum_pair(|| 2.0, || 3.0), 5.0);
    assert_eq!(ShannonEntropy::from_counts([]).bits(), 0.0);
    assert_eq!(ShannonEntropy::from_counts([0, 0, 9]).bits(), 0.0);
    assert_eq!(ShannonEntropy::from_counts([1, 1]).bits(), 1.0);
    for scale in 1..100 {
        let h = ShannonEntropy::from_counts([scale, 3 * scale]).bits();
        assert!((h - 0.8112781244591328).abs() < 1e-12);
    }
    assert!((ShannonEntropy::from_counts([usize::MAX, usize::MAX]).bits() - 1.0).abs() < 1e-12);
}

#[test]
fn range_contraction_matches_independent_dense_multiset_oracle_and_reuses_buffers() {
    use hymeko_tensor::operator::{RangePartition, RangeWorkspace};
    assert!(RangePartition::new(4, vec![0..3, 2..4]).is_none());
    assert!(RangePartition::new(4, std::iter::once(1..5).collect()).is_none());
    assert!(RangePartition::new(4, std::iter::once(2..2).collect()).is_none());
    let partition = RangePartition::new(8, vec![0..3, 4..8]).unwrap();
    assert_eq!(partition.blocks(), &[0..3, 4..8]);
    assert!(partition.retained_bytes() > 0);
    let rows = [
        vec![0, 1, 2],
        vec![2, 1, 1],
        vec![],
        vec![4, 5, 6, 7],
        vec![7, 4, 4],
    ];
    let mut workspace = RangeWorkspace::<f64>::default();
    for (seed, mapping) in (0..20).flat_map(|seed| {
        [
            None,
            Some([7, 1, 5, 0, 6, 2, 4, 3]),
            Some([0, 0, 1, 2, 3, 3, 4, 4]),
        ]
        .map(|mapping| (seed, mapping))
    }) {
        let coordinate = |v: usize| mapping.as_ref().map_or(v, |order| order[v]);
        let input: Vec<_> = (0..8)
            .map(|i| ((i * 7 + seed * 3) % 17) as f64 - 8.0)
            .collect();
        let mut expected = vec![3.0; 8];
        for (i, row) in rows.iter().enumerate() {
            let sum: f64 = row.iter().map(|&v| input[coordinate(v)]).sum();
            for &v in row {
                expected[coordinate(v)] += sum * (i + 1) as f64;
            }
        }
        if mapping.is_some() {
            workspace.prepare_mapped(&input, &partition, coordinate);
        } else {
            workspace.prepare(&input, &partition);
        }
        let mut visitor = workspace.contraction(&partition, |i, v| v * (i + 1) as f64);
        for (i, row) in rows.iter().enumerate() {
            visitor.visit(i, IndexedRow::new(0, row));
        }
        let mut actual = vec![3.0; 8];
        if mapping.is_some() {
            workspace.accumulate_mapped(&partition, &mut actual, coordinate);
        } else {
            workspace.accumulate_into(&partition, &mut actual);
        }
        assert_eq!(actual, expected);
        assert!(workspace.retained_bytes() > 0);
        workspace.prepare(&input, &partition);
        workspace
            .contraction(&partition, Disabled)
            .visit(0, IndexedRow::new(0, &[usize::MAX]));
        let mut zero = vec![0.0; 8];
        workspace.accumulate_into(&partition, &mut zero);
        assert_eq!(zero, vec![0.0; 8]);
    }
    let empty = RangePartition::new(0, vec![]).unwrap();
    workspace.prepare(&[], &empty);
    workspace.accumulate_into(&empty, &mut []);
}

#[test]
fn gather_policy_is_independent_of_destination_accumulation() {
    #[derive(Default)]
    struct Terms(usize);
    impl OrderedAccumulator for Terms {
        fn add(&mut self, _: f64) {
            self.0 += 1;
        }
        fn total(&self) -> f64 {
            self.0 as f64
        }
    }
    let mut output = [0.0; 2];
    let mut visitor = RowContraction::new(&[10.0, 20.0], &mut output, |_| 1.0, |_| 1.0, |_, v| v)
        .with_gather::<Terms>();
    assert_eq!(visitor.dimensions(), (2, 2));
    visitor.pair(
        0,
        &IndexedRow::new(0, &[0, 1, 1]),
        &IndexedRow::new(0, &[0, 1]),
    );
    assert_eq!(output, [3.0, 3.0]);
}
