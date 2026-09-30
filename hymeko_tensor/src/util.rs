use crate::common::Real;
use crate::representations::tensor_coo::TensorCoo;

pub fn print_dense_block<F: Real>(
    coo: &TensorCoo<F>,
    k_sel: usize,
    row0: usize,
    col0: usize,
    rows: usize,
    cols: usize,
) {
    assert!(k_sel < coo.num_slices, "k out of range");
    assert!(row0 < coo.dim_i && col0 < coo.dim_j, "start out of range");

    let r = rows.min(coo.dim_i - row0);
    let c = cols.min(coo.dim_j - col0);

    let mut block = vec![vec![F::zero(); c]; r];

    for e in coo.iter() {
        if e.k != k_sel {
            continue;
        }
        let i = e.i;
        let j = e.j;

        if i >= row0 && i < row0 + r && j >= col0 && j < col0 + c {
            block[i - row0][j - col0] += e.v;
        }
    }

    println!(
        "slice k={k_sel}, block rows [{row0}..{}), cols [{col0}..{})",
        row0 + r,
        col0 + c
    );
    let eps = F::from_other(1e-6_f64);

    for row in &block {
        for &x in row {
            if (x - x.round()).abs() < eps {
                print!("{:>3} ", x.round());
            } else {
                print!("{:>6.2} ", x);
            }
        }
        println!();
    }
}
