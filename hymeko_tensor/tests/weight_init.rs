// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use hymeko_tensor::conv::weight_init::van_der_corput;
    use hymeko_tensor::conv::weight_init::{
        Constant, Kaiming, Ones, WeightInit, Xavier, XavierRandom, Zeros,
    };

    #[test]
    fn xavier_bounds() {
        let init = Xavier;
        let w: Vec<f64> = init.create(128, 64);
        let scale = (6.0_f64 / (128.0 + 64.0)).sqrt();
        for &v in &w {
            assert!(v.abs() <= scale + 1e-10, "got {v}, limit {scale}");
        }
    }

    #[test]
    fn kaiming_bounds() {
        let init = Kaiming;
        let w: Vec<f64> = init.create(256, 128);
        let scale = (6.0_f64 / 256.0).sqrt();
        for &v in &w {
            assert!(v.abs() <= scale + 1e-10);
        }
    }

    #[test]
    fn xavier_random_seeded_deterministic() {
        let a: Vec<f64> = XavierRandom { seed: 42 }.create(64, 32);
        let b: Vec<f64> = XavierRandom { seed: 42 }.create(64, 32);
        assert_eq!(a, b, "Same seed must produce same weights");
    }

    #[test]
    fn xavier_random_different_seeds() {
        let a: Vec<f64> = XavierRandom { seed: 1 }.create(64, 32);
        let b: Vec<f64> = XavierRandom { seed: 2 }.create(64, 32);
        assert_ne!(a, b);
    }

    #[test]
    fn zeros_and_ones() {
        let z: Vec<f64> = Zeros.create(4, 4);
        assert!(z.iter().all(|&v| v == 0.0));
        let o: Vec<f64> = Ones.create(4, 4);
        assert!(o.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn van_der_corput_range() {
        for i in 0..1000 {
            let v = van_der_corput(i);
            assert!((0.0..1.0).contains(&v), "vdc({i}) = {v}");
        }
    }

    #[test]
    fn constant_xavier_scale() {
        let init = Constant::xavier_scale(100);
        let w: Vec<f64> = init.create(100, 50);
        let expected = 1.0 / (100.0f64).sqrt();
        assert!((w[0] - expected).abs() < 1e-10);
    }
}
