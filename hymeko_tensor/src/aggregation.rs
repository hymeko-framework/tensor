use crate::common::Real;

#[derive(Clone, Copy, Debug)]
pub enum SignAgg {
    PreferNonNeutral, // a mostani logika
    WeightedVote,     // később
    Channels3,        // később: + / - / 0 külön csatorna
}

#[derive(Clone, Copy, Debug)]
pub enum WeightAgg {
    Sum,
    Max,
    ProbSum01,  // 1 - Π(1-w)  (feltételez [0,1])
    LukasSat01, // min(1, a+b)
}

#[derive(Clone, Copy, Debug)]
pub struct AggCfg {
    pub sign: SignAgg,
    pub weight: WeightAgg,
    pub clamp01: bool, // ha ProbSum/Lukas, érdemes
}

#[inline(always)]
pub fn agg_sign<F: Real>(cfg: &AggCfg, a: i8, b: i8, wa: F, wb: F) -> i8 {
    match cfg.sign {
        SignAgg::PreferNonNeutral => {
            if a == b {
                return a;
            }
            if a == 0 {
                return b;
            }
            if b == 0 {
                return a;
            }
            0
        }
        SignAgg::WeightedVote => {
            let s = (a as f64) * wa.as_f64() + (b as f64) * wb.as_f64();
            if s > 0.0 {
                1
            } else if s < 0.0 {
                -1
            } else {
                0
            }
        }
        SignAgg::Channels3 => {
            // ezt majd nem i8-ban tároljuk, hanem 3 csatornában.
            // itt most fallback:
            if a == b {
                a
            } else {
                0
            }
        }
    }
}

#[inline(always)]
pub fn clamp01<T: Real>(x: T) -> T {
    x.max(T::zero()).min(T::one())
}

#[inline(always)]
pub fn agg_weight<T: Real>(cfg: &AggCfg, a: T, b: T) -> T {
    let mut out = match cfg.weight {
        WeightAgg::Sum => a + b,
        WeightAgg::Max => a.max(b),
        WeightAgg::ProbSum01 => T::one() - (T::one() - a) * (T::one() - b),
        WeightAgg::LukasSat01 => (a + b).min(T::one()),
    };
    if cfg.clamp01 {
        out = clamp01(out);
    }
    out
}
