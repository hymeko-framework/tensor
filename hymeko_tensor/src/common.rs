use std::fmt::{Debug, Display};
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Sub, SubAssign};

pub trait Real:
    Copy
    + PartialOrd
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + Div<Output = Self>
    + Send
    + Sync
    + Debug
    + Display
    + AsF64
    + AsF32
{
    fn zero() -> Self;
    fn one() -> Self;
    fn neg_one() -> Self;
    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn max(self, rhs: Self) -> Self;
    fn min(self, rhs: Self) -> Self;
    fn round(self) -> Self {
        self
    }
    /// Convert from any other real type, if needed (e.g. for mixed-precision).
    fn from_other<T: Real>(rhs: T) -> Self;

    /// |-------------------------
    /// |  Higher order functions
    /// |-------------------------
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn tan(self) -> Self;
    fn cos(self) -> Self;
    fn cosh(self) -> Self;
    fn sin(self) -> Self;
    fn tanh(self) -> Self;
    fn powi(self, n: i32) -> Self;
}

impl Real for f32 {
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn one() -> Self {
        1.0
    }
    #[inline]
    fn neg_one() -> Self {
        -1.0
    }
    #[inline]
    fn abs(self) -> Self {
        self.abs()
    }
    #[inline]
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    #[inline]
    fn max(self, rhs: Self) -> Self {
        self.max(rhs)
    }
    #[inline]
    fn min(self, rhs: Self) -> Self {
        self.min(rhs)
    }
    #[inline]
    fn round(self) -> Self {
        self.round()
    }

    #[inline]
    fn from_other<T: Real>(rhs: T) -> Self {
        rhs.as_f32()
    }

    /// |-------------------------
    /// | Higher-order functions
    /// |-------------------------
    #[inline]
    fn exp(self) -> Self {
        self.exp()
    }
    #[inline]
    fn ln(self) -> Self {
        self.ln()
    }
    #[inline]
    fn tan(self) -> Self {
        self.tan()
    }
    #[inline]
    fn cos(self) -> Self {
        self.cos()
    }
    #[inline]
    fn cosh(self) -> Self {
        self.cosh()
    }
    #[inline]
    fn sin(self) -> Self {
        self.sin()
    }
    #[inline]
    fn tanh(self) -> Self {
        self.tanh()
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        self.powi(n)
    }
}

impl Real for f64 {
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn one() -> Self {
        1.0
    }
    #[inline]
    fn neg_one() -> Self {
        -1.0
    }
    #[inline]
    fn abs(self) -> Self {
        self.abs()
    }
    #[inline]
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    #[inline]
    fn max(self, rhs: Self) -> Self {
        self.max(rhs)
    }
    #[inline]
    fn min(self, rhs: Self) -> Self {
        self.min(rhs)
    }
    #[inline]
    fn round(self) -> Self {
        self.round()
    }
    #[inline]
    fn from_other<T: Real>(rhs: T) -> Self {
        rhs.as_f64()
    }

    /// |------------------------
    /// | Higher-order functions
    /// |------------------------
    #[inline]
    fn exp(self) -> Self {
        self.exp()
    }
    #[inline]
    fn ln(self) -> Self {
        self.ln()
    }
    #[inline]
    fn tan(self) -> Self {
        self.tan()
    }
    #[inline]
    fn cos(self) -> Self {
        self.cos()
    }
    #[inline]
    fn cosh(self) -> Self {
        self.cosh()
    }
    #[inline]
    fn sin(self) -> Self {
        self.sin()
    }
    #[inline]
    fn tanh(self) -> Self {
        self.tanh()
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        self.powi(n)
    }
}

pub trait AsF64: Copy {
    fn as_f64(self) -> f64;
}
impl AsF64 for f32 {
    fn as_f64(self) -> f64 {
        self as f64
    }
}
impl AsF64 for f64 {
    fn as_f64(self) -> f64 {
        self
    }
}

pub trait AsF32: Copy {
    fn as_f32(self) -> f32;
}
impl AsF32 for f32 {
    fn as_f32(self) -> f32 {
        self
    }
}
impl AsF32 for f64 {
    fn as_f32(self) -> f32 {
        self as f32
    }
}

#[inline(always)]
pub fn signed_incidence<F: Real>(sign: i8) -> F {
    match sign {
        1 => F::one(),
        -1 => F::neg_one(),
        _ => F::one(), // neutral: a "szimmetrikus/abs" nézetekben ez oké
    }
}
