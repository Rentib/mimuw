pub struct Fibonacci {
    // Add here any fields you need.
    a: u128,
    b: u128,
    z: bool,
}

impl Fibonacci {
    /// Create new `Fibonacci`.
    pub fn new() -> Fibonacci {
        Fibonacci {
            a: 0,
            b: 1,
            z: true,
        }
    }

    /// Calculate the n-th Fibonacci number.
    ///
    /// This shall not change the state of the iterator.
    /// The calculations shall wrap around at the boundary of u8.
    /// The calculations might be slow (recursive calculations are acceptable).
    pub fn fibonacci(n: usize) -> u8 {
        let mut a = 0u8;
        let mut b = 1u8;
        for _ in 0..n {
            a = a.wrapping_add(b);
            std::mem::swap(&mut a, &mut b);
        }
        a
    }
}

impl Iterator for Fibonacci {
    type Item = u128;

    /// Calculate the next Fibonacci number.
    ///
    /// The first call to `next()` shall return the 0th Fibonacci number (i.e., `0`).
    /// The calculations shall not overflow and shall not wrap around. If the result
    /// doesn't fit u128, the sequence shall end (the iterator shall return `None`).
    /// The calculations shall be fast (recursive calculations are **un**acceptable).
    fn next(&mut self) -> Option<Self::Item> {
        if self.z {
            self.z = false;
            return Some(0);
        }
        self.a = self.a.checked_add(self.b)?;
        std::mem::swap(&mut self.a, &mut self.b);
        Some(self.a)
    }
}
