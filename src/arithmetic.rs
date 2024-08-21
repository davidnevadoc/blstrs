//! This module provides common utilities, traits and structures for group and
//! field arithmetic.

/// Compute a + b + carry, returning the result and the new carry over.
#[inline(always)]
pub(crate) const fn adc(a: u64, b: u64, carry: u64) -> (u64, u64) {
    let ret = (a as u128) + (b as u128) + (carry as u128);
    (ret as u64, (ret >> 64) as u64)
}

/// Compute a - (b + borrow), returning the result and the new borrow.
#[inline(always)]
pub(crate) const fn sbb(a: u64, b: u64, borrow: u64) -> (u64, u64) {
    let ret = (a as u128).wrapping_sub((b as u128) + ((borrow >> 63) as u128));
    (ret as u64, (ret >> 64) as u64)
}

/// Compute a + (b * c) + carry, returning the result and the new carry over.
#[inline(always)]
pub(crate) const fn mac(a: u64, b: u64, c: u64, carry: u64) -> (u64, u64) {
    let ret = (a as u128) + ((b as u128) * (c as u128)) + (carry as u128);
    (ret as u64, (ret >> 64) as u64)
}

/// Compute a + (b * c), returning the result and the new carry over.
#[inline(always)]
pub(crate) const fn macx(a: u64, b: u64, c: u64) -> (u64, u64) {
    let res = (a as u128) + ((b as u128) * (c as u128));
    (res as u64, (res >> 64) as u64)
}

/// Compute a * b, returning the result in the form of 14 limbs of u64.
#[inline(always)]
pub(crate) fn mul_896(a: [u64; 7], b: [u64; 7]) -> [u64; 14] {
    let (r0, carry) = macx(0, a[0], b[0]);
    let (r1, carry) = macx(carry, a[0], b[1]);
    let (r2, carry) = macx(carry, a[0], b[2]);
    let (r3, carry) = macx(carry, a[0], b[3]);
    let (r4, carry) = macx(carry, a[0], b[4]);
    let (r5, carry) = macx(carry, a[0], b[5]);
    let (r6, carry_out) = macx(carry, a[0], b[6]);

    let (r1, carry) = macx(r1, a[1], b[0]);
    let (r2, carry) = mac(r2, a[1], b[1], carry);
    let (r3, carry) = mac(r3, a[1], b[2], carry);
    let (r4, carry) = mac(r4, a[1], b[3], carry);
    let (r5, carry) = mac(r5, a[1], b[4], carry);
    let (r6, carry) = mac(r6, a[1], b[5], carry);
    let (r7, carry_out) = mac(carry_out, a[1], b[6], carry);

    let (r2, carry) = macx(r2, a[2], b[0]);
    let (r3, carry) = mac(r3, a[2], b[1], carry);
    let (r4, carry) = mac(r4, a[2], b[2], carry);
    let (r5, carry) = mac(r5, a[2], b[3], carry);
    let (r6, carry) = mac(r6, a[2], b[4], carry);
    let (r7, carry) = mac(r7, a[2], b[5], carry);
    let (r8, carry_out) = mac(carry_out, a[2], b[6], carry);

    let (r3, carry) = macx(r3, a[3], b[0]);
    let (r4, carry) = mac(r4, a[3], b[1], carry);
    let (r5, carry) = mac(r5, a[3], b[2], carry);
    let (r6, carry) = mac(r6, a[3], b[3], carry);
    let (r7, carry) = mac(r7, a[3], b[4], carry);
    let (r8, carry) = mac(r8, a[3], b[5], carry);
    let (r9, carry_out) = mac(carry_out, a[3], b[6], carry);

    let (r4, carry) = macx(r4, a[4], b[0]);
    let (r5, carry) = mac(r5, a[4], b[1], carry);
    let (r6, carry) = mac(r6, a[4], b[2], carry);
    let (r7, carry) = mac(r7, a[4], b[3], carry);
    let (r8, carry) = mac(r8, a[4], b[4], carry);
    let (r9, carry) = mac(r9, a[4], b[5], carry);
    let (r10, carry_out) = mac(carry_out, a[4], b[6], carry);

    let (r5, carry) = macx(r5, a[5], b[0]);
    let (r6, carry) = mac(r6, a[5], b[1], carry);
    let (r7, carry) = mac(r7, a[5], b[2], carry);
    let (r8, carry) = mac(r8, a[5], b[3], carry);
    let (r9, carry) = mac(r9, a[5], b[4], carry);
    let (r10, carry) = mac(r10, a[5], b[5], carry);
    let (r11, carry_out) = mac(carry_out, a[5], b[6], carry);

    let (r6, carry) = macx(r6, a[6], b[0]);
    let (r7, carry) = mac(r7, a[6], b[1], carry);
    let (r8, carry) = mac(r8, a[6], b[2], carry);
    let (r9, carry) = mac(r9, a[6], b[3], carry);
    let (r10, carry) = mac(r10, a[6], b[4], carry);
    let (r11, carry) = mac(r11, a[6], b[5], carry);
    let (r12, carry_out) = mac(carry_out, a[6], b[6], carry);

    [
        r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, carry_out,
    ]
}
