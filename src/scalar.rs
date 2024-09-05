use core::convert::TryInto;
use ff::{FieldBits, PrimeFieldBits};
use halo2derive::impl_field;
use rand::RngCore;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

impl_field!(
    bls12381_scalar,
    Scalar,
    modulus = "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001",
    mul_gen = "7",
    zeta = "ac45a4010001a40200000000ffffffff",
    from_uniform = [64],
    endian = "little",
);

halo2curves::extend_field_legendre!(Scalar);
halo2curves::impl_binops_calls!(Scalar);
halo2curves::impl_binops_additive!(Scalar, Scalar);
halo2curves::impl_binops_multiplicative!(Scalar, Scalar);
halo2curves::field_bits!(Scalar);
halo2curves::serialize_deserialize_primefield!(Scalar);
halo2curves::impl_from_u64!(Scalar);
halo2curves::impl_from_bool!(Scalar);

impl PrimeFieldBits for Scalar {
    type ReprBits = [u64; 4];

    fn to_le_bits(&self) -> FieldBits<Self::ReprBits> {
        todo!()
    }

    fn char_le_bits() -> FieldBits<Self::ReprBits> {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    crate::field_testing_suite!(Scalar, "field_arithmetic");
    crate::field_testing_suite!(Scalar, "conversion");
    crate::field_testing_suite!(Scalar, "serialization");
    crate::field_testing_suite!(Scalar, "quadratic_residue");
    crate::field_testing_suite!(Scalar, "bits");
    crate::field_testing_suite!(Scalar, "serialization_check");
    crate::field_testing_suite!(Scalar, "constants");
    crate::field_testing_suite!(Scalar, "sqrt");
    crate::field_testing_suite!(Scalar, "zeta");
    crate::field_testing_suite!(Scalar, "from_uniform_bytes", 64);
}
