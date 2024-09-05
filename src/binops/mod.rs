//! Macros for derivating binary operations in fields and curves.
//! For fields: Addition and multiplication.
//! For curves: Point addition and scalar multiplication.

#[macro_use]
pub(crate) mod add;

#[macro_use]
pub(crate) mod sub;

#[macro_use]
pub(crate) mod mul;

#[macro_export]
/// Implement the group operations, add and sub.
macro_rules! impl_binops_group {
    ($lhs:ident, $rhs:ident, $output:ident) => {
        impl_binops_add!($lhs, $rhs, $output);
        impl_binops_sub!($lhs, $rhs, $output);
    };
    ($lhs:ident, $rhs:ident) => {
        impl_binops_add!($lhs, $rhs);
        impl_binops_sub!($lhs, $rhs);
    };
}
