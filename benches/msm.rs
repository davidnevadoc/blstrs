//! This benchmarks Multi Scalar Multiplication (MSM).
//! Measurement on Bls12-381 G1.
//!
//! To run this benchmark:
//!
//!     cargo bench --bench msm

#[macro_use]
extern crate criterion;

use criterion::{BenchmarkId, Criterion};
use ff::PrimeField;
use group::Group;
use halo2curves::msm::{msm_best, msm_serial};
use halo2curves::CurveAffine;
use rand_core::{RngCore, SeedableRng};
use rand_xorshift::XorShiftRng;
use rayon::current_thread_index;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::time::SystemTime;

const SAMPLE_SIZE: usize = 10;
const SINGLECORE_RANGE: &[u8] = &[3, 8, 10, 12, 14, 16];
// const MULTICORE_RANGE: &[u8] = &[3, 8, 10, 12, 14, 16, 18, 20, 22];
const MULTICORE_RANGE: &[u8] = &[8, 10, 12, 14, 16, 18, 20, 18, 20];
const SEED: [u8; 16] = [
    0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc, 0xe5,
];
const BITS: &[usize] = &[32, 64, 128, 256];

fn generate_curvepoints<C: CurveAffine>(k: u8) -> Vec<C> {
    let n: u64 = {
        assert!(k < 64, "2^64 points maximum.");
        1 << k
    };
    println!("Generating 2^{k} = {n} curve points..",);

    let timer = SystemTime::now();
    let bases = (0..n)
        .into_par_iter()
        .map_init(
            || {
                let mut thread_seed = SEED;
                let uniq = current_thread_index().unwrap().to_ne_bytes();
                assert!(std::mem::size_of::<usize>() == 8);
                for i in 0..uniq.len() {
                    thread_seed[i] += uniq[i];
                    thread_seed[i + 8] += uniq[i];
                }
                XorShiftRng::from_seed(thread_seed)
            },
            |rng, _| <C::CurveExt as Group>::random(rng).into(),
        )
        .collect();
    let end = timer.elapsed().unwrap();
    println!(
        "Generating 2^{k} = {n} curve points took: {} sec.\n\n",
        end.as_secs()
    );
    bases
}

fn generate_coefficients<F: PrimeField>(k: u8, bits: usize) -> Vec<F> {
    let n: u64 = {
        assert!(k < 64, "2^64 scalars maximum.");
        1 << k
    };
    let max_val: Option<u128> = match bits {
        1 => Some(1),
        8 => Some(0xff),
        16 => Some(0xffff),
        32 => Some(0xffff_ffff),
        64 => Some(0xffff_ffff_ffff_ffff),
        128 => Some(0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff),
        256 => None,
        _ => panic!("unexpected bit size {}", bits),
    };

    println!("Generating 2^{k} = {n} coefficients..",);
    let timer = SystemTime::now();
    let coeffs = (0..n)
        .into_par_iter()
        .map_init(
            || {
                let mut thread_seed = SEED;
                let uniq = current_thread_index().unwrap().to_ne_bytes();
                assert!(std::mem::size_of::<usize>() == 8);
                for i in 0..uniq.len() {
                    thread_seed[i] += uniq[i];
                    thread_seed[i + 8] += uniq[i];
                }
                XorShiftRng::from_seed(thread_seed)
            },
            |rng, _| {
                if let Some(max_val) = max_val {
                    let v_lo = rng.next_u64() as u128;
                    let v_hi = rng.next_u64() as u128;
                    let mut v = v_lo + (v_hi << 64);
                    v &= max_val; // Mask the 128bit value to get a lower number of bits
                    F::from_u128(v)
                } else {
                    F::random(rng)
                }
            },
        )
        .collect();
    let end = timer.elapsed().unwrap();
    println!(
        "Generating 2^{k} = {n} coefficients took: {} sec.\n\n",
        end.as_secs()
    );
    coeffs
}

// Generates bases and coefficients for the given ranges and
// bit lenghts.
fn setup<C: CurveAffine>() -> (Vec<C>, Vec<Vec<C::ScalarExt>>) {
    let max_k = *SINGLECORE_RANGE
        .iter()
        .chain(MULTICORE_RANGE.iter())
        .max()
        .unwrap_or(&16);
    assert!(max_k < 64);

    let bases = generate_curvepoints::<C>(max_k);
    let coeffs: Vec<_> = BITS
        .iter()
        .map(|b| generate_coefficients(max_k, *b))
        .collect();

    (bases, coeffs)
}

fn h2c_serial_msm<C: CurveAffine>(c: &mut Criterion) {
    let mut group = c.benchmark_group("halo2curves serial_msm");

    let (bases, coeffs) = setup::<C>();

    for (b_index, b) in BITS.iter().enumerate() {
        for k in SINGLECORE_RANGE {
            // b bits scalar in size k MSM
            let id = format!("{b}b_{k}");
            group
                .bench_function(BenchmarkId::new("singlecore", id), |b| {
                    let n: usize = 1 << k;
                    let mut acc = C::identity().into();
                    b.iter(|| msm_serial(&coeffs[b_index][..n], &bases[..n], &mut acc));
                })
                .sample_size(SAMPLE_SIZE);
        }
    }
    group.finish();
}

fn h2c_parallel_msm<C: CurveAffine>(c: &mut Criterion) {
    let mut group = c.benchmark_group("halo2curves multicore_msm");

    let (bases, coeffs) = setup::<C>();

    for (b_index, b) in BITS.iter().enumerate() {
        for k in MULTICORE_RANGE {
            let id = format!("{b}b_{k}");
            group
                .bench_function(BenchmarkId::new("multicore", id), |b| {
                    let n: usize = 1 << k;
                    b.iter(|| {
                        msm_best(&coeffs[b_index][..n], &bases[..n]);
                    })
                })
                .sample_size(SAMPLE_SIZE);
        }
    }
    group.finish();
}

fn msm_blst(c: &mut Criterion) {
    let mut group = c.benchmark_group("blstrs msm");

    let (bases, coeffs) = setup::<blstrs::G1Affine>();

    for (b_index, b) in BITS.iter().enumerate() {
        // Blstrs version
        for k in MULTICORE_RANGE {
            let id = format!("{b}b_{k}");
            let points: Vec<blstrs::G1Projective> = bases.iter().map(Into::into).collect();
            group
                .bench_function(BenchmarkId::new("blstrs multi_exp", id), |b| {
                    let n: usize = 1 << k;
                    b.iter(|| blstrs::G1Projective::multi_exp(&points[..n], &coeffs[b_index][..n]))
                })
                .sample_size(SAMPLE_SIZE);
        }
    }
    group.finish();

    let mut group = c.benchmark_group("halo2curves multicore_msm");

    for (b_index, b) in BITS.iter().enumerate() {
        for k in MULTICORE_RANGE {
            let id = format!("{b}b_{k}");
            group
                .bench_function(BenchmarkId::new("multicore", id), |b| {
                    let n: usize = 1 << k;
                    b.iter(|| {
                        msm_best(&coeffs[b_index][..n], &bases[..n]);
                    })
                })
                .sample_size(SAMPLE_SIZE);
        }
    }
    group.finish();
}

// fn msm_h2c(c: &mut Criterion) {
//     h2c_parallel_msm::<blstrs::G1Affine>(c)
// }

criterion_group!(benches, msm_blst);
criterion_main!(benches);
